import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


class ResNet18Frontend(nn.Module):
    """Pretrained ResNet-18 adapted for grayscale, with optional early-layer freezing."""
    def __init__(self, embed_dim=256, freeze_early=False):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Adapt conv1 from 3-channel to 1-channel by averaging weights
        pretrained_w = resnet.conv1.weight.data
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data = pretrained_w.mean(dim=1, keepdim=True)
        
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool,
        )
        
        if freeze_early:
            # Freeze conv1 through layer2; keep layer3 and layer4 trainable
            for name, p in self.features.named_parameters():
                if not (name.startswith('6.') or name.startswith('7.')):
                    p.requires_grad = False
        
        self.proj = nn.Linear(512, embed_dim)
    
    def forward(self, x):
        # x: (B, T, 1, H, W) → (B, T, embed_dim)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x).flatten(1)  # (B*T, 512)
        x = self.proj(x)
        return x.view(B, T, -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class VideoStudent(nn.Module):
    """Pretrained ResNet-18 + transformer encoder/decoder. Video-only."""
    def __init__(self, vocab_size=1000, dim=256, enc_layers=4, dec_layers=4,
                 n_heads=4, ff_dim=1024, dropout=0.1,
                 pad_id=1, bos_id=0, eos_id=2, freeze_early=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        
        self.frontend = ResNet18Frontend(embed_dim=dim, freeze_early=freeze_early)
        self.enc_pos = PositionalEncoding(dim)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        
        self.tok_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.dec_pos = PositionalEncoding(dim)
        
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)
        
        self.out_proj = nn.Linear(dim, vocab_size)
    
    def _causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    
    def encode(self, video, padding_mask):
        """video: (B, T, 1, 96, 96), padding_mask: (B, T) True=padded"""
        x = self.frontend(video)
        x = self.enc_pos(x)
        return self.encoder(x, src_key_padding_mask=padding_mask)
    
    def decode(self, encoder_out, prev_tokens, encoder_padding_mask=None,
               decoder_padding_mask=None):
        x = self.tok_emb(prev_tokens)
        x = self.dec_pos(x)
        causal = self._causal_mask(prev_tokens.size(1), x.device)
        x = self.decoder(
            tgt=x, memory=encoder_out,
            tgt_mask=causal,
            tgt_key_padding_mask=decoder_padding_mask,
            memory_key_padding_mask=encoder_padding_mask,
        )
        return self.out_proj(x)
    
    def forward(self, video, video_mask, prev_tokens, decoder_mask=None):
        enc = self.encode(video, padding_mask=video_mask)
        return self.decode(enc, prev_tokens,
                           encoder_padding_mask=video_mask,
                           decoder_padding_mask=decoder_mask)
    
    @torch.no_grad()
    def greedy_decode(self, video, video_mask, max_len=20):
        device = video.device
        enc = self.encode(video, padding_mask=video_mask)
        B = enc.shape[0]
        tokens = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for _ in range(max_len):
            logits = self.decode(enc, tokens, encoder_padding_mask=video_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1)
            tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == self.eos_id)
            if finished.all():
                break
        return tokens[:, 1:]
    
    def count_parameters(self, only_trainable=False):
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
