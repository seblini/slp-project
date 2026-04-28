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

    @torch.no_grad()
    def beam_search_decode(self, video, video_mask, beam_size=5, max_len=20,
                           length_penalty=1.0):
        """
        Standard beam search with length normalization.
        
        Args:
            video: (B, T, 1, H, W)
            video_mask: (B, T)
            beam_size: number of beams to keep at each step
            max_len: max decode length
            length_penalty: 1.0 = no penalty, >1 favors longer, <1 favors shorter
        
        Returns:
            best_tokens: (B, T_dec) — top-1 hypothesis per sample
        """
        device = video.device
        B = video.shape[0]
        K = beam_size
        
        # Encode once, expand for each beam
        enc = self.encode(video, padding_mask=video_mask)  # (B, T_v, D)
        enc = enc.unsqueeze(1).expand(-1, K, -1, -1).reshape(B*K, *enc.shape[1:])
        enc_mask = video_mask.unsqueeze(1).expand(-1, K, -1).reshape(B*K, -1)
        
        # Initialize beams: each starts with BOS, score 0
        tokens = torch.full((B*K, 1), self.bos_id, dtype=torch.long, device=device)
        scores = torch.zeros(B, K, device=device)
        # Force only first beam per sample to be "alive"; others get -inf
        # so duplicate beams don't all expand from BOS in step 0
        scores[:, 1:] = float('-inf')
        
        finished = torch.zeros(B, K, dtype=torch.bool, device=device)
        finished_scores = torch.full((B, K), float('-inf'), device=device)
        finished_seqs = [[None] * K for _ in range(B)]
        
        for step in range(max_len):
            # Decode current step for all B*K beams
            logits = self.decode(enc, tokens, encoder_padding_mask=enc_mask)
            next_logits = logits[:, -1, :]  # (B*K, V)
            log_probs = torch.log_softmax(next_logits, dim=-1)  # (B*K, V)
            
            # Combine with cumulative beam score
            log_probs = log_probs.view(B, K, -1)
            cum_scores = scores.unsqueeze(-1) + log_probs  # (B, K, V)
            
            # Mask finished beams: only allow them to "produce" PAD with their current score
            # so their cumulative score doesn't change
            if finished.any():
                fin_mask = finished.unsqueeze(-1)  # (B, K, 1)
                cum_scores = cum_scores.masked_fill(fin_mask, float('-inf'))
                # For finished beams, set the PAD-token score to the existing beam score
                pad_scores = cum_scores[..., self.pad_id]  # (B, K)
                pad_scores = torch.where(finished, scores, pad_scores)
                cum_scores[..., self.pad_id] = pad_scores
            
            # Pick top K from K*V candidates per sample
            flat = cum_scores.view(B, -1)  # (B, K*V)
            top_scores, top_idx = flat.topk(K, dim=-1)  # (B, K)
            
            beam_idx = top_idx // log_probs.shape[-1]   # which beam (0..K-1)
            token_idx = top_idx % log_probs.shape[-1]    # which token (0..V-1)
            
            # Reorder tokens by chosen beams, append new token
            tokens = tokens.view(B, K, -1)
            tokens = torch.gather(tokens, 1,
                                  beam_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
            tokens = torch.cat([tokens, token_idx.unsqueeze(-1)], dim=-1)
            tokens = tokens.view(B*K, -1)
            
            # Reorder finished mask
            finished = torch.gather(finished, 1, beam_idx)
            scores = top_scores
            
            # Mark new EOS as finished
            new_eos = (token_idx == self.eos_id) & ~finished
            for b in range(B):
                for k in range(K):
                    if new_eos[b, k]:
                        seq = tokens.view(B, K, -1)[b, k].tolist()
                        # Length-normalize the score
                        norm = ((step + 1) ** length_penalty)
                        final_score = scores[b, k].item() / norm
                        if final_score > finished_scores[b, k].item():
                            finished_scores[b, k] = final_score
                            finished_seqs[b][k] = seq
            finished = finished | new_eos
            
            if finished.all():
                break
        
        # For any beam that never finished, use its current sequence
        tokens_view = tokens.view(B, K, -1)
        for b in range(B):
            for k in range(K):
                if finished_seqs[b][k] is None:
                    norm = ((tokens_view.shape[-1]) ** length_penalty)
                    finished_scores[b, k] = scores[b, k].item() / norm
                    finished_seqs[b][k] = tokens_view[b, k].tolist()
        
        # Pick best beam per sample
        best_idx = finished_scores.argmax(dim=-1)  # (B,)
        best_tokens = []
        for b in range(B):
            seq = finished_seqs[b][best_idx[b].item()]
            # Drop BOS
            seq = seq[1:]
            best_tokens.append(torch.tensor(seq, dtype=torch.long))
        
        # Pad to same length for batching
        max_T = max(t.shape[0] for t in best_tokens)
        out = torch.full((B, max_T), self.pad_id, dtype=torch.long, device=device)
        for b, t in enumerate(best_tokens):
            out[b, :t.shape[0]] = t.to(device)
        return out
