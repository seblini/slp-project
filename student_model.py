"""
Student model for AV-HuBERT knowledge distillation.

Uses ResNet-18 (pretrained on ImageNet) as the visual feature extractor,
modified to accept grayscale input. Video-only — learns from a teacher
that has access to both audio and video.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


class ResNet18Frontend(nn.Module):
    """
    ResNet-18 pretrained on ImageNet, modified for grayscale mouth ROI frames.
    Input: (batch, num_frames, 1, 88, 88)
    Output: (batch, num_frames, embed_dim)
    """
    def __init__(self, embed_dim=256, freeze_early=True):
        super().__init__()
        resnet = models.resnet18(pretrained=True)

        # Adapt first conv layer from 3-channel RGB to 1-channel grayscale
        # Average the pretrained weights across input channels to retain
        # learned edge/texture features rather than reinitializing randomly
        pretrained_weight = resnet.conv1.weight.data
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data = pretrained_weight.mean(dim=1, keepdim=True)

        # Keep everything except the final FC classification layer
        self.features = nn.Sequential(
            resnet.conv1,       # -> 64 channels
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,      # -> 64 channels
            resnet.layer2,      # -> 128 channels
            resnet.layer3,      # -> 256 channels
            resnet.layer4,      # -> 512 channels
            resnet.avgpool,     # -> 512 x 1 x 1
        )

        # Optionally freeze early layers (conv1, bn1, layer1, layer2)
        # These learn generic visual features that transfer well and
        # don't need fine-tuning, which saves memory and speeds up training
        if freeze_early:
            for name, param in self.features.named_parameters():
                # Freeze everything before layer3 (index 6 in the Sequential)
                if not name.startswith('6.') and not name.startswith('7.'):
                    param.requires_grad = False

        # Project ResNet's 512-dim output to our model's embed_dim
        self.proj = nn.Linear(512, embed_dim)

    def forward(self, x):
        """
        x: (batch, num_frames, 1, H, W)
        returns: (batch, num_frames, embed_dim)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)       # (B*T, 512, 1, 1)
        x = x.view(B * T, -1)      # (B*T, 512)
        x = self.proj(x)           # (B*T, embed_dim)
        x = x.view(B, T, -1)       # (B, T, embed_dim)
        return x


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


class StudentLipReader(nn.Module):
    """
    Video-only seq2seq model for lip reading with ResNet-18 backbone.

    Args:
        vocab_size: must match the teacher's vocabulary (1000 for AV-HuBERT)
        embed_dim: hidden dimension for transformer layers
        encoder_layers: number of transformer encoder layers
        decoder_layers: number of transformer decoder layers
        n_heads: number of attention heads
        ff_dim: feedforward dimension in transformer
        dropout: dropout rate
        pad_idx: padding token index from the teacher's dictionary
        freeze_early_resnet: freeze conv1 through layer2 of ResNet-18
    """
    def __init__(
        self,
        vocab_size=1000,
        embed_dim=256,
        encoder_layers=4,
        decoder_layers=4,
        n_heads=4,
        ff_dim=512,
        dropout=0.1,
        pad_idx=1,
        freeze_early_resnet=True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx

        # Visual feature extraction (ResNet-18)
        self.visual_frontend = ResNet18Frontend(
            embed_dim=embed_dim,
            freeze_early=freeze_early_resnet,
        )

        # Temporal transformer encoder
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        # Autoregressive transformer decoder
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_decoder = PositionalEncoding(embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Output projection to vocabulary
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def _make_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def encode(self, video_frames):
        """
        video_frames: (batch, num_frames, 1, H, W)
        returns: (batch, num_frames, embed_dim)
        """
        x = self.visual_frontend(video_frames)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        return x

    def decode(self, encoder_out, prev_tokens):
        """
        encoder_out: (batch, num_frames, embed_dim)
        prev_tokens: (batch, seq_len) token ids
        returns: logits (batch, seq_len, vocab_size)
        """
        x = self.token_embedding(prev_tokens)
        x = self.pos_decoder(x)
        causal_mask = self._make_causal_mask(prev_tokens.size(1), prev_tokens.device)
        x = self.decoder(x, encoder_out, tgt_mask=causal_mask)
        logits = self.output_proj(x)
        return logits

    def forward(self, video_frames, prev_tokens):
        """
        Full forward pass.
        video_frames: (batch, num_frames, 1, H, W)
        prev_tokens: (batch, seq_len)
        returns: logits (batch, seq_len, vocab_size)
        """
        encoder_out = self.encode(video_frames)
        logits = self.decode(encoder_out, prev_tokens)
        return logits

    def count_parameters(self, only_trainable=False):
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def print_parameter_breakdown(self):
        """Print parameter counts per component."""
        components = {
            'ResNet-18 frontend': self.visual_frontend,
            '  resnet features': self.visual_frontend.features,
            '  projection (512->embed)': self.visual_frontend.proj,
            'Transformer encoder (4 layers)': self.encoder,
            'Token embedding': self.token_embedding,
            'Transformer decoder (4 layers)': self.decoder,
            'Output projection': self.output_proj,
        }
        print(f"{'Component':<35} {'Total':>12} {'Trainable':>12}")
        print("-" * 61)
        for name, module in components.items():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name:<35} {total:>12,} {trainable:>12,}")
        print("-" * 61)
        total = self.count_parameters(only_trainable=False)
        trainable = self.count_parameters(only_trainable=True)
        frozen = total - trainable
        print(f"{'TOTAL':<35} {total:>12,} {trainable:>12,}")
        print(f"{'FROZEN':<35} {'':>12} {frozen:>12,}")


class DistillationTrainer:
    """
    Handles the distillation loss computation.

    Args:
        student: StudentLipReader model
        temperature: softens the teacher distribution (higher = softer)
        alpha: weight for soft loss vs hard loss (0 = all hard, 1 = all soft)
        lr: learning rate
    """
    def __init__(self, student, temperature=2.0, alpha=0.7, lr=1e-3):
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        # Only optimize trainable parameters (respects frozen ResNet layers)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, student.parameters()),
            lr=lr,
        )

    def compute_loss(self, student_logits, teacher_soft_targets, hard_targets):
        """
        student_logits: (batch, seq_len, vocab_size) raw logits from student
        teacher_soft_targets: (batch, seq_len, vocab_size) probabilities from teacher
        hard_targets: (batch, seq_len) ground truth token ids

        Returns: combined loss, soft_loss, hard_loss
        """
        T = self.temperature

        # Soft loss: KL divergence between student and teacher distributions
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        soft_loss = F.kl_div(
            student_log_probs,
            teacher_soft_targets,
            reduction='batchmean',
        ) * (T ** 2)

        # Hard loss: cross entropy with ground truth
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            hard_targets.view(-1),
            ignore_index=self.student.pad_idx,
        )

        combined = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return combined, soft_loss, hard_loss

    def train_step(self, video_frames, prev_tokens, teacher_soft_targets, hard_targets):
        """
        Single training step.
        Returns: dict with loss values
        """
        self.student.train()
        self.optimizer.zero_grad()

        student_logits = self.student(video_frames, prev_tokens)
        loss, soft_loss, hard_loss = self.compute_loss(
            student_logits, teacher_soft_targets, hard_targets
        )

        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'soft_loss': soft_loss.item(),
            'hard_loss': hard_loss.item(),
        }
