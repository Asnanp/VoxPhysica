"""
VocalMorph V3 — Unified Height Estimation Model
=================================================

Nuclear redesign for 2-3cm MAE on all height ranges.

Architecture:
  - Multi-scale acoustic frontend with SE calibration
  - Deep Conformer encoder (configurable depth/width) with stochastic depth
  - Hierarchical multi-resolution pooling across encoder layers
  - Multi-token physics cross-attention fusion
  - Height-focused regression tower with Wing + Huber + ordinal ranking + isometric loss
  - Height-range adaptive calibration for short/medium/tall speakers
  - Auxiliary height-bin classification head with finer bins
  - Lightweight gender auxiliary head
  - Strong regularization: DropPath, MC Dropout, feature mixup, label smoothing

Key improvements over V3.0:
  - Hierarchical pooling: aggregates representations from multiple encoder layers
  - Multi-token physics: projects physics scalars into multiple tokens for richer fusion
  - Wing loss: specifically designed for precise regression around small errors
  - Adaptive height calibration: learns bin-specific offsets to fix short/tall bias
  - Deeper regression tower with 4 SwiGLU blocks
  - 5 height bins instead of 3 for finer structural supervision

Author: Asnan P
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────
# Building Blocks
# ────────────────────────────────────────────────────────────


class DropPath(nn.Module):
    """Stochastic depth — drops entire residual branches during training."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


class MCDropout(nn.Dropout):
    """Dropout that stays active during inference for MC uncertainty."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, self.p, training=True, inplace=self.inplace)


class RelativePositionalEncoding(nn.Module):
    """Learnable relative positional bias for self-attention."""

    def __init__(self, max_len: int = 2048, n_heads: int = 8):
        super().__init__()
        self.max_len = max_len
        self.n_heads = n_heads
        self.bias = nn.Embedding(2 * max_len - 1, n_heads)

    def forward(self, length: int) -> torch.Tensor:
        positions = torch.arange(length, device=self.bias.weight.device)
        relative = positions.unsqueeze(0) - positions.unsqueeze(1) + self.max_len - 1
        relative = relative.clamp(0, 2 * self.max_len - 2)
        return self.bias(relative).permute(2, 0, 1)  # (n_heads, T, T)


class ConformerConvModule(nn.Module):
    """Conformer convolution module with GLU gating and depthwise conv."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Linear(d_model, 2 * d_model)
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=padding, groups=d_model, bias=False
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.ln(x)
        x = self.pointwise1(x)
        x = x.chunk(2, dim=-1)
        x = x[0] * torch.sigmoid(x[1])  # GLU
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.activation(x)
        x = x.transpose(1, 2)  # (B, T, D)
        x = self.pointwise2(x)
        return self.dropout(x)


class ConformerFeedForward(nn.Module):
    """Conformer feed-forward module with expansion and residual."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)


class ConformerBlock(nn.Module):
    """Single Conformer block: FF -> MHSA -> Conv -> FF (with half-step FFN)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        ff_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.ff1 = ConformerFeedForward(d_model, ff_expansion, dropout)
        self.attn_ln = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(d_model, conv_kernel, dropout)
        self.ff2 = ConformerFeedForward(d_model, ff_expansion, dropout)
        self.final_ln = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.size(0)
        # Half-step FFN
        x = x + 0.5 * self.drop_path(self.ff1(x))
        # MHSA
        residual = x
        x_norm = self.attn_ln(x)
        # Expand attn_bias from (n_heads, T, T) to (B*n_heads, T, T)
        expanded_bias = None
        if attn_bias is not None:
            expanded_bias = attn_bias.unsqueeze(0).expand(B, -1, -1, -1)
            expanded_bias = expanded_bias.reshape(B * self.mhsa.num_heads, attn_bias.size(1), attn_bias.size(2))
        attn_out, _ = self.mhsa(
            x_norm, x_norm, x_norm,
            key_padding_mask=padding_mask,
            attn_mask=expanded_bias,
        )
        x = residual + self.drop_path(self.attn_dropout(attn_out))
        # Conv module
        x = x + self.drop_path(self.conv(x))
        # Half-step FFN
        x = x + 0.5 * self.drop_path(self.ff2(x))
        return self.final_ln(x)


class MultiHeadAttentiveStatsPooling(nn.Module):
    """Multi-head attentive statistics pooling for sequence summary."""

    def __init__(self, d_model: int, n_heads: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.n_heads = n_heads
        self.attention = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_heads),
        )
        self.proj = nn.Linear(d_model * 2 * n_heads, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape

        # Compute attention weights: (B, T, n_heads)
        attn_logits = self.attention(x)

        if padding_mask is not None:
            attn_logits = attn_logits.masked_fill(
                padding_mask.unsqueeze(-1), float("-inf")
            )

        attn_weights = F.softmax(attn_logits, dim=1)  # (B, T, n_heads)

        stats = []
        for h in range(self.n_heads):
            w = attn_weights[:, :, h].unsqueeze(-1)  # (B, T, 1)
            weighted_mean = (x * w).sum(dim=1)  # (B, D)
            weighted_var = ((x - weighted_mean.unsqueeze(1)) ** 2 * w).sum(dim=1)
            weighted_std = (weighted_var + 1e-8).sqrt()
            stats.extend([weighted_mean, weighted_std])

        pooled = torch.cat(stats, dim=-1)  # (B, D * 2 * n_heads)
        return self.ln(self.proj(pooled))


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class PhysicsCrossAttention(nn.Module):
    """Multi-token cross-attention for physics features.

    Projects the 11 raw physics scalars into multiple key-value tokens so the
    model can learn which physics signals matter most for height prediction.
    Each physics feature gets its own token for fine-grained attention.
    """

    def __init__(self, d_model: int, phys_dim: int = 11, n_heads: int = 4,
                 n_tokens: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_tokens = n_tokens
        self.phys_proj = nn.Sequential(
            nn.Linear(phys_dim, d_model * n_tokens),
            nn.GELU(),
            nn.Linear(d_model * n_tokens, d_model * n_tokens),
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, embedding: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        # embedding: (B, D), physics: (B, phys_dim)
        B = embedding.size(0)
        D = embedding.size(1)
        phys_kv = self.phys_proj(physics).view(B, self.n_tokens, D)  # (B, n_tokens, D)
        q = self.ln_q(embedding).unsqueeze(1)  # (B, 1, D)
        kv = self.ln_kv(phys_kv)

        attn_out, _ = self.cross_attn(q, kv, kv)  # (B, 1, D)
        attn_out = attn_out.squeeze(1)  # (B, D)

        fused = embedding + torch.sigmoid(self.gate) * attn_out
        fused = fused + self.ffn(fused)
        return fused


class HierarchicalPooling(nn.Module):
    """Pools representations from multiple encoder layers via learned gating."""

    def __init__(self, d_model: int, n_layers: int, pool_heads: int = 4,
                 pool_hidden: int = 128):
        super().__init__()
        self.n_layers = n_layers
        self.layer_weights = nn.Parameter(torch.ones(n_layers) / n_layers)
        self.pooling = MultiHeadAttentiveStatsPooling(
            d_model, n_heads=pool_heads, hidden_dim=pool_hidden,
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self, layer_outputs: List[torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weights = F.softmax(self.layer_weights, dim=0)
        combined = torch.zeros_like(layer_outputs[-1])
        for i, layer_out in enumerate(layer_outputs):
            combined = combined + weights[i] * layer_out
        combined = self.ln(combined)
        return self.pooling(combined, padding_mask=padding_mask)


class HeightRegressionTower(nn.Module):
    """Deep SwiGLU regression tower with 4 residual blocks."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.15):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.ln_in = nn.LayerNorm(hidden_dim)

        self.block1 = SwiGLU(hidden_dim, hidden_dim * 2)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.drop1 = MCDropout(dropout)

        self.block2 = SwiGLU(hidden_dim, hidden_dim * 2)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.drop2 = MCDropout(dropout)

        self.block3 = SwiGLU(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.drop3 = MCDropout(dropout * 0.7)

        self.block4 = SwiGLU(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.drop4 = MCDropout(dropout * 0.5)

        self.proj_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.ln_in(self.proj_in(x))

        residual = feat
        feat = self.drop1(self.ln1(self.block1(feat)))
        feat = feat + residual

        residual = feat
        feat = self.drop2(self.ln2(self.block2(feat)))
        feat = feat + residual

        residual = feat
        feat = self.drop3(self.ln3(self.block3(feat)))
        feat = feat + residual

        residual = feat
        feat = self.drop4(self.ln4(self.block4(feat)))
        feat = feat + residual

        return self.proj_out(feat).squeeze(-1)


class HeightBinHead(nn.Module):
    """Auxiliary height-bin classification for structural regularization.

    Default 5 bins: very_short (<155), short (155-165), medium (165-175),
    tall (175-185), very_tall (>185).
    """

    def __init__(self, in_dim: int, n_bins: int = 5, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, n_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class HeightCalibrationHead(nn.Module):
    """Learns bin-specific offsets to fix systematic bias in short/tall ranges.

    Predicts a small additive correction conditioned on the bin logits, so the
    model can learn that short speakers are systematically over-predicted, etc.
    """

    def __init__(self, embedding_dim: int, n_bins: int = 5, dropout: float = 0.1):
        super().__init__()
        self.bin_embed = nn.Embedding(n_bins, 64)
        self.correction_net = nn.Sequential(
            nn.Linear(embedding_dim + 64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, embedding: torch.Tensor, bin_logits: torch.Tensor) -> torch.Tensor:
        bin_probs = F.softmax(bin_logits, dim=-1)  # (B, n_bins)
        # Weighted sum of bin embeddings
        bin_features = torch.matmul(bin_probs, self.bin_embed.weight)  # (B, 64)
        combined = torch.cat([embedding, bin_features], dim=-1)
        correction = self.correction_net(combined).squeeze(-1)
        return correction * self.scale


class GenderHead(nn.Module):
    """Lightweight gender classification head."""

    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ────────────────────────────────────────────────────────────
# V3 Model
# ────────────────────────────────────────────────────────────


class MultiScaleAcousticFrontend(nn.Module):
    """
    Biological multi-scale feature extractor.
    Analyses short (plosives), medium (formants), and long (prosody/VTL)
    temporal features dynamically before passing them to the Conformer.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.15):
        super().__init__()
        # Parallel convolutions for different biological time-scales
        self.conv_short = nn.Conv1d(in_dim, out_dim // 4, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(in_dim, out_dim // 4, kernel_size=7, padding=3)
        self.conv_long_a = nn.Conv1d(in_dim, out_dim // 4, kernel_size=15, padding=7)
        self.conv_long_b = nn.Conv1d(in_dim, out_dim // 4, kernel_size=25, padding=12)

        # Squeeze-and-Excitation element (Channel-wise biometric attention)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_dim, out_dim // 4, 1),
            nn.GELU(),
            nn.Conv1d(out_dim // 4, out_dim, 1),
            nn.Sigmoid()
        )

        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, T, in_dim)
        x_t = x.transpose(1, 2)  # (B, in_dim, T)

        # Multi-scale extraction
        x_short = self.conv_short(x_t)
        x_med = self.conv_medium(x_t)
        x_long_a = self.conv_long_a(x_t)
        x_long_b = self.conv_long_b(x_t)

        feat = torch.cat([x_short, x_med, x_long_a, x_long_b], dim=1)  # (B, out_dim, T)

        # Squeeze-and-Excitation calibration
        scale = self.se(feat)
        feat = feat * scale

        feat = feat.transpose(1, 2)  # (B, T, out_dim)
        return self.proj(feat)


class VocalMorphV3(nn.Module):
    """
    Unified height estimation model.

    No short/tall segmentation, no domain adversarial.
    Raw acoustic power + physics cross-attention focused entirely on height.
    """

    def __init__(
        self,
        input_dim: int = 136,
        d_model: int = 384,
        n_heads: int = 8,
        n_blocks: int = 6,
        ff_expansion: int = 4,
        conv_kernel: int = 15,
        dropout: float = 0.22,
        drop_path: float = 0.12,
        pool_heads: int = 4,
        pool_hidden: int = 192,
        n_height_bins: int = 5,
        use_physics_cross_attn: bool = True,
        use_height_bin_aux: bool = True,
        use_feature_mixup: bool = True,
        mixup_alpha: float = 0.2,
        use_hierarchical_pooling: bool = True,
        use_height_calibration: bool = True,
        physics_tokens: int = 4,
        tower_hidden_dim: int = 320,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_feature_mixup = use_feature_mixup
        self.mixup_alpha = mixup_alpha
        self.use_physics_cross_attn = use_physics_cross_attn
        self.use_height_bin_aux = use_height_bin_aux
        self.use_hierarchical_pooling = use_hierarchical_pooling
        self.use_height_calibration = use_height_calibration
        self.n_blocks = n_blocks

        self.linguistic_dim = input_dim - 11

        self.input_norm = nn.LayerNorm(self.linguistic_dim)
        self.input_proj = MultiScaleAcousticFrontend(
            in_dim=self.linguistic_dim, out_dim=d_model, dropout=dropout
        )
        self.input_ln = nn.LayerNorm(d_model)

        self.phys_norm = nn.LayerNorm(11)

        # Relative positional encoding
        self.rel_pos = RelativePositionalEncoding(max_len=2048, n_heads=n_heads)

        # Conformer encoder stack
        dp_rates = [drop_path * (i / max(n_blocks - 1, 1)) for i in range(n_blocks)]
        self.encoder = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_expansion=ff_expansion,
                conv_kernel=conv_kernel,
                dropout=dropout,
                drop_path=dp_rates[i],
            )
            for i in range(n_blocks)
        ])

        # Pooling
        if use_hierarchical_pooling:
            self.pooling = HierarchicalPooling(
                d_model, n_layers=n_blocks,
                pool_heads=pool_heads, pool_hidden=pool_hidden,
            )
        else:
            self.pooling = MultiHeadAttentiveStatsPooling(
                d_model, n_heads=pool_heads, hidden_dim=pool_hidden
            )

        # Physics cross-attention fusion
        if use_physics_cross_attn:
            self.physics_cross_attn = PhysicsCrossAttention(
                d_model, phys_dim=11, n_heads=4, n_tokens=physics_tokens,
                dropout=dropout * 0.5,
            )

        # Task heads
        self.gender_head = GenderHead(d_model, dropout=dropout * 0.5)

        # Height head: d_model + 2 (gender logits)
        tower_dim = d_model + 2
        if not use_physics_cross_attn:
            tower_dim = d_model + 13  # fallback: concat physics + gender
        self.height_head = HeightRegressionTower(
            tower_dim, hidden_dim=tower_hidden_dim, dropout=dropout,
        )

        # Auxiliary height-bin head
        if use_height_bin_aux:
            self.height_bin_head = HeightBinHead(
                d_model, n_bins=n_height_bins, dropout=dropout * 0.5,
            )

        # Height calibration head
        if use_height_calibration and use_height_bin_aux:
            self.height_calibration = HeightCalibrationHead(
                d_model, n_bins=n_height_bins, dropout=dropout * 0.5,
            )

        # Weight initialization
        self._init_weights()

        # Report
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[VocalMorphV3] Trainable parameters: {n_params:,}")
        print(f"[VocalMorphV3] d_model={d_model}, n_blocks={n_blocks}, n_heads={n_heads}")
        print(f"[VocalMorphV3] dropout={dropout}, drop_path={drop_path}")
        print(f"[VocalMorphV3] physics_cross_attn={use_physics_cross_attn}, height_bin_aux={use_height_bin_aux}")
        print(f"[VocalMorphV3] feature_mixup={use_feature_mixup}, mixup_alpha={mixup_alpha}")
        print(f"[VocalMorphV3] hierarchical_pooling={use_hierarchical_pooling}")
        print(f"[VocalMorphV3] height_calibration={use_height_calibration}")
        print(f"[VocalMorphV3] tower_hidden_dim={tower_hidden_dim}, physics_tokens={physics_tokens}")
        print(f"[VocalMorphV3] n_height_bins={n_height_bins}")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_feature_mixup(
        self,
        features: torch.Tensor,
        height: Optional[torch.Tensor] = None,
        gender: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float]:
        """Apply mixup augmentation on features and targets during training."""
        if not self.training or not self.use_feature_mixup:
            return features, height, gender, 1.0

        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
        lam = max(lam, 1.0 - lam)

        B = features.size(0)
        perm = torch.randperm(B, device=features.device)

        mixed_features = lam * features + (1.0 - lam) * features[perm]
        mixed_height = None
        if height is not None:
            mixed_height = lam * height + (1.0 - lam) * height[perm]
        mixed_gender = gender  # keep original for classification

        return mixed_features, mixed_height, mixed_gender, lam

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        height_targets: Optional[torch.Tensor] = None,
        gender_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, T, D) acoustic feature sequences
            padding_mask: (B, T) True where padded
            height_targets: optional (B,) for mixup during training
            gender_targets: optional (B,) for mixup during training

        Returns:
            dict with 'height', 'gender_logits', 'embedding', optionally 'height_bin_logits',
            'mixed_height_targets', 'mixup_lambda'
        """
        B, T, D = features.shape

        # 1. Biological Splitting (Biometric Bypass)
        features_ling = features[:, :, :-11]
        features_phys = features[:, :, -11:]

        # Apply feature mixup if in training
        mixup_lam = 1.0
        mixed_height = height_targets
        if self.training and self.use_feature_mixup and height_targets is not None:
            features_ling, mixed_height, gender_targets, mixup_lam = self._apply_feature_mixup(
                features_ling, height_targets, gender_targets,
            )

        # Input normalization + projection
        x = self.input_norm(features_ling)
        x = self.input_proj(x)
        x = self.input_ln(x)

        # Relative positional bias
        attn_bias = self.rel_pos(T)  # (n_heads, T, T)

        # Conformer encoder with layer output collection
        layer_outputs = []
        for block in self.encoder:
            x = block(x, padding_mask=padding_mask, attn_bias=attn_bias)
            layer_outputs.append(x)

        # Pooling -> embedding
        if self.use_hierarchical_pooling:
            embedding = self.pooling(layer_outputs, padding_mask=padding_mask)
        else:
            embedding = self.pooling(x, padding_mask=padding_mask)

        # Physics pooling
        if padding_mask is not None:
            phys_mask = (~padding_mask).unsqueeze(-1).float()
            physics_pooled = (features_phys * phys_mask).sum(dim=1) / (phys_mask.sum(dim=1) + 1e-9)
        else:
            physics_pooled = features_phys.mean(dim=1)
        physics_pooled = self.phys_norm(physics_pooled)

        # Physics cross-attention fusion
        if self.use_physics_cross_attn:
            embedding_fused = self.physics_cross_attn(embedding, physics_pooled)
        else:
            embedding_fused = embedding

        # Gender prediction
        gender_logits = self.gender_head(embedding_fused)

        # Height regression
        if self.use_physics_cross_attn:
            hybrid_embedding = torch.cat([embedding_fused, gender_logits], dim=-1)
        else:
            hybrid_embedding = torch.cat([embedding_fused, physics_pooled, gender_logits], dim=-1)

        height_preds = self.height_head(hybrid_embedding)

        result: Dict[str, torch.Tensor] = {
            "height": height_preds,
            "gender_logits": gender_logits,
            "embedding": embedding,
        }

        # Auxiliary height-bin classification
        if self.use_height_bin_aux:
            bin_logits = self.height_bin_head(embedding_fused)
            result["height_bin_logits"] = bin_logits

            # Height calibration: additive correction based on bin predictions
            if self.use_height_calibration:
                calibration = self.height_calibration(embedding_fused, bin_logits)
                result["height"] = height_preds + calibration
                result["height_uncalibrated"] = height_preds
                result["height_calibration"] = calibration

        # Pass mixed targets back for loss computation
        if mixed_height is not None:
            result["mixed_height_targets"] = mixed_height
        result["mixup_lambda"] = torch.tensor(mixup_lam, device=features.device)

        return result

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        n_samples: int = 30,
    ) -> Dict[str, torch.Tensor]:
        """MC dropout uncertainty estimation."""
        self.train()  # Keep dropout active
        preds = []
        for _ in range(n_samples):
            out = self.forward(features, padding_mask)
            preds.append(out["height"].unsqueeze(0))

        self.eval()
        stacked = torch.cat(preds, dim=0)  # (n_samples, B)
        return {
            "height_mean": stacked.mean(0),
            "height_std": stacked.std(0),
            "height_samples": stacked,
        }


# ────────────────────────────────────────────────────────────
# Loss Function
# ────────────────────────────────────────────────────────────


class V3HeightLoss(nn.Module):
    """
    Height-focused loss combining:
      - Wing loss for precise regression near zero error
      - Huber (smooth L1) for stable gradient on outliers
      - Ordinal ranking for correct height ordering
      - Isometric embedding regularization
      - Auxiliary height-bin classification
      - Gender classification
      - Calibration regularization
    """

    def __init__(
        self,
        huber_delta: float = 0.3,
        huber_weight: float = 0.5,
        wing_weight: float = 0.5,
        wing_width: float = 5.0,
        wing_curvature: float = 0.5,
        ranking_weight: float = 0.25,
        ranking_margin: float = 0.15,
        isometric_weight: float = 0.08,
        height_bin_weight: float = 0.2,
        gender_weight: float = 0.1,
        label_smoothing: float = 0.05,
        calibration_reg_weight: float = 0.01,
    ):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.huber_delta = huber_delta
        self.huber_weight = huber_weight
        self.wing_weight = wing_weight
        self.wing_width = wing_width
        self.wing_curvature = wing_curvature
        self.ranking_weight = ranking_weight
        self.ranking_margin = ranking_margin
        self.isometric_weight = isometric_weight
        self.height_bin_weight = height_bin_weight
        self.gender_weight = gender_weight
        self.label_smoothing = label_smoothing
        self.calibration_reg_weight = calibration_reg_weight

    def _wing_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Wing loss: log-based loss for small errors, linear for large errors.

        More sensitive to small errors than L1/Huber, which helps push MAE
        below 3cm. From "Wing Loss for Robust Facial Landmark Localisation
        with Convolutional Neural Networks" (Feng et al., 2018).
        """
        diff = (pred - target).abs()
        w = self.wing_width
        eps = self.wing_curvature
        C = w - w * math.log(1.0 + w / eps)
        loss = torch.where(
            diff < w,
            w * torch.log(1.0 + diff / eps),
            diff - C,
        )
        return loss.mean()

    def _ordinal_ranking_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        B = pred.size(0)
        if B < 2:
            return torch.tensor(0.0, device=pred.device)

        diff_pred = pred.unsqueeze(0) - pred.unsqueeze(1)
        diff_true = target.unsqueeze(0) - target.unsqueeze(1)

        min_diff = 0.15
        valid = diff_true.abs() > min_diff

        sign = diff_true.sign()
        loss = F.relu(self.ranking_margin - sign * diff_pred)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        return loss[valid].mean()

    def _isometric_loss(self, embedding: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Euclidean embedding distance should be proportional to height difference."""
        B = embedding.size(0)
        if B < 2:
            return torch.tensor(0.0, device=embedding.device)

        feat_diff = embedding.unsqueeze(0) - embedding.unsqueeze(1)  # (B, B, D)
        feat_dist = torch.norm(feat_diff, p=2, dim=-1)  # (B, B)

        true_dist = (target.unsqueeze(0) - target.unsqueeze(1)).abs()  # (B, B)

        feat_dist_norm = feat_dist / (feat_dist.mean() + 1e-8)
        true_dist_norm = true_dist / (true_dist.mean() + 1e-8)

        return F.l1_loss(feat_dist_norm, true_dist_norm)

    def _height_to_bin(self, height_cm: torch.Tensor, n_bins: int = 5) -> torch.Tensor:
        """Convert raw height in cm to bin labels."""
        bins = torch.zeros_like(height_cm, dtype=torch.long)
        if n_bins == 5:
            bins[height_cm >= 155.0] = 1
            bins[height_cm >= 165.0] = 2
            bins[height_cm >= 175.0] = 3
            bins[height_cm >= 185.0] = 4
        elif n_bins == 3:
            bins[height_cm >= 165.0] = 1
            bins[height_cm >= 175.0] = 2
        else:
            # Generic evenly-spaced bins from 145 to 195
            bin_edges = torch.linspace(145, 195, n_bins + 1, device=height_cm.device)
            for i in range(1, n_bins):
                bins[height_cm >= bin_edges[i]] = i
        return bins

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        height_pred = preds["height"]
        height_target = targets["height"]
        embedding = preds.get("embedding")

        # Force float32 for loss computation
        height_pred = height_pred.float()
        height_target = height_target.float()

        # 1. Huber loss
        huber = F.huber_loss(height_pred, height_target, delta=self.huber_delta)

        # 2. Wing loss
        wing = torch.tensor(0.0, device=huber.device)
        if self.wing_weight > 0.0:
            wing = self._wing_loss(height_pred, height_target)

        l_primary = huber * self.huber_weight + wing * self.wing_weight

        # 3. Ordinal ranking loss
        l_ranking = torch.tensor(0.0, device=l_primary.device)
        if self.ranking_weight > 0.0:
            l_ranking = self._ordinal_ranking_loss(height_pred, height_target)

        # 4. Isometric embedding loss
        l_iso = torch.tensor(0.0, device=l_primary.device)
        if self.isometric_weight > 0.0 and embedding is not None:
            l_iso = self._isometric_loss(embedding.float(), height_target)

        # 5. Gender classification loss
        gender_pred = preds.get("gender_logits")
        gender_target = targets.get("gender")
        l_gender = torch.tensor(0.0, device=l_primary.device)
        if gender_pred is not None and gender_target is not None:
            l_gender = F.cross_entropy(
                gender_pred,
                gender_target.long(),
                label_smoothing=self.label_smoothing
            )

        # 6. Height-bin auxiliary loss
        l_bin = torch.tensor(0.0, device=l_primary.device)
        height_bin_logits = preds.get("height_bin_logits")
        height_cm_target = targets.get("height_cm")
        if height_bin_logits is not None and height_cm_target is not None:
            n_bins = height_bin_logits.size(-1)
            bin_labels = self._height_to_bin(height_cm_target.float(), n_bins=n_bins)
            l_bin = F.cross_entropy(height_bin_logits, bin_labels, label_smoothing=0.1)

        # 7. Calibration regularization
        l_calib_reg = torch.tensor(0.0, device=l_primary.device)
        calibration = preds.get("height_calibration")
        if calibration is not None and self.calibration_reg_weight > 0.0:
            l_calib_reg = calibration.float().pow(2).mean()

        # Total
        total = (
            l_primary
            + self.ranking_weight * l_ranking
            + self.isometric_weight * l_iso
            + self.gender_weight * l_gender
            + self.height_bin_weight * l_bin
            + self.calibration_reg_weight * l_calib_reg
        )

        # Track standard L1 for monitoring
        err = (height_pred - height_target).abs()

        return {
            "total": total,
            "height_l1": err.mean(),
            "height_huber": huber,
            "height_wing": wing,
            "height_ranking": l_ranking,
            "height_iso": l_iso,
            "height_bin_ce": l_bin,
            "gender_ce": l_gender,
            "calibration_reg": l_calib_reg,
        }


# ────────────────────────────────────────────────────────────
# Builder
# ────────────────────────────────────────────────────────────


def build_v3_model(config: dict) -> VocalMorphV3:
    """Build VocalMorphV3 from config dict."""
    model_cfg = config.get("model", {}).get("v3", {})
    input_dim = int(config.get("model", {}).get("input_dim", 136))

    return VocalMorphV3(
        input_dim=input_dim,
        d_model=int(model_cfg.get("d_model", 384)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        n_blocks=int(model_cfg.get("n_blocks", 6)),
        ff_expansion=int(model_cfg.get("ff_expansion", 4)),
        conv_kernel=int(model_cfg.get("conv_kernel", 15)),
        dropout=float(model_cfg.get("dropout", 0.22)),
        drop_path=float(model_cfg.get("drop_path", 0.12)),
        pool_heads=int(model_cfg.get("pool_heads", 4)),
        pool_hidden=int(model_cfg.get("pool_hidden", 192)),
        n_height_bins=int(model_cfg.get("n_height_bins", 5)),
        use_physics_cross_attn=bool(model_cfg.get("use_physics_cross_attn", True)),
        use_height_bin_aux=bool(model_cfg.get("use_height_bin_aux", True)),
        use_feature_mixup=bool(model_cfg.get("use_feature_mixup", True)),
        mixup_alpha=float(model_cfg.get("mixup_alpha", 0.2)),
        use_hierarchical_pooling=bool(model_cfg.get("use_hierarchical_pooling", True)),
        use_height_calibration=bool(model_cfg.get("use_height_calibration", True)),
        physics_tokens=int(model_cfg.get("physics_tokens", 4)),
        tower_hidden_dim=int(model_cfg.get("tower_hidden_dim", 320)),
    )


def build_v3_loss(config: dict) -> V3HeightLoss:
    """Build V3 loss function from config."""
    loss_cfg = config.get("training", {}).get("loss", {})
    return V3HeightLoss(
        huber_delta=float(loss_cfg.get("huber_delta", 0.3)),
        huber_weight=float(loss_cfg.get("huber_weight", 0.5)),
        wing_weight=float(loss_cfg.get("wing_weight", 0.5)),
        wing_width=float(loss_cfg.get("wing_width", 5.0)),
        wing_curvature=float(loss_cfg.get("wing_curvature", 0.5)),
        ranking_weight=float(loss_cfg.get("ranking_weight", 0.25)),
        ranking_margin=float(loss_cfg.get("ranking_margin", 0.15)),
        isometric_weight=float(loss_cfg.get("isometric_weight", 0.08)),
        height_bin_weight=float(loss_cfg.get("height_bin_weight", 0.2)),
        gender_weight=float(loss_cfg.get("gender_weight", 0.1)),
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.05)),
        calibration_reg_weight=float(loss_cfg.get("calibration_reg_weight", 0.01)),
    )
