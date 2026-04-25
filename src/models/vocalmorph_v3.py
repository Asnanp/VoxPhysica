"""
VocalMorph V3 — Unified Height Estimation Model
=================================================

Nuclear redesign for 2-3cm MAE on all height ranges.

Architecture:
  - Deep Conformer encoder (6 blocks, d_model=512, 8 heads)
  - Multi-head attentive statistics pooling
  - Height-focused regression tower with Huber + ordinal ranking loss
  - Lightweight gender auxiliary head
  - Strong regularization: DropPath, MC Dropout, label smoothing

Key differences from V2:
  - No physics path, no reliability tower, no domain adversarial
  - No separate short/tall handling — unified training
  - Much deeper and wider backbone (6 blocks × 512d vs 1 block × 64d)
  - Height-only focus (95%+ gradient signal to height)
  - Ordinal ranking loss for correct height ordering

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
    """Single Conformer block: FF → MHSA → Conv → FF (with half-step FFN)."""

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
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class HeightRegressionTower(nn.Module):
    """Deep SwiGLU Regression tower for extreme absolute sub-centimeter scale refinement."""

    def __init__(self, in_dim: int, dropout: float = 0.15):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, 256)
        
        self.blocks = nn.Sequential(
            SwiGLU(256, 1024),
            nn.LayerNorm(256),
            MCDropout(dropout),
            
            SwiGLU(256, 512),
            nn.LayerNorm(256),
            MCDropout(dropout),
            
            SwiGLU(256, 256),
            nn.LayerNorm(256),
            MCDropout(dropout * 0.5),
        )
        self.proj_out = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.proj_in(x)
        feat = self.blocks(feat)
        return self.proj_out(feat).squeeze(-1)


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
    Analyses short (plosives), medium (formants), and long (prosody/VTL) temporal features dynamically
    before passing them to the Conformer.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.15):
        super().__init__()
        # Parallel convolutions for different biological time-scales
        self.conv_short = nn.Conv1d(in_dim, out_dim // 4, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(in_dim, out_dim // 2, kernel_size=11, padding=5)
        self.conv_long = nn.Conv1d(in_dim, out_dim // 4, kernel_size=25, padding=12)
        
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
        x_long = self.conv_long(x_t)
        
        feat = torch.cat([x_short, x_med, x_long], dim=1)  # (B, out_dim, T)
        
        # Squeeze-and-Excitation calibration
        scale = self.se(feat)
        feat = feat * scale
        
        feat = feat.transpose(1, 2)  # (B, T, out_dim)
        return self.proj(feat)


class VocalMorphV3(nn.Module):
    """
    Unified height estimation model.

    No short/tall segmentation, no physics path, no domain adversarial.
    Just raw acoustic power focused entirely on height.
    """

    def __init__(
        self,
        input_dim: int = 136,
        d_model: int = 512,
        n_heads: int = 8,
        n_blocks: int = 6,
        ff_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.15,
        drop_path: float = 0.1,
        pool_heads: int = 4,
        pool_hidden: int = 256,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection (We leave the last 11 purely physical dims out of the Conformer noise)
        self.linguistic_dim = input_dim - 11
        
        self.input_norm = nn.LayerNorm(self.linguistic_dim)
        self.input_proj = MultiScaleAcousticFrontend(
            in_dim=self.linguistic_dim, out_dim=d_model, dropout=dropout
        )
        self.input_ln = nn.LayerNorm(d_model)
        
        # Scaling wrapper for the 11 raw physics features to prevent FP16 nan overflow
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
        self.pooling = MultiHeadAttentiveStatsPooling(
            d_model, n_heads=pool_heads, hidden_dim=pool_hidden
        )

        # Task heads
        self.gender_head = GenderHead(d_model, dropout=dropout * 0.5)
        
        # Gender+Physics Conditioned SwiGLU: d_model + 11 (physics bypass) + 2 (gender logits)
        tower_dim = d_model + 13
        self.height_head = HeightRegressionTower(tower_dim, dropout=dropout)

        # Weight initialization
        self._init_weights()

        # Report
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[VocalMorphV3] Trainable parameters: {n_params:,}")
        print(f"[VocalMorphV3] d_model={d_model}, n_blocks={n_blocks}, n_heads={n_heads}")
        print(f"[VocalMorphV3] dropout={dropout}, drop_path={drop_path}")

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

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, T, D) acoustic feature sequences
            padding_mask: (B, T) True where padded

        Returns:
            dict with 'height', 'gender_logits'
        """
        B, T, D = features.shape

        # 1. Biological Splitting (Biometric Bypass)
        # First (D - 11) are MFCCs/Spectral. Last 11 are raw Praat Physics.
        features_ling = features[:, :, :-11]
        features_phys = features[:, :, -11:]
        
        # Input normalization + projection (Linguistic Noise only)
        x = self.input_norm(features_ling)
        x = self.input_proj(x)
        x = self.input_ln(x)

        # Relative positional bias
        attn_bias = self.rel_pos(T)  # (n_heads, T, T)

        # Conformer encoder
        for block in self.encoder:
            x = block(x, padding_mask=padding_mask, attn_bias=attn_bias)

        # Pooling → embedding
        embedding = self.pooling(x, padding_mask=padding_mask)

        # 2. Gender Manifold Extraction
        gender_logits = self.gender_head(embedding)

        # 3. Biometric Bypass Re-injection
        # Extract the pure temporal mean of the VTL / Pitch physics
        # Mask out padded zeroes when doing the mean
        if padding_mask is not None:
            phys_mask = (~padding_mask).unsqueeze(-1).float()
            physics_pooled = (features_phys * phys_mask).sum(dim=1) / (phys_mask.sum(dim=1) + 1e-9)
        else:
            physics_pooled = features_phys.mean(dim=1)
            
        # Safely scale the raw values into a stable [-1, 1] variance margin for FP16
        physics_pooled = self.phys_norm(physics_pooled)

        # 4. Hybrid Synthesis (Conformer Audio + Raw Physics + Gender Condition)
        hybrid_embedding = torch.cat([embedding, physics_pooled, gender_logits], dim=-1)

        # Task heads
        height_preds = self.height_head(hybrid_embedding)

        return {
            "height": height_preds,
            "gender_logits": gender_logits,
            "embedding": embedding,
        }

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
    Height-focused loss combining Pure L1 MAE + Ordinal Ranking + Isometric Latent Distance.
    """

    def __init__(
        self,
        huber_delta: float = 0.02,
        huber_weight: float = 1.0,
        ranking_weight: float = 0.3,
        ranking_margin: float = 0.15,
        gender_weight: float = 0.1,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.huber_delta = huber_delta
        self.huber_weight = huber_weight
        self.ranking_weight = ranking_weight
        self.ranking_margin = ranking_margin
        self.gender_weight = gender_weight
        self.label_smoothing = label_smoothing

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
        """Physical Isometry: Euclidean embedding distance must linearly map to height difference."""
        B = embedding.size(0)
        if B < 2:
            return torch.tensor(0.0, device=embedding.device)
        
        # Calculate pairwise Euclidean embedding distance
        feat_diff = embedding.unsqueeze(0) - embedding.unsqueeze(1)  # (B, B, D)
        feat_dist = torch.norm(feat_diff, p=2, dim=-1)  # (B, B)
        
        # Calculate pairwise true height absolute difference
        true_dist = (target.unsqueeze(0) - target.unsqueeze(1)).abs()  # (B, B)
        
        # We want feat_dist to be proportional to true_dist.
        # So absolute error between normalized distance matrices
        feat_dist_norm = feat_dist / (feat_dist.mean() + 1e-8)
        true_dist_norm = true_dist / (true_dist.mean() + 1e-8)
        
        return F.l1_loss(feat_dist_norm, true_dist_norm)

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        height_pred = preds["height"]
        height_target = targets["height"]
        embedding = preds.get("embedding")

        # Force loss variables to float32 to prevent AMP Float16 gradient explosion near zero
        height_pred = height_pred.float()
        height_target = height_target.float()

        # 1. Calculate raw absolute error
        err = (height_pred - height_target).abs()
        
        # 2. Huber Loss (Smooth L1) for precise targeting and stability
        huber = F.huber_loss(height_pred, height_target, delta=self.huber_delta)
        
        # 3. Blended Loss Mapping
        l_primary = huber * self.huber_weight
        # 4. Gender Classification Loss (Cross Entropy with Label Smoothing)
        gender_pred = preds.get("gender_logits")
        gender_target = targets.get("gender")
        
        l_gender = torch.tensor(0.0, device=l_primary.device)
        if gender_pred is not None and gender_target is not None:
            l_gender = F.cross_entropy(
                gender_pred, 
                gender_target.long(), 
                label_smoothing=self.label_smoothing
            )
            
        # Total Loss Pipeline
        total = l_primary + (l_gender * self.gender_weight)

        return {
            "total": total,
            "height_l1": err.mean(),  # just track standard L1 strictly in the dict
            "height_mse": torch.tensor(0.0, device=total.device),
            "height_iso": torch.tensor(0.0, device=total.device),
            "gender_ce": l_gender,
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
        d_model=int(model_cfg.get("d_model", 512)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        n_blocks=int(model_cfg.get("n_blocks", 6)),
        ff_expansion=int(model_cfg.get("ff_expansion", 4)),
        conv_kernel=int(model_cfg.get("conv_kernel", 31)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        drop_path=float(model_cfg.get("drop_path", 0.1)),
        pool_heads=int(model_cfg.get("pool_heads", 4)),
        pool_hidden=int(model_cfg.get("pool_hidden", 256)),
    )


def build_v3_loss(config: dict) -> V3HeightLoss:
    """Build V3 loss function from config."""
    loss_cfg = config.get("training", {}).get("loss", {})
    return V3HeightLoss(
        huber_delta=float(loss_cfg.get("huber_delta", 0.02)),
        huber_weight=float(loss_cfg.get("huber_weight", 1.0)),
        ranking_weight=float(loss_cfg.get("ranking_weight", 0.3)),
        ranking_margin=float(loss_cfg.get("ranking_margin", 0.15)),
        gender_weight=float(loss_cfg.get("gender_weight", 0.1)),
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.05)),
    )
