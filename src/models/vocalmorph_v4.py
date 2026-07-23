"""
VocalMorph V4 — SSL-Powered Height Estimation Model
=====================================================

Architecture:
  - Pretrained wav2vec2-base / HuBERT backbone (768-dim frame features)
  - Weighted layer combination across SSL transformer layers
  - 2-block Conformer adapter (768→512)
  - Attentive statistics pooling
  - Gender-conditioned height regression tower (2 SwiGLU blocks)
  - Clean Huber + MAE + Gender CE loss

Key design principles:
  - Let the SSL backbone do the heavy lifting (12 pretrained transformer layers)
  - Keep the adapter/head lightweight to avoid overfitting on 775 speakers
  - Simple, stable loss — no competing objectives
  - Supports both pre-extracted features and end-to-end training

Target: 3cm MAE on speaker-level height estimation.

Author: Asnan P
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────
# Building Blocks
# ────────────────────────────────────────────────────────────


class MCDropout(nn.Dropout):
    """Dropout that stays active during inference for MC uncertainty."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, self.p, training=True, inplace=self.inplace)


class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU-gated linear unit."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class ConformerConvModule(nn.Module):
    """Conformer convolution module with GLU gating."""

    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1):
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
        x = self.ln(x)
        x = self.pointwise1(x)
        x = x.chunk(2, dim=-1)
        x = x[0] * torch.sigmoid(x[1])
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        x = self.pointwise2(x)
        return self.dropout(x)


class ConformerFeedForward(nn.Module):
    """Conformer feed-forward with expansion."""

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


class ConformerAdapterBlock(nn.Module):
    """Lightweight Conformer block for adapting SSL features."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        ff_expansion: int = 4,
        conv_kernel: int = 15,
        dropout: float = 0.1,
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

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Half-step FFN
        x = x + 0.5 * self.ff1(x)
        # MHSA
        residual = x
        x_norm = self.attn_ln(x)
        attn_out, _ = self.mhsa(
            x_norm, x_norm, x_norm, key_padding_mask=padding_mask
        )
        x = residual + self.attn_dropout(attn_out)
        # Conv
        x = x + self.conv(x)
        # Half-step FFN
        x = x + 0.5 * self.ff2(x)
        return self.final_ln(x)


class AttentiveStatsPooling(nn.Module):
    """Attentive statistics pooling — produces mean + std summary."""

    def __init__(self, d_model: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.proj = nn.Linear(d_model * 2, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: (B, T, D)
        attn_logits = self.attention(x)  # (B, T, 1)
        if padding_mask is not None:
            attn_logits = attn_logits.masked_fill(
                padding_mask.unsqueeze(-1), float("-inf")
            )
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, T, 1)

        weighted_mean = (x * attn_weights).sum(dim=1)  # (B, D)
        weighted_var = ((x - weighted_mean.unsqueeze(1)) ** 2 * attn_weights).sum(dim=1)
        weighted_std = (weighted_var + 1e-8).sqrt()

        pooled = torch.cat([weighted_mean, weighted_std], dim=-1)  # (B, 2D)
        return self.ln(self.proj(pooled))


# ────────────────────────────────────────────────────────────
# SSL Backbone
# ────────────────────────────────────────────────────────────


class SSLBackbone(nn.Module):
    """
    Pretrained wav2vec2-base feature extractor.
    
    Extracts frame-level features from multiple transformer layers
    and combines them with learned weights.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        n_layers_to_use: int = 4,
        freeze: bool = True,
        output_dim: int = 768,
    ):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.ssl = Wav2Vec2Model.from_pretrained(model_name)
        self.output_dim = output_dim
        self.n_layers_to_use = n_layers_to_use

        # Learned weights for combining layers
        self.layer_weights = nn.Parameter(
            torch.ones(n_layers_to_use) / n_layers_to_use
        )

        if freeze:
            self.freeze()

    def freeze(self):
        """Freeze all SSL parameters."""
        for param in self.ssl.parameters():
            param.requires_grad = False

    def unfreeze_top_layers(self, n_layers: int = 2):
        """Unfreeze top N transformer layers for fine-tuning."""
        # First freeze everything
        self.freeze()
        # Then unfreeze the top N layers
        total_layers = len(self.ssl.encoder.layers)
        for i in range(total_layers - n_layers, total_layers):
            for param in self.ssl.encoder.layers[i].parameters():
                param.requires_grad = True
        n_unfrozen = sum(p.requires_grad for p in self.ssl.parameters())
        print(f"[SSLBackbone] Unfroze top {n_layers} layers ({n_unfrozen:,} params)")

    @torch.no_grad()
    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract features without gradient tracking (for pre-extraction)."""
        return self._forward_impl(waveforms)

    def _forward_impl(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: (B, num_samples) raw audio at 16kHz

        Returns:
            (B, T, 768) frame-level features
        """
        outputs = self.ssl(
            waveforms,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get last N hidden states
        hidden_states = outputs.hidden_states  # tuple of (B, T, 768)
        selected = hidden_states[-self.n_layers_to_use:]

        # Weighted combination
        weights = F.softmax(self.layer_weights, dim=0)
        combined = torch.zeros_like(selected[0])
        for i, hs in enumerate(selected):
            combined = combined + weights[i] * hs

        return combined

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(waveforms)


# ────────────────────────────────────────────────────────────
# V4 Model
# ────────────────────────────────────────────────────────────


class VocalMorphV4(nn.Module):
    """
    VocalMorph V4 — SSL-powered height estimation.

    Can operate in two modes:
    1. Pre-extracted features: input is (B, T, ssl_dim) from pre-extracted .npz
    2. End-to-end: input is raw waveform, passed through SSL backbone

    Architecture: SSL → Adapter → Pool → Gender → Height
    """

    def __init__(
        self,
        input_dim: int = 768,
        adapter_dim: int = 512,
        adapter_heads: int = 8,
        adapter_blocks: int = 2,
        adapter_ff_expansion: int = 4,
        adapter_conv_kernel: int = 15,
        dropout: float = 0.15,
        pool_hidden: int = 128,
        tower_hidden: int = 256,
        meta_dim: int = 0,
        meta_hidden: int = 64,
        height_bin_classes: int = 0,
        height_calibration_scale: float = 0.0,
        height_expert_count: int = 0,
        height_expert_scale: float = 0.0,
        use_ssl_backbone: bool = False,
        ssl_model_name: str = "facebook/wav2vec2-base",
        ssl_freeze: bool = True,
        ssl_layers_to_use: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.adapter_dim = adapter_dim
        self.meta_dim = max(0, int(meta_dim))
        self.height_bin_classes = max(0, int(height_bin_classes))
        self.height_calibration_scale = max(0.0, float(height_calibration_scale))
        self.height_expert_count = max(0, int(height_expert_count))
        self.height_expert_scale = max(0.0, float(height_expert_scale))
        self.use_ssl_backbone = use_ssl_backbone

        # Optional SSL backbone
        if use_ssl_backbone:
            self.ssl_backbone = SSLBackbone(
                model_name=ssl_model_name,
                n_layers_to_use=ssl_layers_to_use,
                freeze=ssl_freeze,
                output_dim=input_dim,
            )

        # Input projection: SSL dim → adapter dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, adapter_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
        )

        # Conformer adapter blocks (lightweight — SSL already has 12 layers)
        self.adapter = nn.ModuleList([
            ConformerAdapterBlock(
                d_model=adapter_dim,
                n_heads=adapter_heads,
                ff_expansion=adapter_ff_expansion,
                conv_kernel=adapter_conv_kernel,
                dropout=dropout,
            )
            for _ in range(adapter_blocks)
        ])

        # Attentive statistics pooling
        self.pooling = AttentiveStatsPooling(adapter_dim, hidden_dim=pool_hidden)

        # Gender head (auxiliary)
        self.gender_head = nn.Sequential(
            nn.Linear(adapter_dim, 64),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2),
        )

        if self.meta_dim > 0:
            self.meta_encoder = nn.Sequential(
                nn.LayerNorm(self.meta_dim),
                nn.Linear(self.meta_dim, meta_hidden),
                nn.SiLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(meta_hidden, meta_hidden),
                nn.SiLU(),
            )
            meta_out_dim = int(meta_hidden)
        else:
            self.meta_encoder = None
            meta_out_dim = 0

        # Height regression tower
        # Input: pooled voice embedding + predicted gender logits + biological cues.
        tower_input = adapter_dim + 2 + meta_out_dim
        self.height_tower = nn.Sequential(
            nn.Linear(tower_input, tower_hidden),
            nn.LayerNorm(tower_hidden),
        )
        self.height_block1 = SwiGLU(tower_hidden, tower_hidden * 2)
        self.height_ln1 = nn.LayerNorm(tower_hidden)
        self.height_drop1 = MCDropout(dropout)

        self.height_block2 = SwiGLU(tower_hidden, tower_hidden)
        self.height_ln2 = nn.LayerNorm(tower_hidden)
        self.height_drop2 = MCDropout(dropout * 0.7)

        self.height_out = nn.Linear(tower_hidden, 1)
        if self.height_expert_count > 1 and self.height_expert_scale > 0.0:
            expert_hidden = max(64, tower_hidden // 2)
            self.height_expert_gate = nn.Sequential(
                nn.Linear(tower_hidden, expert_hidden),
                nn.SiLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(expert_hidden, self.height_expert_count),
            )
            self.height_experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(tower_hidden, expert_hidden),
                        nn.SiLU(),
                        nn.Dropout(dropout * 0.5),
                        nn.Linear(expert_hidden, 1),
                    )
                    for _ in range(self.height_expert_count)
                ]
            )
        else:
            self.height_expert_gate = None
            self.height_experts = None
        if self.height_bin_classes > 1:
            bin_hidden = max(64, tower_hidden // 2)
            self.height_bin_head = nn.Sequential(
                nn.Linear(tower_hidden, bin_hidden),
                nn.SiLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(bin_hidden, self.height_bin_classes),
            )
            self.height_calibration_head = nn.Sequential(
                nn.Linear(tower_hidden + self.height_bin_classes + 1, bin_hidden),
                nn.SiLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(bin_hidden, 1),
            )
        else:
            self.height_bin_head = None
            self.height_calibration_head = None

        # Init
        self._init_weights()
        # Height targets are normalized. Start near the training-set mean instead
        # of emitting large random centimeter offsets in the first epochs.
        nn.init.normal_(self.height_out.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.height_out.bias)
        if self.height_experts is not None:
            for expert in self.height_experts:
                final = expert[-1]
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        if self.height_calibration_head is not None:
            final = self.height_calibration_head[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

        # Report
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[VocalMorphV4] Trainable: {n_params:,} / Total: {n_total:,}")
        print(f"[VocalMorphV4] adapter_dim={adapter_dim}, blocks={adapter_blocks}")
        print(f"[VocalMorphV4] tower_hidden={tower_hidden}, dropout={dropout}")
        print(f"[VocalMorphV4] meta_dim={self.meta_dim}, meta_hidden={meta_hidden}")
        print(
            "[VocalMorphV4] height_bin_classes="
            f"{self.height_bin_classes}, calibration_scale={self.height_calibration_scale}"
        )
        print(
            "[VocalMorphV4] height_expert_count="
            f"{self.height_expert_count}, expert_scale={self.height_expert_scale}"
        )
        print(f"[VocalMorphV4] ssl_backbone={use_ssl_backbone}")

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

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, T, D) SSL features or (B, num_samples) raw waveform
            padding_mask: (B, T) True where padded

        Returns:
            dict with 'height', 'gender_logits', 'embedding'
        """
        # If using SSL backbone, extract features from waveform
        if self.use_ssl_backbone and features.ndim == 2:
            features = self.ssl_backbone(features)
            padding_mask = None  # SSL handles its own masking

        B, T, D = features.shape

        # Project to adapter dimension
        x = self.input_proj(features)

        # Conformer adapter
        for block in self.adapter:
            x = block(x, padding_mask=padding_mask)

        # Pool to utterance-level embedding
        embedding = self.pooling(x, padding_mask=padding_mask)  # (B, adapter_dim)

        # Gender prediction
        gender_logits = self.gender_head(embedding)

        # Height regression: embed + predicted gender info + explicit acoustic biology.
        height_parts = [embedding, gender_logits]
        if self.meta_encoder is not None:
            if metadata is None:
                metadata = embedding.new_zeros((B, self.meta_dim))
            metadata = torch.nan_to_num(
                metadata.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            height_parts.append(self.meta_encoder(metadata))
        height_input = torch.cat(height_parts, dim=-1)
        h = self.height_tower(height_input)

        residual = h
        h = self.height_drop1(self.height_ln1(self.height_block1(h)))
        h = h + residual

        residual = h
        h = self.height_drop2(self.height_ln2(self.height_block2(h)))
        h = h + residual

        height_base = self.height_out(h).squeeze(-1)
        height = height_base
        height_bin_logits = None
        height_calibration = None
        height_expert_logits = None
        height_expert_offsets = None
        height_expert_offset = None
        if self.height_expert_gate is not None and self.height_experts is not None:
            height_expert_logits = self.height_expert_gate(h)
            expert_probs = height_expert_logits.softmax(dim=-1)
            height_expert_offsets = torch.cat(
                [torch.tanh(expert(h)) for expert in self.height_experts],
                dim=-1,
            ) * self.height_expert_scale
            height_expert_offset = (expert_probs * height_expert_offsets).sum(dim=-1)
            height = height + height_expert_offset
        if self.height_bin_head is not None:
            height_bin_logits = self.height_bin_head(h)
            bin_probs = height_bin_logits.softmax(dim=-1)
            calibration_input = torch.cat([h, bin_probs, height_base.unsqueeze(-1)], dim=-1)
            height_calibration = (
                torch.tanh(self.height_calibration_head(calibration_input)).squeeze(-1)
                * self.height_calibration_scale
            )
            height = height_base + height_calibration

        out = {
            "height": height,
            "height_base": height_base,
            "gender_logits": gender_logits,
            "embedding": embedding,
        }
        if height_bin_logits is not None:
            out["height_bin_logits"] = height_bin_logits
            out["height_calibration"] = height_calibration
        if height_expert_logits is not None:
            out["height_expert_logits"] = height_expert_logits
            out["height_expert_offsets"] = height_expert_offsets
            out["height_expert_offset"] = height_expert_offset
        return out

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        n_samples: int = 30,
    ) -> Dict[str, torch.Tensor]:
        """MC dropout uncertainty estimation."""
        self.train()
        preds = []
        for _ in range(n_samples):
            out = self.forward(features, padding_mask)
            preds.append(out["height"].unsqueeze(0))
        self.eval()
        stacked = torch.cat(preds, dim=0)
        return {
            "height_mean": stacked.mean(0),
            "height_std": stacked.std(0),
        }


# ────────────────────────────────────────────────────────────
# V4 Loss
# ────────────────────────────────────────────────────────────


class V4Loss(nn.Module):
    """
    Simple, stable height-focused loss.

    L = huber_weight * Huber(height) + mae_weight * MAE(height)
        + gender_weight * CE(gender) + smoothl1_weight * SmoothL1(height)
    
    No wing loss, no ranking, no isometric, no bin classification.
    Just clean, well-balanced objectives that don't fight each other.
    """

    def __init__(
        self,
        huber_delta: float = 1.0,
        huber_weight: float = 0.5,
        mae_weight: float = 0.3,
        smoothl1_weight: float = 0.2,
        gender_weight: float = 0.15,
        label_smoothing: float = 0.05,
        height_bin_loss_weight_short: float = 1.0,
        height_bin_loss_weight_medium: float = 1.0,
        height_bin_loss_weight_tall: float = 1.0,
        height_bin_loss_weight_male_short: float = 1.0,
        height_bin_loss_weight_female_short: float = 1.0,
        height_extreme_short_cm: float = 152.0,
        height_extreme_tall_cm: float = 190.0,
        height_extreme_loss_weight_short: float = 1.0,
        height_extreme_loss_weight_tall: float = 1.0,
        source_loss_weight_timit: float = 1.0,
        source_loss_weight_nisp: float = 1.0,
        source_loss_weight_external: float = 1.0,
        height_bin_weight: float = 0.0,
        height_bin_label_smoothing: float = 0.05,
        height_base_weight: float = 0.0,
        calibration_reg_weight: float = 0.0,
        height_expert_weight: float = 0.0,
        height_expert_label_smoothing: float = 0.03,
        height_expert_reg_weight: float = 0.0,
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.huber_weight = huber_weight
        self.mae_weight = mae_weight
        self.smoothl1_weight = smoothl1_weight
        self.gender_weight = gender_weight
        self.label_smoothing = label_smoothing
        self.height_bin_loss_weight_short = height_bin_loss_weight_short
        self.height_bin_loss_weight_medium = height_bin_loss_weight_medium
        self.height_bin_loss_weight_tall = height_bin_loss_weight_tall
        self.height_bin_loss_weight_male_short = height_bin_loss_weight_male_short
        self.height_bin_loss_weight_female_short = height_bin_loss_weight_female_short
        self.height_extreme_short_cm = height_extreme_short_cm
        self.height_extreme_tall_cm = height_extreme_tall_cm
        self.height_extreme_loss_weight_short = height_extreme_loss_weight_short
        self.height_extreme_loss_weight_tall = height_extreme_loss_weight_tall
        self.source_loss_weight_timit = source_loss_weight_timit
        self.source_loss_weight_nisp = source_loss_weight_nisp
        self.source_loss_weight_external = source_loss_weight_external
        self.height_bin_weight = height_bin_weight
        self.height_bin_label_smoothing = height_bin_label_smoothing
        self.height_base_weight = height_base_weight
        self.calibration_reg_weight = calibration_reg_weight
        self.height_expert_weight = height_expert_weight
        self.height_expert_label_smoothing = height_expert_label_smoothing
        self.height_expert_reg_weight = height_expert_reg_weight

    @staticmethod
    def _height_to_bins(height_raw: torch.Tensor, n_bins: int) -> torch.Tensor:
        """Convert raw centimeter heights to coarse ordinal body-size bins."""
        labels = torch.zeros_like(height_raw, dtype=torch.long)
        if n_bins <= 1:
            return labels
        if n_bins == 3:
            labels[height_raw >= 160.0] = 1
            labels[height_raw >= 175.0] = 2
            return labels
        if n_bins == 5:
            labels[height_raw >= 155.0] = 1
            labels[height_raw >= 165.0] = 2
            labels[height_raw >= 175.0] = 3
            labels[height_raw >= 185.0] = 4
            return labels
        edges = torch.linspace(145.0, 195.0, n_bins + 1, device=height_raw.device)
        for idx in range(1, n_bins):
            labels[height_raw >= edges[idx]] = idx
        return labels.clamp(min=0, max=n_bins - 1)

    def _height_weights(
        self, height_pred: torch.Tensor, targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        height_raw = targets.get("height_raw")
        if height_raw is None:
            return torch.ones_like(height_pred)
        height_raw = height_raw.to(device=height_pred.device, dtype=torch.float32)
        weights = torch.full_like(height_pred, float(self.height_bin_loss_weight_medium))
        short = height_raw < 160.0
        tall = height_raw >= 175.0
        weights = torch.where(
            short, weights * float(self.height_bin_loss_weight_short), weights
        )
        weights = torch.where(
            tall, weights * float(self.height_bin_loss_weight_tall), weights
        )
        gender = targets.get("gender")
        if gender is not None:
            gender = gender.to(device=height_pred.device, dtype=torch.long)
            male = gender == 1
            female = gender == 0
            weights = torch.where(
                short & male,
                weights * float(self.height_bin_loss_weight_male_short),
                weights,
            )
            weights = torch.where(
                short & female,
                weights * float(self.height_bin_loss_weight_female_short),
                weights,
            )
        weights = torch.where(
            height_raw <= float(self.height_extreme_short_cm),
            weights * float(self.height_extreme_loss_weight_short),
            weights,
        )
        weights = torch.where(
            height_raw >= float(self.height_extreme_tall_cm),
            weights * float(self.height_extreme_loss_weight_tall),
            weights,
        )
        weights = weights / weights.mean().clamp(min=1e-6)

        source_id = targets.get("source_id")
        if source_id is not None:
            source_id = source_id.to(device=height_pred.device, dtype=torch.long)
            weights = torch.where(
                source_id == 0,
                weights * float(self.source_loss_weight_timit),
                weights,
            )
            weights = torch.where(
                source_id == 1,
                weights * float(self.source_loss_weight_nisp),
                weights,
            )
            weights = torch.where(
                source_id == 2,
                weights * float(self.source_loss_weight_external),
                weights,
            )
        return weights

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        height_pred = preds["height"].float()
        height_target = targets["height"].float()

        sample_weights = self._height_weights(height_pred, targets)

        # Primary: Huber loss (robust to outliers)
        l_huber_each = F.huber_loss(
            height_pred, height_target, delta=self.huber_delta, reduction="none"
        )
        l_huber = (l_huber_each * sample_weights).mean()

        # Direct MAE optimization
        l_mae = ((height_pred - height_target).abs() * sample_weights).mean()

        # Smooth L1 for stable gradients near zero
        l_smooth_each = F.smooth_l1_loss(
            height_pred, height_target, beta=0.5, reduction="none"
        )
        l_smooth = (l_smooth_each * sample_weights).mean()

        # Gender auxiliary
        l_gender = torch.tensor(0.0, device=height_pred.device)
        if "gender_logits" in preds and "gender" in targets:
            l_gender = F.cross_entropy(
                preds["gender_logits"],
                targets["gender"].long(),
                label_smoothing=self.label_smoothing,
            )

        l_bin = torch.tensor(0.0, device=height_pred.device)
        if (
            self.height_bin_weight > 0.0
            and "height_bin_logits" in preds
            and targets.get("height_raw") is not None
        ):
            height_raw = targets["height_raw"].to(device=height_pred.device, dtype=torch.float32)
            bin_logits = preds["height_bin_logits"].float()
            bin_labels = self._height_to_bins(height_raw, int(bin_logits.shape[-1]))
            bin_each = F.cross_entropy(
                bin_logits,
                bin_labels,
                reduction="none",
                label_smoothing=float(self.height_bin_label_smoothing),
            )
            l_bin = (bin_each * sample_weights.detach()).mean()

        l_base = torch.tensor(0.0, device=height_pred.device)
        if self.height_base_weight > 0.0 and "height_base" in preds:
            base_each = F.smooth_l1_loss(
                preds["height_base"].float(),
                height_target,
                beta=0.5,
                reduction="none",
            )
            l_base = (base_each * sample_weights.detach()).mean()

        l_calib_reg = torch.tensor(0.0, device=height_pred.device)
        if self.calibration_reg_weight > 0.0 and preds.get("height_calibration") is not None:
            l_calib_reg = preds["height_calibration"].float().pow(2).mean()

        l_expert = torch.tensor(0.0, device=height_pred.device)
        if (
            self.height_expert_weight > 0.0
            and "height_expert_logits" in preds
            and targets.get("height_raw") is not None
        ):
            height_raw = targets["height_raw"].to(device=height_pred.device, dtype=torch.float32)
            expert_logits = preds["height_expert_logits"].float()
            expert_labels = self._height_to_bins(height_raw, int(expert_logits.shape[-1]))
            expert_each = F.cross_entropy(
                expert_logits,
                expert_labels,
                reduction="none",
                label_smoothing=float(self.height_expert_label_smoothing),
            )
            l_expert = (expert_each * sample_weights.detach()).mean()

        l_expert_reg = torch.tensor(0.0, device=height_pred.device)
        if self.height_expert_reg_weight > 0.0 and preds.get("height_expert_offsets") is not None:
            l_expert_reg = preds["height_expert_offsets"].float().pow(2).mean()

        # Total
        total = (
            self.huber_weight * l_huber
            + self.mae_weight * l_mae
            + self.smoothl1_weight * l_smooth
            + self.gender_weight * l_gender
            + self.height_bin_weight * l_bin
            + self.height_base_weight * l_base
            + self.calibration_reg_weight * l_calib_reg
            + self.height_expert_weight * l_expert
            + self.height_expert_reg_weight * l_expert_reg
        )

        return {
            "total": total,
            "height_huber": l_huber,
            "height_mae": l_mae,
            "height_smooth_l1": l_smooth,
            "gender_ce": l_gender,
            "height_bin_ce": l_bin,
            "height_base": l_base,
            "calibration_reg": l_calib_reg,
            "height_expert_ce": l_expert,
            "height_expert_reg": l_expert_reg,
        }


# ────────────────────────────────────────────────────────────
# Builder
# ────────────────────────────────────────────────────────────


def build_v4_model(config: dict) -> VocalMorphV4:
    """Build VocalMorphV4 from config."""
    model_cfg = config.get("model", {}).get("v4", {})
    input_dim = int(config.get("model", {}).get("input_dim", 768))

    return VocalMorphV4(
        input_dim=input_dim,
        adapter_dim=int(model_cfg.get("adapter_dim", 512)),
        adapter_heads=int(model_cfg.get("adapter_heads", 8)),
        adapter_blocks=int(model_cfg.get("adapter_blocks", 2)),
        adapter_ff_expansion=int(model_cfg.get("adapter_ff_expansion", 4)),
        adapter_conv_kernel=int(model_cfg.get("adapter_conv_kernel", 15)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        pool_hidden=int(model_cfg.get("pool_hidden", 128)),
        tower_hidden=int(model_cfg.get("tower_hidden", 256)),
        meta_dim=int(model_cfg.get("meta_dim", 0)),
        meta_hidden=int(model_cfg.get("meta_hidden", 64)),
        height_bin_classes=int(model_cfg.get("height_bin_classes", 0)),
        height_calibration_scale=float(model_cfg.get("height_calibration_scale", 0.0)),
        height_expert_count=int(model_cfg.get("height_expert_count", 0)),
        height_expert_scale=float(model_cfg.get("height_expert_scale", 0.0)),
        use_ssl_backbone=bool(model_cfg.get("use_ssl_backbone", False)),
        ssl_model_name=str(model_cfg.get("ssl_model_name", "facebook/wav2vec2-base")),
        ssl_freeze=bool(model_cfg.get("ssl_freeze", True)),
        ssl_layers_to_use=int(model_cfg.get("ssl_layers_to_use", 4)),
    )


def build_v4_loss(config: dict) -> V4Loss:
    """Build V4 loss from config."""
    loss_cfg = config.get("training", {}).get("loss", {})
    return V4Loss(
        huber_delta=float(loss_cfg.get("huber_delta", 1.0)),
        huber_weight=float(loss_cfg.get("huber_weight", 0.5)),
        mae_weight=float(loss_cfg.get("mae_weight", 0.3)),
        smoothl1_weight=float(loss_cfg.get("smoothl1_weight", 0.2)),
        gender_weight=float(loss_cfg.get("gender_weight", 0.15)),
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.05)),
        height_bin_loss_weight_short=float(
            loss_cfg.get("height_bin_loss_weight_short", 1.0)
        ),
        height_bin_loss_weight_medium=float(
            loss_cfg.get("height_bin_loss_weight_medium", 1.0)
        ),
        height_bin_loss_weight_tall=float(
            loss_cfg.get("height_bin_loss_weight_tall", 1.0)
        ),
        height_bin_loss_weight_male_short=float(
            loss_cfg.get("height_bin_loss_weight_male_short", 1.0)
        ),
        height_bin_loss_weight_female_short=float(
            loss_cfg.get("height_bin_loss_weight_female_short", 1.0)
        ),
        height_extreme_short_cm=float(loss_cfg.get("height_extreme_short_cm", 152.0)),
        height_extreme_tall_cm=float(loss_cfg.get("height_extreme_tall_cm", 190.0)),
        height_extreme_loss_weight_short=float(
            loss_cfg.get("height_extreme_loss_weight_short", 1.0)
        ),
        height_extreme_loss_weight_tall=float(
            loss_cfg.get("height_extreme_loss_weight_tall", 1.0)
        ),
        source_loss_weight_timit=float(loss_cfg.get("source_loss_weight_timit", 1.0)),
        source_loss_weight_nisp=float(loss_cfg.get("source_loss_weight_nisp", 1.0)),
        source_loss_weight_external=float(
            loss_cfg.get("source_loss_weight_external", 1.0)
        ),
        height_bin_weight=float(loss_cfg.get("height_bin_weight", 0.0)),
        height_bin_label_smoothing=float(
            loss_cfg.get("height_bin_label_smoothing", 0.05)
        ),
        height_base_weight=float(loss_cfg.get("height_base_weight", 0.0)),
        calibration_reg_weight=float(loss_cfg.get("calibration_reg_weight", 0.0)),
        height_expert_weight=float(loss_cfg.get("height_expert_weight", 0.0)),
        height_expert_label_smoothing=float(
            loss_cfg.get("height_expert_label_smoothing", 0.03)
        ),
        height_expert_reg_weight=float(loss_cfg.get("height_expert_reg_weight", 0.0)),
    )


if __name__ == "__main__":
    # Quick test
    B, T, D = 4, 100, 768
    model = VocalMorphV4(input_dim=D)
    x = torch.randn(B, T, D)
    out = model(x)
    for k, v in out.items():
        print(k, tuple(v.shape))
    print(f"\nParams: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
