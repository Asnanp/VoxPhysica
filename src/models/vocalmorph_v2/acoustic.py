"""Acoustic path modules for VocalMorph V2."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.ecapa import SERes2Block, TDNNBlock

from .layers import AttentiveStatsPooling, ConformerBlock, SinusoidalPositionalEncoding
from .utils import _validate_sequence_inputs


class LearnedFeatureNormalizer(nn.Module):
    """Masked per-utterance normalization with learnable affine recovery."""

    def __init__(self, input_dim: int, eps: float = 1e-5):
        super().__init__()
        self.input_dim = int(input_dim)
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(self.input_dim))
        self.bias = nn.Parameter(torch.zeros(self.input_dim))

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.input_dim:
            raise ValueError(f"x must have shape (B, T, {self.input_dim}), got {tuple(x.shape)}")

        if padding_mask is None:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False)
        else:
            valid = (~padding_mask).to(dtype=x.dtype).unsqueeze(-1)
            denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean = (x * valid).sum(dim=1, keepdim=True) / denom
            centered = (x - mean) * valid
            var = centered.pow(2).sum(dim=1, keepdim=True) / denom

        normalized = (x - mean) / torch.sqrt(var + self.eps)
        normalized = normalized * self.scale.view(1, 1, -1) + self.bias.view(1, 1, -1)
        if padding_mask is not None:
            normalized = normalized.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return normalized


class AcousticPathECAPAConformer(nn.Module):
    """
    Full-sequence acoustic path:
    ECAPA front-end -> projected sequence -> Conformer stack -> attentive stats pooling.
    """

    def __init__(
        self,
        input_dim: int,
        ecapa_channels: int = 512,
        ecapa_scale: int = 8,
        conformer_d_model: int = 256,
        conformer_heads: int = 8,
        conformer_blocks: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.05,
        layer_scale_init: float = 1e-4,
        rel_pos_max_distance: int = 128,
        pooling_hidden_dim: int = 128,
        use_feature_normalization: bool = True,
        feature_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.conformer_d_model = int(conformer_d_model)
        self.feature_normalizer = (
            LearnedFeatureNormalizer(input_dim, eps=feature_norm_eps) if use_feature_normalization else None
        )
        self.front = TDNNBlock(input_dim, ecapa_channels, kernel_size=5, dilation=1)
        self.block1 = SERes2Block(ecapa_channels, scale=ecapa_scale, dilation=2)
        self.block2 = SERes2Block(ecapa_channels, scale=ecapa_scale, dilation=3)
        self.block3 = SERes2Block(ecapa_channels, scale=ecapa_scale, dilation=4)
        self.mfa = nn.Sequential(
            nn.Conv1d(ecapa_channels * 3, ecapa_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(ecapa_channels),
            nn.ReLU(),
        )
        self.sequence_proj = nn.Conv1d(ecapa_channels, conformer_d_model, kernel_size=1, bias=False)
        self.sequence_norm = nn.LayerNorm(conformer_d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(conformer_d_model)
        self.conformer = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=conformer_d_model,
                    n_heads=conformer_heads,
                    ff_mult=4,
                    dropout=dropout,
                    drop_path=drop_path,
                    layer_scale_init=layer_scale_init,
                    rel_pos_max_distance=rel_pos_max_distance,
                )
                for _ in range(conformer_blocks)
            ]
        )
        self.post_norm = nn.LayerNorm(conformer_d_model)
        self.pool = AttentiveStatsPooling(conformer_d_model, hidden_dim=pooling_hidden_dim)
        self.embedding_proj = nn.Sequential(
            nn.LayerNorm(conformer_d_model * 2),
            nn.Linear(conformer_d_model * 2, conformer_d_model),
        )

    def _assert_sequence_shape(
        self,
        seq: torch.Tensor,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> None:
        if seq.ndim != 3 or seq.size(-1) != self.conformer_d_model:
            raise ValueError(f"Expected acoustic sequence (B, T, {self.conformer_d_model}), got {tuple(seq.shape)}")
        if seq.size(1) == 1:
            if padding_mask is None:
                valid_lengths = torch.full((features.size(0),), features.size(1), device=features.device, dtype=torch.long)
            else:
                valid_lengths = (~padding_mask).sum(dim=1)
            if torch.any(valid_lengths > 1):
                raise ValueError("Conformer received a single token even though the input had more than one valid frame.")

    def _assert_pooling_shape(
        self,
        pooled: torch.Tensor,
        attention: torch.Tensor,
        sequence: torch.Tensor,
    ) -> None:
        batch_size, time_steps, feature_dim = sequence.shape
        expected_pooled = (batch_size, feature_dim * 2)
        expected_attention = (batch_size, time_steps)
        if tuple(pooled.shape) != expected_pooled:
            raise ValueError(f"Expected pooled acoustic shape {expected_pooled}, got {tuple(pooled.shape)}")
        if tuple(attention.shape) != expected_attention:
            raise ValueError(f"Expected acoustic attention shape {expected_attention}, got {tuple(attention.shape)}")

    def forward(self, features: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        mask = _validate_sequence_inputs(
            features,
            padding_mask,
            expected_feature_dim=self.input_dim,
            name="acoustic_features",
        )
        features_in = features if self.feature_normalizer is None else self.feature_normalizer(features, padding_mask=mask)
        x = features_in.transpose(1, 2)
        x = self.front(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x = self.mfa(torch.cat([x1, x2, x3], dim=1))
        x = self.sequence_proj(x).transpose(1, 2)
        x = self.sequence_norm(x)
        self._assert_sequence_shape(x, features, mask)

        x = self.positional_encoding(x)
        for block in self.conformer:
            x = block(x, padding_mask=mask)
        x = self.post_norm(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        pooled, attn = self.pool(x, padding_mask=mask)
        self._assert_pooling_shape(pooled, attn, x)
        acoustic_embedding = self.embedding_proj(pooled)
        if acoustic_embedding.shape != (features.size(0), self.conformer_d_model):
            raise ValueError(
                f"Expected acoustic embedding shape {(features.size(0), self.conformer_d_model)}, got {tuple(acoustic_embedding.shape)}"
            )
        return {
            "sequence": x,
            "sequence_attention": attn,
            "acoustic_embedding": acoustic_embedding,
        }


__all__ = ["AcousticPathECAPAConformer", "LearnedFeatureNormalizer"]
