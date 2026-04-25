"""Prediction heads for VocalMorph V2."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEFAULT_PHYSICS_CONSTANTS, EPS, PhysicsConstants
from .layers import MCDropout
from .types import RegressionHeadOutput
from .utils import _plausible_spacing, _plausible_vtl, _validate_class_labels


class ProbabilisticRegressionHead(nn.Module):
    """MC-dropout compatible heteroscedastic regression head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 192,
        dropout: float = 0.30,
        var_scale: float = 0.5,
        min_var: float = EPS,
    ):
        super().__init__()
        self.min_var = float(min_var)
        self.var_scale = float(var_scale)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            MCDropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            MCDropout(dropout),
        )
        self.mu_head = nn.Linear(hidden_dim // 2, 1)
        self.var_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> RegressionHeadOutput:
        h = self.trunk(x)
        mu = self.mu_head(h).squeeze(-1)
        raw_var = (
            F.softplus(self.var_head(h).squeeze(-1)) * self.var_scale + self.min_var
        )
        logvar = torch.log(raw_var).clamp(min=math.log(self.min_var), max=3.0)
        return {"mu": mu, "var": logvar.exp(), "logvar": logvar}


class BayesianHeightHead(ProbabilisticRegressionHead):
    def forward(
        self,
        x: torch.Tensor,
        physics_residual: Optional[torch.Tensor] = None,
        prior_residual: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = super().forward(x)
        mu_base = out["mu"]
        mu = mu_base
        if prior_residual is not None:
            mu = mu + prior_residual
        if physics_residual is not None:
            mu = mu + physics_residual
        return {
            "mu_base": mu_base,
            "mu": mu,
            "var": out["var"],
            "logvar": out["logvar"],
        }


class MCRegressionHead(ProbabilisticRegressionHead):
    """Backward-compatible mean-only wrapper retained for external imports."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)["mu"]


class HeightPriorHead(nn.Module):
    """Conditional prior around which height residuals are learned."""

    def __init__(
        self,
        physics_dim: int = 64,
        physics_summary_dim: int = 3,
        domain_emb_dim: int = 8,
        hidden_dim: int = 96,
        dropout: float = 0.10,
        constants: PhysicsConstants = DEFAULT_PHYSICS_CONSTANTS,
    ):
        super().__init__()
        self.constants = constants
        self.physics_summary_dim = int(physics_summary_dim)
        in_dim = physics_dim + self.physics_summary_dim + 1 + 2 + domain_emb_dim
        self.domain_emb = nn.Embedding(2, domain_emb_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        physics_embedding: torch.Tensor,
        age_pred: torch.Tensor,
        gender_logits: torch.Tensor,
        domain: Optional[torch.Tensor],
        physics_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if domain is None:
            domain_idx = torch.zeros(
                physics_embedding.size(0),
                device=physics_embedding.device,
                dtype=torch.long,
            )
        else:
            _validate_class_labels(domain, "domain", self.domain_emb.num_embeddings)
            domain_idx = domain.to(device=physics_embedding.device, dtype=torch.long)
        dom = self.domain_emb(domain_idx)
        age_feat = age_pred.detach().unsqueeze(-1)
        gender_probs = torch.softmax(gender_logits.detach(), dim=-1)
        if physics_summary is None:
            physics_summary = torch.zeros(
                physics_embedding.size(0),
                self.physics_summary_dim,
                device=physics_embedding.device,
                dtype=physics_embedding.dtype,
            )
        x = torch.cat(
            [physics_embedding, physics_summary, age_feat, gender_probs, dom], dim=-1
        )
        return self.constants.prior_tanh_scale * torch.tanh(self.net(x).squeeze(-1))


class AcousticPhysicsConsistencyHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 96,
        dropout: float = 0.10,
        constants: PhysicsConstants = DEFAULT_PHYSICS_CONSTANTS,
    ):
        super().__init__()
        self.constants = constants
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.net(x)
        spacing = _plausible_spacing(raw[:, 0], self.constants)
        vtl = _plausible_vtl(raw[:, 1], self.constants)
        return {"spacing": spacing, "vtl": vtl}


class ReliabilityAdaptivePhysicsGate(nn.Module):
    """Learn a conservative scalar gate for physics-conditioned features."""

    def __init__(
        self,
        acoustic_dim: int = 256,
        physics_dim: int = 64,
        reliability_dim: int = 5,
        hidden_dim: int = 96,
        dropout: float = 0.10,
        gate_floor: float = 0.10,
    ):
        super().__init__()
        if not 0.0 <= gate_floor < 1.0:
            raise ValueError(f"gate_floor must be in [0, 1), got {gate_floor}")
        self.gate_floor = float(gate_floor)
        in_dim = acoustic_dim + physics_dim + reliability_dim + 4
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        acoustic_embedding: torch.Tensor,
        physics_embedding: torch.Tensor,
        physics_reliability: torch.Tensor,
        physics_confidence: torch.Tensor,
        quality_score: torch.Tensor,
    ) -> torch.Tensor:
        if acoustic_embedding.ndim != 2:
            raise ValueError(
                f"acoustic_embedding must have shape (B, D), got {tuple(acoustic_embedding.shape)}"
            )
        if physics_embedding.ndim != 2:
            raise ValueError(
                f"physics_embedding must have shape (B, D), got {tuple(physics_embedding.shape)}"
            )
        if physics_reliability.ndim != 2:
            raise ValueError(
                f"physics_reliability must have shape (B, R), got {tuple(physics_reliability.shape)}"
            )
        if physics_confidence.ndim != 1:
            raise ValueError(
                f"physics_confidence must have shape (B,), got {tuple(physics_confidence.shape)}"
            )
        if quality_score.ndim != 1:
            raise ValueError(
                f"quality_score must have shape (B,), got {tuple(quality_score.shape)}"
            )
        if acoustic_embedding.size(0) != physics_embedding.size(
            0
        ) or acoustic_embedding.size(0) != physics_reliability.size(0):
            raise ValueError(
                "Batch dimensions for acoustic, physics, and reliability inputs must match"
            )

        reliability_mean = physics_reliability.mean(dim=-1, keepdim=True)
        reliability_std = physics_reliability.std(dim=-1, keepdim=True, unbiased=False)
        x = torch.cat(
            [
                acoustic_embedding,
                physics_embedding,
                physics_reliability,
                physics_confidence.unsqueeze(-1),
                quality_score.unsqueeze(-1),
                reliability_mean,
                reliability_std,
            ],
            dim=-1,
        )
        learned_gate = torch.sigmoid(self.net(x).squeeze(-1))
        confidence = physics_confidence.clamp(min=0.0, max=1.0)
        gate = self.gate_floor + (1.0 - self.gate_floor) * learned_gate * confidence
        return gate.clamp(min=self.gate_floor, max=1.0)


class HeightFeatureAdapter(nn.Module):
    """Small task-specific adapter that sharpens the representation used for height."""

    def __init__(
        self,
        fused_dim: int = 256,
        physics_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.10,
        adapter_scale: float = 0.25,
    ):
        super().__init__()
        self.adapter_scale = float(adapter_scale)
        in_dim = fused_dim + physics_dim + 2
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fused_dim),
        )

    def forward(
        self,
        fused_embedding: torch.Tensor,
        physics_embedding: torch.Tensor,
        physics_gate: torch.Tensor,
        quality_score: torch.Tensor,
    ) -> torch.Tensor:
        if fused_embedding.ndim != 2:
            raise ValueError(
                f"fused_embedding must have shape (B, D), got {tuple(fused_embedding.shape)}"
            )
        if physics_embedding.ndim != 2:
            raise ValueError(
                f"physics_embedding must have shape (B, D), got {tuple(physics_embedding.shape)}"
            )
        if physics_gate.ndim != 1 or quality_score.ndim != 1:
            raise ValueError("physics_gate and quality_score must have shape (B,)")
        x = torch.cat(
            [
                fused_embedding,
                physics_embedding,
                physics_gate.unsqueeze(-1),
                quality_score.unsqueeze(-1),
            ],
            dim=-1,
        )
        return fused_embedding + self.adapter_scale * torch.tanh(self.net(x))


class _ResidualRefinementBlock(nn.Module):
    """Small gated residual MLP block for task-specific feature refinement."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.10):
        super().__init__()
        self.main = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = torch.tanh(self.main(x))
        return x + self.gate(x) * residual


class HeightConditionedRefiner(nn.Module):
    """Build a height-first representation from fused, acoustic, and reliability cues."""

    def __init__(
        self,
        fused_dim: int,
        acoustic_dim: int,
        physics_dim: int,
        reliability_dim: int,
        summary_dim: int = 3,
        hidden_dim: int = 192,
        num_blocks: int = 2,
        dropout: float = 0.10,
        context_scale: float = 0.35,
    ):
        super().__init__()
        self.context_scale = float(context_scale)
        context_dim = (
            fused_dim
            + acoustic_dim
            + acoustic_dim
            + physics_dim
            + reliability_dim
            + summary_dim
            + 3
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fused_dim),
        )
        self.blocks = nn.ModuleList(
            [
                _ResidualRefinementBlock(fused_dim, hidden_dim, dropout=dropout)
                for _ in range(max(1, int(num_blocks)))
            ]
        )

    def forward(
        self,
        *,
        base_height_features: torch.Tensor,
        acoustic_embedding: torch.Tensor,
        cross_embedding: torch.Tensor,
        physics_embedding: torch.Tensor,
        reliability_features: torch.Tensor,
        prior_summary: torch.Tensor,
        quality_score: torch.Tensor,
        usable_clip_probability: torch.Tensor,
        physics_gate: torch.Tensor,
    ) -> torch.Tensor:
        if base_height_features.ndim != 2:
            raise ValueError(
                f"base_height_features must have shape (B, D), got {tuple(base_height_features.shape)}"
            )
        if acoustic_embedding.ndim != 2 or cross_embedding.ndim != 2:
            raise ValueError("acoustic_embedding and cross_embedding must have shape (B, D)")
        if physics_embedding.ndim != 2 or reliability_features.ndim != 2:
            raise ValueError("physics_embedding and reliability_features must have shape (B, D)")
        if prior_summary.ndim != 2:
            raise ValueError(f"prior_summary must have shape (B, D), got {tuple(prior_summary.shape)}")
        if quality_score.ndim != 1 or usable_clip_probability.ndim != 1 or physics_gate.ndim != 1:
            raise ValueError("quality_score, usable_clip_probability, and physics_gate must have shape (B,)")

        context = torch.cat(
            [
                base_height_features,
                acoustic_embedding,
                cross_embedding,
                physics_embedding,
                reliability_features,
                prior_summary,
                quality_score.unsqueeze(-1),
                usable_clip_probability.unsqueeze(-1),
                physics_gate.unsqueeze(-1),
            ],
            dim=-1,
        )
        refined = base_height_features + self.context_scale * torch.tanh(
            self.context_proj(context)
        )
        for block in self.blocks:
            refined = block(refined)
        return refined


class HeightBinClassificationHead(nn.Module):
    """Auxiliary classification head that regularizes the height representation."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 96,
        n_classes: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()
        if n_classes < 3:
            raise ValueError(f"n_classes must be >= 3, got {n_classes}")
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = [
    "AcousticPhysicsConsistencyHead",
    "BayesianHeightHead",
    "HeightBinClassificationHead",
    "HeightConditionedRefiner",
    "HeightFeatureAdapter",
    "HeightPriorHead",
    "MCRegressionHead",
    "ProbabilisticRegressionHead",
    "ReliabilityAdaptivePhysicsGate",
]
