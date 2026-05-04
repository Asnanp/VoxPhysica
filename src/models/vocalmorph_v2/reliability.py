"""Shared clip-reliability estimation and Omega speaker pooling utilities."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn

from .config import AggregationConfig, ReliabilityConfig


RELIABILITY_FEATURE_DIM = 10
RELIABILITY_COMPONENT_KEYS = (
    "capture",
    "speech",
    "snr",
    "clipping",
    "distance",
    "voiced",
    "duration",
    "frames",
    "uncertainty",
    "drift",
)
HANDCRAFTED_WEIGHTS = {
    "capture": 0.20,
    "speech": 0.10,
    "snr": 0.10,
    "clipping": 0.10,
    "distance": 0.10,
    "voiced": 0.10,
    "duration": 0.05,
    "frames": 0.05,
    "uncertainty": 0.10,
    "drift": 0.10,
}


def _ensure_vector(
    value: Optional[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    default: float,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), float(default), device=device, dtype=dtype)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, device=device, dtype=dtype)
    else:
        value = value.to(device=device, dtype=dtype)
    if value.ndim == 0:
        value = value.expand(batch_size)
    elif value.ndim > 1:
        value = value.reshape(batch_size, -1)[:, 0]
    if value.size(0) != batch_size:
        raise ValueError(
            f"Expected metadata vector with batch size {batch_size}, got shape {tuple(value.shape)}"
        )
    return value


def _metadata_value(
    metadata: Optional[Mapping[str, Any]],
    key: str,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    default: float,
) -> torch.Tensor:
    value = None if metadata is None else metadata.get(key)
    return _ensure_vector(
        value,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        default=default,
    )


def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)


def reliability_feature_vector(
    *,
    metadata: Optional[Mapping[str, Any]],
    valid_frames: torch.Tensor,
    crop_frames: Optional[int],
    pred_std_cm: Optional[torch.Tensor] = None,
    use_feature_drift: bool = True,
) -> Dict[str, torch.Tensor]:
    """Build the canonical Omega clip-reliability feature vector."""

    if valid_frames.ndim != 1:
        raise ValueError(f"valid_frames must have shape (B,), got {tuple(valid_frames.shape)}")

    batch_size = valid_frames.size(0)
    device = valid_frames.device
    dtype = valid_frames.dtype

    capture_quality = _metadata_value(
        metadata,
        "capture_quality_score",
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        default=0.50,
    )
    speech_ratio = _metadata_value(
        metadata,
        "speech_ratio",
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        default=0.70,
    )
    snr_db = _metadata_value(
        metadata,
        "snr_db_estimate",
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        default=15.0,
    )
    clipped_ratio = _metadata_value(
        metadata,
        "clipped_ratio",
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        default=0.0,
    )
    distance_cm = _metadata_value(
        metadata,
        "distance_cm_estimate",
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        default=18.0,
    )
    voiced_ratio = _metadata_value(
        metadata,
        "voiced_ratio",
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        default=0.75,
    )
    duration_s = _metadata_value(
        metadata,
        "duration_s",
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        default=4.0,
    )
    if use_feature_drift:
        drift_z = _metadata_value(
            metadata,
            "feature_drift_zscore",
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            default=float("nan"),
        )
        alt_ood = _metadata_value(
            metadata,
            "ood_zscore",
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            default=float("nan"),
        )
        drift_z = torch.where(torch.isfinite(drift_z), drift_z, alt_ood)
    else:
        drift_z = torch.full((batch_size,), float("nan"), device=device, dtype=dtype)

    pred_std = _ensure_vector(
        pred_std_cm,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        default=0.0,
    )

    crop_frames_value = float(crop_frames) if crop_frames is not None else float(torch.max(valid_frames).item() if valid_frames.numel() else 1.0)
    crop_frames_value = max(crop_frames_value, 1.0)

    components = {
        "capture": _clamp01(capture_quality),
        "speech": _clamp01((speech_ratio - 0.35) / 0.55),
        "snr": _clamp01((snr_db - 5.0) / 20.0),
        "clipping": _clamp01(1.0 - clipped_ratio / 0.08),
        "distance": _clamp01(1.0 - torch.abs(distance_cm - 18.0) / 30.0),
        "voiced": _clamp01((voiced_ratio - 0.20) / 0.70),
        "duration": _clamp01(duration_s / 4.0),
        "frames": _clamp01(torch.sqrt(valid_frames.clamp(min=1.0) / crop_frames_value)),
        "uncertainty": _clamp01(1.0 / (1.0 + pred_std.clamp(min=0.0) / 4.0)),
        "drift": _clamp01(
            torch.where(
                torch.isfinite(drift_z),
                torch.exp(-0.5 * torch.minimum(drift_z.abs(), torch.full_like(drift_z, 3.0)).pow(2)),
                torch.ones_like(drift_z),
            )
        ),
    }
    features = torch.stack([components[key] for key in RELIABILITY_COMPONENT_KEYS], dim=-1)
    return {
        "features": features,
        "pred_std_cm": pred_std,
        **components,
    }


def compose_handcrafted_clip_reliability(
    *,
    metadata: Optional[Mapping[str, Any]],
    valid_frames: torch.Tensor,
    crop_frames: Optional[int],
    pred_std_cm: Optional[torch.Tensor],
    min_weight: float,
    use_feature_drift: bool = True,
) -> Dict[str, torch.Tensor]:
    feature_payload = reliability_feature_vector(
        metadata=metadata,
        valid_frames=valid_frames,
        crop_frames=crop_frames,
        pred_std_cm=pred_std_cm,
        use_feature_drift=use_feature_drift,
    )
    features = feature_payload["features"]
    handcrafted = torch.zeros(features.size(0), device=features.device, dtype=features.dtype)
    for key, weight in HANDCRAFTED_WEIGHTS.items():
        handcrafted = handcrafted + float(weight) * feature_payload[key]
    handcrafted = handcrafted.clamp(min=float(min_weight), max=1.0)
    usable_logit = torch.logit(handcrafted.clamp(min=1e-4, max=1.0 - 1e-4))
    return {
        "clip_reliability_prior": handcrafted,
        "usable_clip_logit": usable_logit,
        "reliability_embedding": features,
        **feature_payload,
    }


class MetadataReliabilityTower(nn.Module):
    """Small MLP that learns a trust prior from audited clip metadata."""

    def __init__(
        self,
        in_dim: int = RELIABILITY_FEATURE_DIM,
        hidden_dim: int = 32,
        embedding_dim: int = 16,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.prior_head = nn.Linear(hidden_dim, 1)
        self.usable_head = nn.Linear(hidden_dim, 1)
        self.embedding_head = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.backbone(features)
        usable_logit = self.usable_head(hidden).squeeze(-1)
        prior = torch.sigmoid(self.prior_head(hidden).squeeze(-1))
        embedding = self.embedding_head(hidden)
        return {
            "clip_reliability_prior": prior,
            "usable_clip_logit": usable_logit,
            "reliability_embedding": embedding,
        }


def compose_clip_reliability(
    *,
    config: ReliabilityConfig,
    metadata: Optional[Mapping[str, Any]],
    valid_frames: torch.Tensor,
    crop_frames: Optional[int],
    pred_std_cm: Optional[torch.Tensor] = None,
    tower: Optional[MetadataReliabilityTower] = None,
) -> Dict[str, torch.Tensor]:
    handcrafted = compose_handcrafted_clip_reliability(
        metadata=metadata,
        valid_frames=valid_frames,
        crop_frames=crop_frames,
        pred_std_cm=pred_std_cm,
        min_weight=config.min_weight,
        use_feature_drift=config.use_feature_drift,
    )
    if config.mode != "learned" or tower is None:
        return handcrafted

    learned = tower(handcrafted["features"])
    prior = learned["clip_reliability_prior"].clamp(min=config.min_weight, max=1.0)
    return {
        "clip_reliability_prior": prior,
        "usable_clip_logit": learned["usable_clip_logit"],
        "reliability_embedding": learned["reliability_embedding"],
        "features": handcrafted["features"],
        "handcrafted_clip_reliability_prior": handcrafted["clip_reliability_prior"],
        **{key: handcrafted[key] for key in RELIABILITY_COMPONENT_KEYS},
        "pred_std_cm": handcrafted["pred_std_cm"],
    }


def uncertainty_temper_from_std(pred_std_cm: torch.Tensor) -> torch.Tensor:
    return (1.0 / (1.0 + pred_std_cm.clamp(min=0.0) / 3.0)).clamp(min=0.05, max=1.0)


def omega_reliability_pool(
    values: torch.Tensor,
    *,
    clip_reliability: torch.Tensor,
    pred_var: Optional[torch.Tensor],
    config: AggregationConfig,
) -> Dict[str, Any]:
    """Robust 1D speaker pooling used by the Omega aggregation path."""

    if values.ndim != 1:
        raise ValueError(f"values must have shape (N,), got {tuple(values.shape)}")
    if clip_reliability.ndim != 1 or clip_reliability.size(0) != values.size(0):
        raise ValueError("clip_reliability must align with values")

    base_weights = clip_reliability.clamp(min=1e-6)
    pred_std = None
    if pred_var is not None:
        if pred_var.ndim != 1 or pred_var.size(0) != values.size(0):
            raise ValueError("pred_var must align with values")
        pred_std = torch.sqrt(pred_var.clamp(min=1e-6))
        base_weights = (base_weights * uncertainty_temper_from_std(pred_std)).clamp(min=1e-6)

    median = values.median()
    abs_dev = (values - median).abs()
    mad = abs_dev.median().clamp(min=1e-4)
    modified_z = 0.6745 * abs_dev / mad
    inlier_mask = modified_z <= float(config.omega_mad_z)
    if int(inlier_mask.sum().item()) < int(config.omega_min_survivors):
        inlier_mask = torch.ones_like(inlier_mask, dtype=torch.bool)

    kept_values = values[inlier_mask]
    kept_weights = base_weights[inlier_mask]
    kept_weights = kept_weights / kept_weights.sum().clamp(min=1e-6)

    center = torch.sum(kept_values * kept_weights)
    mad_kept = (kept_values - center).abs().median()
    delta_cm = max(1.5, float(config.omega_huber_delta_scale) * float(mad_kept.item()))
    delta = torch.tensor(delta_cm, device=values.device, dtype=values.dtype)

    for _ in range(2):
        residual = (kept_values - center).abs()
        huber_weights = torch.where(
            residual <= delta,
            torch.ones_like(residual),
            delta / residual.clamp(min=1e-6),
        )
        weights = (kept_weights * huber_weights).clamp(min=1e-6)
        weights = weights / weights.sum().clamp(min=1e-6)
        center = torch.sum(kept_values * weights)

    dispersion = torch.sum(weights * (kept_values - center).pow(2))
    aleatoric = torch.tensor(0.0, device=values.device, dtype=values.dtype)
    if pred_var is not None:
        aleatoric = torch.sum(weights * pred_var[inlier_mask].clamp(min=1e-6))
    pooled_var = (dispersion + aleatoric).clamp(min=1e-6)
    rejected_count = int((~inlier_mask).sum().item())
    return {
        "mean": center,
        "var": pooled_var,
        "std": torch.sqrt(pooled_var),
        "rejected_count": rejected_count,
        "surviving_count": int(inlier_mask.sum().item()),
        "effective_weight_sum": float((base_weights[inlier_mask]).sum().item()),
        "inlier_mask": inlier_mask,
        "base_weights": base_weights,
        "mad_cm": mad,
        "delta_cm": delta,
        "pred_std_cm": pred_std,
    }


__all__ = [
    "HANDCRAFTED_WEIGHTS",
    "MetadataReliabilityTower",
    "RELIABILITY_COMPONENT_KEYS",
    "RELIABILITY_FEATURE_DIM",
    "compose_clip_reliability",
    "compose_handcrafted_clip_reliability",
    "omega_reliability_pool",
    "reliability_feature_vector",
    "uncertainty_temper_from_std",
]
