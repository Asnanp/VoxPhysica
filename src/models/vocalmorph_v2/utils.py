"""Reusable tensor validation, masking, aggregation, and numeric helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

from .config import (
    DEFAULT_PHYSICS_CONSTANTS,
    EPS,
    AggregationConfig,
    PhysicsConstants,
    ReliabilityConfig,
)
from .reliability import compose_handcrafted_clip_reliability, omega_reliability_pool
from .types import RegressionHeadOutput


def _sanitize_padding_mask(
    padding_mask: Optional[torch.Tensor],
    *,
    expected_shape: Optional[Tuple[int, int]] = None,
    name: str = "padding_mask",
) -> Optional[torch.Tensor]:
    """Validate and normalize a `(B, T)` padding mask to boolean form."""
    if padding_mask is None:
        return None

    if padding_mask.ndim != 2:
        raise ValueError(f"{name} must have shape (B, T), got {tuple(padding_mask.shape)}")

    if expected_shape is not None and tuple(padding_mask.shape) != tuple(expected_shape):
        raise ValueError(f"{name} must have shape {tuple(expected_shape)}, got {tuple(padding_mask.shape)}")

    if padding_mask.dtype == torch.bool:
        mask = padding_mask
    else:
        if not bool(torch.isfinite(padding_mask).all()):
            raise ValueError(f"{name} must be finite if provided as a numeric tensor")
        valid_entries = (padding_mask == 0) | (padding_mask == 1)
        if not bool(valid_entries.all()):
            bad = padding_mask[~valid_entries][:8]
            raise TypeError(f"{name} must be boolean or 0/1 valued. Example invalid values: {bad}")
        mask = padding_mask.to(dtype=torch.bool)

    all_padded = mask.all(dim=1)
    if all_padded.any():
        mask = mask.clone()
        mask[all_padded, 0] = False
    return mask


def _validate_sequence_inputs(
    features: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    *,
    expected_feature_dim: Optional[int] = None,
    name: str = "features",
) -> Optional[torch.Tensor]:
    """Validate a `(B, T, D)` feature tensor and optional `(B, T)` padding mask."""
    if features.ndim != 3:
        raise ValueError(f"{name} must have shape (B, T, D), got {tuple(features.shape)}")
    if not torch.is_floating_point(features):
        raise TypeError(f"{name} must be a floating-point tensor, got dtype={features.dtype}")
    if features.size(1) < 1:
        raise ValueError(f"{name} must contain at least one time step, got shape {tuple(features.shape)}")
    if expected_feature_dim is not None and features.size(-1) != expected_feature_dim:
        raise ValueError(
            f"{name} expected feature dim {expected_feature_dim}, got {features.size(-1)} with shape {tuple(features.shape)}"
        )
    if padding_mask is None:
        return None
    return _sanitize_padding_mask(
        padding_mask,
        expected_shape=(features.shape[0], features.shape[1]),
        name=f"{name}_padding_mask",
    )


def _validate_batch_axis(
    x: torch.Tensor,
    batch_size: int,
    *,
    name: str,
    expected_ndim: Optional[int] = None,
) -> None:
    """Validate that a tensor is batch-aligned with the current mini-batch."""
    if x.ndim == 0:
        raise ValueError(f"{name} must include a batch dimension, got scalar shape {tuple(x.shape)}")
    if x.size(0) != batch_size:
        raise ValueError(f"{name} expected leading batch dim {batch_size}, got {tuple(x.shape)}")
    if expected_ndim is not None and x.ndim != expected_ndim:
        raise ValueError(f"{name} expected ndim={expected_ndim}, got shape {tuple(x.shape)}")


def _clone_tensor_mapping(mapping: Optional[Mapping[str, Any]]) -> Optional[MutableMapping[str, Any]]:
    if mapping is None:
        return None
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in mapping.items()}


def _masked_mean(x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"x must have shape (B, T, D), got {tuple(x.shape)}")
    if padding_mask is None:
        return x.mean(dim=1)
    if padding_mask.ndim != 2 or padding_mask.shape[:2] != x.shape[:2]:
        raise ValueError(
            "padding_mask shape must match the first two dims of x: "
            f"x={tuple(x.shape)}, padding_mask={tuple(padding_mask.shape)}"
        )
    valid = (~padding_mask).to(dtype=x.dtype).unsqueeze(-1)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (x * valid).sum(dim=1) / denom


def _safe_mean(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros((), device=device, dtype=torch.float32)
    return x.mean()


def _denorm_tensor(
    value: torch.Tensor,
    key: str,
    target_stats: Optional[Mapping[str, Mapping[str, float]]],
) -> torch.Tensor:
    if target_stats is None:
        return value
    stats = target_stats.get(key, {})
    mean = float(stats.get("mean", 0.0))
    std = float(stats.get("std", 1.0))
    return value * std + mean


def _masked_feature_mean(
    x: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    *,
    validity_mask: Optional[torch.Tensor] = None,
    default: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return feature mean, availability flag, and reliability over valid frames."""
    if x.ndim != 3 or x.size(-1) != 1:
        raise ValueError(f"x must have shape (B, T, 1), got {tuple(x.shape)}")
    if padding_mask is not None and tuple(padding_mask.shape) != tuple(x.shape[:2]):
        raise ValueError(
            "padding_mask shape must match x[:, :, 0]: "
            f"x={tuple(x.shape)}, padding_mask={tuple(padding_mask.shape)}"
        )

    valid_frames = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
    if padding_mask is not None:
        valid_frames = ~padding_mask
    if validity_mask is not None:
        if tuple(validity_mask.shape) != tuple(x.shape[:2]):
            raise ValueError(
                "validity_mask shape must match x[:, :, 0]: "
                f"x={tuple(x.shape)}, validity_mask={tuple(validity_mask.shape)}"
            )
        valid_frames = valid_frames & validity_mask

    weights = valid_frames.to(dtype=x.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1)
    mean = (x * weights).sum(dim=1) / denom.clamp(min=1.0)
    default_value = torch.full_like(mean, float(default))
    mean = torch.where(denom > 0, mean, default_value)

    total_frames = (
        torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
        if padding_mask is None
        else (~padding_mask).to(dtype=x.dtype)
    ).sum(dim=1).clamp(min=1.0)
    reliability = (denom.squeeze(-1) / total_frames).clamp(min=0.0, max=1.0)
    has_observation = denom.squeeze(-1) > 0
    return mean, has_observation, reliability


def _zero_regression_output(reference: torch.Tensor, *, floor: float = EPS) -> RegressionHeadOutput:
    mu = torch.zeros(reference.size(0), device=reference.device, dtype=reference.dtype)
    var = torch.full_like(mu, float(floor))
    return {"mu": mu, "var": var, "logvar": torch.full_like(mu, math.log(float(floor)))}


def _plausible_spacing(raw: torch.Tensor, constants: PhysicsConstants = DEFAULT_PHYSICS_CONSTANTS) -> torch.Tensor:
    return constants.spacing_min_hz + constants.spacing_range_hz * torch.sigmoid(raw)


def _plausible_vtl(raw: torch.Tensor, constants: PhysicsConstants = DEFAULT_PHYSICS_CONSTANTS) -> torch.Tensor:
    return constants.vtl_min_cm + constants.vtl_range_cm * torch.sigmoid(raw)


def _validate_class_labels(x: Optional[torch.Tensor], name: str, n_classes: int) -> None:
    """Strict validation for integer class labels."""
    if x is None:
        return

    x_float = x.to(dtype=torch.float32)
    x_long = x_float.long()
    valid = torch.isfinite(x_float) & (x_float >= 0) & (x_float == x_long.float()) & (x_long < n_classes)
    if not bool(valid.all()):
        bad = x[~valid]
        raise ValueError(f"Invalid {name} labels detected. Example values: {bad[:8]}")


def aggregate_by_speaker(
    speaker_ids: Sequence[str],
    preds: Mapping[str, torch.Tensor],
    variances: Optional[Mapping[str, Optional[torch.Tensor]]] = None,
    quality: Optional[torch.Tensor] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    method: str = "inverse_variance",
    aggregation_config: Optional[AggregationConfig] = None,
    reliability_config: Optional[ReliabilityConfig] = None,
    target_stats: Optional[Mapping[str, Mapping[str, float]]] = None,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """Aggregate utterance-level predictions into speaker-level estimates."""
    method_aliases = {
        "inverse_variance": "legacy_inverse_variance",
        "legacy_inverse_variance": "legacy_inverse_variance",
        "mean": "mean",
        "omega_robust_reliability_pool": "omega_robust_reliability_pool",
    }
    if method not in method_aliases:
        raise ValueError(
            f"Unsupported aggregation method '{method}'. "
            "Expected one of {'mean', 'inverse_variance', 'legacy_inverse_variance', 'omega_robust_reliability_pool'}."
        )
    resolved_method = method_aliases[method]
    if not preds:
        raise ValueError("preds must contain at least one prediction tensor for speaker aggregation")
    if len(speaker_ids) == 0:
        return {"utterance": {"speaker_ids": []}, "speaker": {}}

    first_tensor = next(iter(preds.values()))
    if not isinstance(first_tensor, torch.Tensor):
        raise TypeError("preds values must be tensors for speaker aggregation")
    if first_tensor.ndim == 0:
        raise ValueError("Prediction tensors must have a leading utterance dimension for speaker aggregation")
    n_items = first_tensor.size(0)
    if len(speaker_ids) != n_items:
        raise ValueError(f"speaker_ids length ({len(speaker_ids)}) does not match predictions ({n_items})")

    for key, value in preds.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Prediction '{key}' must be a tensor, got {type(value).__name__}")
        _validate_batch_axis(value, n_items, name=f"preds['{key}']")

    device = first_tensor.device
    quality_vec = (
        quality.to(device=device, dtype=torch.float32)
        if quality is not None
        else torch.ones(n_items, device=device)
    )
    _validate_batch_axis(quality_vec, n_items, name="quality", expected_ndim=1)
    if not bool(torch.isfinite(quality_vec).all()):
        raise ValueError("quality must contain only finite values")
    quality_vec = quality_vec.clamp(min=eps)

    aggregation_cfg = aggregation_config or AggregationConfig()
    reliability_cfg = reliability_config or ReliabilityConfig()
    metadata_map: Dict[str, torch.Tensor] = {}
    if metadata:
        for key, value in metadata.items():
            if value is None:
                continue
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if tensor.ndim == 0:
                tensor = tensor.expand(n_items)
            elif tensor.ndim > 1:
                tensor = tensor.reshape(n_items, -1)[:, 0]
            _validate_batch_axis(
                tensor.to(device=device),
                n_items,
                name=f"metadata['{key}']",
                expected_ndim=1,
            )
            metadata_map[key] = tensor.to(device=device, dtype=torch.float32)

    height_variance = None
    if variances is not None and variances.get("height") is not None:
        height_variance = variances["height"]
        if not isinstance(height_variance, torch.Tensor):
            raise TypeError("variances['height'] must be a tensor when provided")
        _validate_batch_axis(height_variance, n_items, name="variances['height']")
        height_variance = height_variance.to(device=device, dtype=torch.float32).clamp(min=eps)
    height_std_scale = (
        float(target_stats.get("height", {}).get("std", 1.0))
        if isinstance(target_stats, Mapping)
        else 1.0
    )
    pred_std_cm = (
        torch.sqrt(height_variance.clamp(min=eps)) * height_std_scale
        if height_variance is not None
        else None
    )

    valid_frames = metadata_map.get("valid_frames")
    if valid_frames is None:
        valid_frames = torch.ones(n_items, device=device, dtype=torch.float32)
    crop_frames = None
    if valid_frames.numel() > 0 and torch.isfinite(valid_frames).any():
        crop_frames = int(torch.nan_to_num(valid_frames, nan=1.0).max().item())
    reliability_payload = compose_handcrafted_clip_reliability(
        metadata=metadata_map,
        valid_frames=valid_frames,
        crop_frames=crop_frames,
        pred_std_cm=pred_std_cm,
        min_weight=reliability_cfg.min_weight,
        use_feature_drift=reliability_cfg.use_feature_drift,
    )
    clip_reliability = reliability_payload["clip_reliability_prior"].to(
        device=device, dtype=torch.float32
    )
    combined_quality = (quality_vec * clip_reliability).clamp(min=eps)

    grouped: Dict[str, List[int]] = {}
    for idx, speaker_id in enumerate(speaker_ids):
        grouped.setdefault(str(speaker_id), []).append(idx)

    speaker_out: Dict[str, Any] = {}
    for speaker_id, idxs in grouped.items():
        idx_tensor = torch.tensor(idxs, device=device, dtype=torch.long)
        quality_subset = quality_vec.index_select(0, idx_tensor)
        reliability_subset = clip_reliability.index_select(0, idx_tensor)
        combined_subset = combined_quality.index_select(0, idx_tensor)
        entry: Dict[str, Any] = {
            "count": len(idxs),
            "quality": quality_subset.mean(),
            "clip_reliability": reliability_subset.mean(),
        }
        for key, value in preds.items():
            vals = value.index_select(0, idx_tensor)
            var = None
            if variances is not None and key in variances and variances[key] is not None:
                variance_tensor = variances[key]
                if not isinstance(variance_tensor, torch.Tensor):
                    raise TypeError(f"Variance '{key}' must be a tensor when provided")
                _validate_batch_axis(variance_tensor, n_items, name=f"variances['{key}']")
                if not bool(torch.isfinite(variance_tensor).all()):
                    raise ValueError(f"Variance '{key}' must contain only finite values")
                var = variance_tensor.index_select(0, idx_tensor).to(device=device, dtype=torch.float32).clamp(min=eps)

            if resolved_method == "omega_robust_reliability_pool" and vals.ndim == 1:
                pooled = omega_reliability_pool(
                    vals.to(dtype=torch.float32),
                    clip_reliability=reliability_subset.to(dtype=torch.float32),
                    pred_var=var.to(dtype=torch.float32) if var is not None else None,
                    config=aggregation_cfg,
                )
                mean = pooled["mean"].to(device=device, dtype=vals.dtype)
                agg_var = pooled["var"].to(device=device, dtype=torch.float32)
                entry[f"{key}_rejected_count"] = float(pooled["rejected_count"])
                entry[f"{key}_surviving_count"] = float(pooled["surviving_count"])
                entry[f"{key}_effective_weight_sum"] = float(pooled["effective_weight_sum"])
            elif resolved_method == "legacy_inverse_variance" and var is not None:
                if vals.ndim == 1:
                    weights = (combined_subset / var).clamp(min=eps)
                    mean = (vals * weights).sum() / weights.sum().clamp(min=eps)
                    agg_var = 1.0 / weights.sum().clamp(min=eps)
                else:
                    weights = (combined_subset.unsqueeze(-1) / var).clamp(min=eps)
                    mean = (vals * weights).sum(dim=0) / weights.sum(dim=0).clamp(min=eps)
                    agg_var = 1.0 / weights.sum(dim=0).clamp(min=eps)
            else:
                if vals.ndim == 1:
                    weights = combined_subset
                    mean = (vals * weights).sum() / weights.sum().clamp(min=eps)
                    agg_var = vals.var(unbiased=False) if len(idxs) > 1 else torch.zeros((), device=device)
                else:
                    weights = combined_subset.unsqueeze(-1)
                    mean = (vals * weights).sum(dim=0) / weights.sum(dim=0).clamp(min=eps)
                    agg_var = vals.var(dim=0, unbiased=False) if len(idxs) > 1 else torch.zeros_like(mean)

            entry[key] = mean
            entry[f"{key}_var"] = agg_var
            entry[f"{key}_std"] = torch.sqrt(agg_var.clamp(min=0.0))

        if "gender_probs" in entry:
            entry["gender_pred"] = int(entry["gender_probs"].argmax(dim=-1).item())
        speaker_out[speaker_id] = entry

    return {
        "utterance": {
            "speaker_ids": list(speaker_ids),
            "quality": quality_vec,
            "clip_reliability": clip_reliability,
            "preds": dict(preds),
            "variances": dict(variances or {}),
            "metadata": metadata_map,
            "aggregation_method": resolved_method,
        },
        "speaker": speaker_out,
    }


def build_multi_crops(
    features: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    crop_size: Optional[int] = None,
    n_crops: int = 1,
) -> List[Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, int]]]:
    """Generate deterministic temporal crops for multi-crop inference."""
    mask = _validate_sequence_inputs(features, padding_mask, name="multi_crop_features")
    if int(n_crops) != n_crops:
        raise ValueError(f"n_crops must be an integer, got {n_crops}")
    n_crops = int(n_crops)
    if n_crops < 1:
        raise ValueError(f"n_crops must be >= 1, got {n_crops}")
    if crop_size is not None:
        if int(crop_size) != crop_size:
            raise ValueError(f"crop_size must be an integer when provided, got {crop_size}")
        crop_size = int(crop_size)
        if crop_size <= 0:
            raise ValueError(f"crop_size must be positive when provided, got {crop_size}")
    if crop_size is None or n_crops <= 1 or crop_size >= features.size(1):
        return [(features, mask, {"start": 0, "end": features.size(1)})]

    max_start = max(0, features.size(1) - crop_size)
    if n_crops == 2:
        starts = [0, max_start]
    else:
        starts = torch.linspace(0, max_start, steps=n_crops).round().long().tolist()
    starts = sorted(set(int(s) for s in starts))

    crops: List[Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, int]]] = []
    for start in starts:
        end = start + crop_size
        crop_features = features[:, start:end, :]
        crop_mask = None if mask is None else _sanitize_padding_mask(mask[:, start:end])
        crops.append((crop_features, crop_mask, {"start": start, "end": end}))
    return crops


__all__ = [
    "_clone_tensor_mapping",
    "_denorm_tensor",
    "_masked_feature_mean",
    "_masked_mean",
    "_plausible_spacing",
    "_plausible_vtl",
    "_safe_mean",
    "_sanitize_padding_mask",
    "_validate_batch_axis",
    "_validate_class_labels",
    "_validate_sequence_inputs",
    "_zero_regression_output",
    "aggregate_by_speaker",
    "build_multi_crops",
]
