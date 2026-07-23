"""VocalMorph V5: range-aware speaker height architecture.

V5 is a clean architecture path, not a calibration or prediction-stack phase.
It is built for the existing clip-level trainer but changes the inductive bias:

- masked multi-scale temporal encoder for pre-extracted acoustic/SSL features
- learnable height-range query tokens
- mixture-of-height experts anchored in real centimeter ranges
- source and clip-quality conditioning
- native robust height loss with NLL, bin, ranking, and speaker consistency terms

All regression outputs are in the normalized target space used by the dataset.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vocalmorph_v2.utils import aggregate_by_speaker


def _finite(x: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=value, posinf=value, neginf=value)


def _valid_mask(padding_mask: Optional[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    if padding_mask is None:
        return torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
    return ~padding_mask.to(device=x.device, dtype=torch.bool)


def _mask_zero(x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if padding_mask is None:
        return x
    return x.masked_fill(padding_mask.unsqueeze(-1).to(device=x.device), 0.0)


def _masked_mean(x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
    valid = _valid_mask(padding_mask, x).unsqueeze(-1).to(dtype=x.dtype)
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def _masked_std(x: torch.Tensor, padding_mask: Optional[torch.Tensor], mean: Optional[torch.Tensor] = None) -> torch.Tensor:
    valid = _valid_mask(padding_mask, x).unsqueeze(-1).to(dtype=x.dtype)
    if mean is None:
        mean = _masked_mean(x, padding_mask)
    var = ((x - mean.unsqueeze(1)).pow(2) * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    return torch.sqrt(var.clamp_min(1e-6))


def _masked_max(x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if padding_mask is None:
        return x.max(dim=1).values
    masked = x.masked_fill(padding_mask.unsqueeze(-1).to(device=x.device), -1e4)
    return masked.max(dim=1).values


def _height_bin_ids(height_cm: torch.Tensor, n_bins: int) -> torch.Tensor:
    if n_bins <= 3:
        bins = torch.bucketize(height_cm, torch.tensor([165.0, 178.0], device=height_cm.device))
    else:
        edges = torch.tensor([155.0, 165.0, 172.0, 180.0], device=height_cm.device)
        bins = torch.bucketize(height_cm, edges)
    return bins.clamp(min=0, max=n_bins - 1).long()


class SwiGLUBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, hidden * 2)
        self.down = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.up(self.norm(x)).chunk(2, dim=-1)
        return x + self.dropout(self.down(F.silu(a) * b))


class DepthwiseTemporalBlock(nn.Module):
    def __init__(self, dim: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim, bias=False)
        self.pointwise = nn.Conv1d(dim, dim * 2, 1)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        y = _mask_zero(self.norm(x), padding_mask).transpose(1, 2)
        y = self.depthwise(y)
        y = self.pointwise(y).transpose(1, 2)
        a, b = y.chunk(2, dim=-1)
        y = self.out(F.silu(a) * b)
        return _mask_zero(residual + self.dropout(y), padding_mask)


class RangeAwareConformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int, kernel: int, dropout: float, stochastic_depth: float = 0.0) -> None:
        super().__init__()
        self.stochastic_depth = float(stochastic_depth)
        self.ff1 = SwiGLUBlock(dim, dim * ff_mult, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = DepthwiseTemporalBlock(dim, kernel, dropout)
        self.ff2 = SwiGLUBlock(dim, dim * ff_mult, dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor], is_training: bool = True) -> torch.Tensor:
        # Stochastic depth: randomly drop entire block during training
        if is_training and self.stochastic_depth > 0.0 and self.training:
            if torch.rand(1, device=x.device).item() < self.stochastic_depth:
                return x
        x = self.ff1(x)
        residual = x
        y = self.attn_norm(x)
        y, _ = self.attn(y, y, y, key_padding_mask=padding_mask, need_weights=False)
        x = _mask_zero(residual + self.attn_drop(y), padding_mask)
        x = self.conv(x, padding_mask)
        x = self.ff2(x)
        return _mask_zero(self.out_norm(x), padding_mask)


class RangeQueryPooling(nn.Module):
    def __init__(self, dim: int, heads: int, n_queries: int, dropout: float) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.proj = nn.Sequential(
            nn.LayerNorm(dim * n_queries + dim * 3),
            nn.Linear(dim * n_queries + dim * 3, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        bsz = x.size(0)
        q = self.queries.unsqueeze(0).expand(bsz, -1, -1)
        tokens, _ = self.attn(q, self.norm(x), self.norm(x), key_padding_mask=padding_mask, need_weights=False)
        mean = _masked_mean(x, padding_mask)
        std = _masked_std(x, padding_mask, mean)
        maxv = _masked_max(x, padding_mask)
        return self.proj(torch.cat([tokens.reshape(bsz, -1), mean, std, maxv], dim=-1))


class MetadataEncoder(nn.Module):
    def __init__(self, source_embed_dim: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.source_embed = nn.Embedding(4, source_embed_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(10 + source_embed_dim),
            nn.Linear(10 + source_embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        *,
        batch_size: int,
        device: torch.device,
        domain: Optional[torch.Tensor],
        clip_metadata: Optional[Mapping[str, Any]],
    ) -> torch.Tensor:
        if domain is None:
            source = torch.ones(batch_size, device=device, dtype=torch.long)
        else:
            source = domain.to(device=device, dtype=torch.long).clamp(0, 3)
        meta = clip_metadata or {}

        def scalar(key: str, default: float = 0.0) -> torch.Tensor:
            value = meta.get(key)
            if not isinstance(value, torch.Tensor):
                return torch.full((batch_size,), default, device=device, dtype=torch.float32)
            value = value.to(device=device, dtype=torch.float32).reshape(batch_size)
            return _finite(value, default)

        valid_frames = scalar("valid_frames", 1.0).clamp_min(1.0)
        pieces = [
            scalar("duration_s", 0.0) / 10.0,
            scalar("speech_ratio", 0.0),
            scalar("snr_db_estimate", 0.0) / 35.0,
            scalar("capture_quality_score", 0.5),
            scalar("voiced_ratio", 0.0),
            scalar("clipped_ratio", 0.0) * 10.0,
            scalar("distance_cm_estimate", 0.0) / 100.0,
            scalar("distance_confidence", 0.0),
            scalar("quality_ok", 1.0),
            torch.log1p(valid_frames) / 7.0,
        ]
        meta_vec = torch.stack(pieces, dim=-1).clamp(-5.0, 5.0)
        return self.net(torch.cat([meta_vec, self.source_embed(source)], dim=-1))


class AdaptiveBinWeights:
    """Tracks running per-height-bin MAE and computes adaptive loss weights.
    
    Dynamically adjusts short/medium/tall bin weights so the worst-performing
    bin gets upweighted. Uses momentum-based running statistics to avoid
    oscillation from batch noise.
    """

    def __init__(self, momentum: float = 0.95, ramp_epochs: int = 20) -> None:
        self.momentum = float(momentum)
        self.ramp_epochs = max(1, int(ramp_epochs))
        self.short_mae: Optional[torch.Tensor] = None
        self.medium_mae: Optional[torch.Tensor] = None
        self.tall_mae: Optional[torch.Tensor] = None
        self.n_updates: int = 0

    @torch.no_grad()
    def update(self, err: torch.Tensor, height_cm: torch.Tensor) -> None:
        """Update running per-bin MAE from a batch."""
        err = err.detach().abs().float()
        h = height_cm.detach().float()
        mask_short = h < 165.0
        mask_medium = (h >= 165.0) & (h < 178.0)
        mask_tall = h >= 178.0

        def _update_running(current: Optional[torch.Tensor], mask: torch.Tensor) -> Optional[torch.Tensor]:
            if not mask.any():
                return current
            batch_mae = err[mask].mean()
            if current is None:
                return batch_mae.clone()
            return self.momentum * current + (1.0 - self.momentum) * batch_mae

        self.short_mae = _update_running(self.short_mae, mask_short)
        self.medium_mae = _update_running(self.medium_mae, mask_medium)
        self.tall_mae = _update_running(self.tall_mae, mask_tall)
        self.n_updates += 1

    def get_weights(
        self,
        *,
        short_base: float = 2.0,
        medium_base: float = 1.0,
        tall_base: float = 1.2,
        max_ratio: float = 5.0,
        min_ratio: float = 0.6,
    ) -> Tuple[float, float, float]:
        """
        Compute adaptive per-bin weights based on running MAE.
        
        The idea: if short speakers have 2x the error of medium speakers,
        the short weight should be ~2x the medium weight (capped by max_ratio).
        """
        if self.n_updates < 10 or any(m is None for m in (self.short_mae, self.medium_mae, self.tall_mae)):
            return float(short_base), float(medium_base), float(tall_base)

        maes = torch.tensor([self.short_mae.item(), self.medium_mae.item(), self.tall_mae.item()])
        max_mae = float(maes.max().clamp_min(0.01))
        min_mae = float(maes.min().clamp_min(0.01))

        # How much worse is each bin relative to the best-performing bin?
        ratios = [float(m / min_mae) for m in maes]

        # Blend from static base weights toward adaptive weights over ramp_epochs
        blend = min(1.0, float(self.n_updates) / float(self.ramp_epochs))

        bases = [float(short_base), float(medium_base), float(tall_base)]
        adaptive = [r * float(medium_base) for r in ratios]

        weights = []
        for i in range(3):
            w = (1.0 - blend) * bases[i] + blend * adaptive[i]
            w = max(float(min_ratio), min(float(max_ratio), w))
            weights.append(w)

        return tuple(weights)


class VocalMorphV5Loss(nn.Module):
    def __init__(
        self,
        *,
        height_nll_weight: float = 0.25,
        height_huber_weight: float = 0.55,
        height_mae_weight: float = 0.30,
        weight_weight: float = 0.04,
        age_weight: float = 0.02,
        gender_weight: float = 0.08,
        height_bin_weight: float = 0.25,
        ranking_weight: float = 0.0,
        speaker_consistency_weight: float = 0.05,
        gate_entropy_weight: float = 0.0,
        short_weight: float = 2.0,
        medium_weight: float = 1.00,
        tall_weight: float = 1.20,
        huber_delta_norm: float = 0.50,
        rank_margin_cm: float = 2.5,
        adaptive_weighting: bool = True,
        adaptive_ramp_epochs: int = 20,
    ) -> None:
        super().__init__()
        self.height_nll_weight = float(height_nll_weight)
        self.height_huber_weight = float(height_huber_weight)
        self.height_mae_weight = float(height_mae_weight)
        self.weight_weight = float(weight_weight)
        self.age_weight = float(age_weight)
        self.gender_weight = float(gender_weight)
        self.height_bin_weight = float(height_bin_weight)
        self.ranking_weight = float(ranking_weight)
        self.speaker_consistency_weight = float(speaker_consistency_weight)
        self.gate_entropy_weight = float(gate_entropy_weight)
        self.short_weight = float(short_weight)
        self.medium_weight = float(medium_weight)
        self.tall_weight = float(tall_weight)
        self.huber_delta_norm = float(huber_delta_norm)
        self.rank_margin_cm = float(rank_margin_cm)
        self.adaptive_weighting = bool(adaptive_weighting)
        self.register_buffer("height_mean", torch.tensor(170.0, dtype=torch.float32))
        self.register_buffer("height_std", torch.tensor(9.0, dtype=torch.float32))
        self.adaptive_weights = AdaptiveBinWeights(momentum=0.95, ramp_epochs=adaptive_ramp_epochs) if adaptive_weighting else None

    def set_target_stats(self, target_stats: Optional[Mapping[str, Mapping[str, float]]]) -> None:
        if not target_stats:
            return
        stats = target_stats.get("height", {})
        self.height_mean.fill_(float(stats.get("mean", float(self.height_mean.item()))))
        self.height_std.fill_(max(float(stats.get("std", float(self.height_std.item()))), 1e-3))

    def _height_weights(self, targets: Mapping[str, torch.Tensor], pred: torch.Tensor) -> torch.Tensor:
        raw = targets.get("height_raw")
        if raw is None:
            raw = pred * self.height_std + self.height_mean
        raw = raw.to(device=pred.device, dtype=torch.float32)
        
        # Use adaptive weights if enabled (dynamically adjusts based on per-bin error)
        if self.adaptive_weighting and self.adaptive_weights is not None:
            sw, mw, tw = self.adaptive_weights.get_weights(
                short_base=self.short_weight,
                medium_base=self.medium_weight,
                tall_base=self.tall_weight,
            )
        else:
            sw, mw, tw = self.short_weight, self.medium_weight, self.tall_weight
        
        weights = torch.full_like(pred, float(mw))
        weights = torch.where(raw < 165.0, torch.full_like(weights, float(sw)), weights)
        weights = torch.where(raw >= 178.0, torch.full_like(weights, float(tw)), weights)
        return weights

    def _ranking_loss(self, pred: torch.Tensor, target: torch.Tensor, raw_target: Optional[torch.Tensor]) -> torch.Tensor:
        if pred.numel() < 2 or self.ranking_weight <= 0.0:
            return pred.new_zeros(())
        raw = raw_target.to(device=pred.device, dtype=torch.float32) if raw_target is not None else target * self.height_std + self.height_mean
        diff_cm = raw[:, None] - raw[None, :]
        valid = diff_cm.abs() >= max(self.rank_margin_cm, 1e-3)
        if not bool(valid.any()):
            return pred.new_zeros(())
        sign = diff_cm.sign()
        pred_diff = pred[:, None] - pred[None, :]
        margin = self.rank_margin_cm / self.height_std.clamp_min(1e-3)
        loss = F.relu(margin - sign * pred_diff)
        return loss[valid].mean()

    def _speaker_consistency(self, pred: torch.Tensor, target: torch.Tensor, speaker_idx: Optional[torch.Tensor]) -> torch.Tensor:
        if speaker_idx is None or pred.numel() < 2 or self.speaker_consistency_weight <= 0.0:
            return pred.new_zeros(())
        speaker_idx = speaker_idx.to(device=pred.device, dtype=torch.long)
        losses = []
        for sid in torch.unique(speaker_idx):
            if int(sid.item()) < 0:
                continue
            mask = speaker_idx == sid
            if int(mask.sum().item()) < 2:
                continue
            group_pred = pred[mask]
            group_target = target[mask]
            pooled = group_pred.mean()
            losses.append((pooled - group_target.mean()).abs() + 0.25 * group_pred.var(unbiased=False))
        if not losses:
            return pred.new_zeros(())
        return torch.stack(losses).mean()

    def forward(self, preds: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        height = preds["height"].float()
        target_h = targets["height"].to(device=height.device, dtype=torch.float32)
        h_weights = self._height_weights(targets, height)
        h_weights = h_weights / h_weights.mean().clamp_min(1e-6)
        err = height - target_h
        
        # Update adaptive weight tracking (training only — validation data would leak)
        if self.training and self.adaptive_weighting and self.adaptive_weights is not None:
            height_cm = targets.get("height_raw")
            if height_cm is None:
                height_cm = target_h * self.height_std + self.height_mean
            else:
                height_cm = height_cm.to(device=height.device, dtype=torch.float32)
            self.adaptive_weights.update(err, height_cm)
        var = preds.get("height_var", torch.ones_like(height)).float().clamp(1e-4, 50.0)
        nll = 0.5 * (err.pow(2) / var + torch.log(var))
        height_nll = (nll * h_weights).mean()
        height_huber = (F.huber_loss(height, target_h, delta=self.huber_delta_norm, reduction="none") * h_weights).mean()
        height_mae = (err.abs() * h_weights).mean()
        height_loss = (
            self.height_nll_weight * height_nll
            + self.height_huber_weight * height_huber
            + self.height_mae_weight * height_mae
        )

        weight_loss = height.new_zeros(())
        if "weight" in preds and "weight" in targets:
            mask = targets.get("weight_mask", torch.ones_like(target_h)).to(device=height.device, dtype=torch.float32)
            if bool((mask > 0).any()):
                raw = F.smooth_l1_loss(preds["weight"].float(), targets["weight"].float(), reduction="none")
                weight_loss = (raw * mask).sum() / mask.sum().clamp_min(1.0)

        age_loss = height.new_zeros(())
        if "age" in preds and "age" in targets:
            age_valid = torch.isfinite(targets["age"].float())
            if age_valid.any():
                age_loss = F.smooth_l1_loss(preds["age"].float()[age_valid], targets["age"].float()[age_valid])

        gender_loss = height.new_zeros(())
        if "gender_logits" in preds and "gender" in targets:
            gender_valid = torch.isfinite(targets["gender"].float())
            if gender_valid.any():
                gender_loss = F.cross_entropy(preds["gender_logits"][gender_valid], targets["gender"].long()[gender_valid])

        bin_loss = height.new_zeros(())
        if "height_bin_logits" in preds and "height_raw" in targets:
            bin_valid = torch.isfinite(targets["height_raw"].float())
            if bin_valid.any():
                labels = _height_bin_ids(targets["height_raw"].float()[bin_valid], preds["height_bin_logits"].shape[-1])
                bin_loss = F.cross_entropy(preds["height_bin_logits"][bin_valid], labels)

        ranking = self._ranking_loss(height, target_h, targets.get("height_raw"))
        consistency = self._speaker_consistency(height, target_h, targets.get("speaker_idx"))
        gate_entropy = height.new_zeros(())
        if "height_gate" in preds:
            gate = preds["height_gate"].clamp_min(1e-8)
            gate_entropy = -(gate * gate.log()).sum(dim=-1).mean()

        total = (
            height_loss
            + self.weight_weight * weight_loss
            + self.age_weight * age_loss
            + self.gender_weight * gender_loss
            + self.height_bin_weight * bin_loss
            + self.ranking_weight * ranking
            + self.speaker_consistency_weight * consistency
            + self.gate_entropy_weight * gate_entropy
        )
        return {
            "total": total,
            "height": height_loss,
            "height_nll": height_nll.detach(),
            "height_mae_proxy": height_mae.detach(),
            "weight": weight_loss,
            "age": age_loss,
            "gender": gender_loss,
            "height_bin": bin_loss,
            "height_ranking": ranking,
            "speaker_consistency": consistency,
            "height_gate_entropy": gate_entropy.detach(),
        }


class VocalMorphV5(nn.Module):
    expects_domain = True

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_blocks: int = 8,
        ff_mult: int = 4,
        conv_kernel: int = 15,
        dropout: float = 0.20,
        stochastic_depth: float = 0.05,
        n_height_experts: int = 7,
        expert_delta_cm: float = 12.0,
        direct_blend: float = 0.15,
        meta_dim: int = 80,
        height_bin_classes: int = 5,
        loss_cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.n_height_experts = int(n_height_experts)
        self.expert_delta_cm = float(expert_delta_cm)
        self.direct_blend = float(direct_blend)
        self.height_bin_classes = int(height_bin_classes)
        self.stochastic_depth = float(stochastic_depth)

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        kernels = (7, 15, 31)
        self.conv_stem = nn.ModuleList(DepthwiseTemporalBlock(d_model, k, dropout) for k in kernels)
        self.blocks = nn.ModuleList(
            RangeAwareConformerBlock(d_model, n_heads, ff_mult, conv_kernel, dropout, stochastic_depth=stochastic_depth)
            for _ in range(n_blocks)
        )
        self.pool = RangeQueryPooling(d_model, n_heads, n_queries=n_height_experts, dropout=dropout)
        self.meta = MetadataEncoder(source_embed_dim=16, hidden=max(64, meta_dim), out_dim=meta_dim, dropout=dropout)
        fused_dim = d_model + meta_dim
        self.fuse = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        anchors = torch.linspace(156.0, 184.0, steps=n_height_experts)
        if n_height_experts == 5:
            anchors = torch.tensor([154.0, 162.0, 169.5, 178.0, 188.0], dtype=torch.float32)
        elif n_height_experts == 7:
            anchors = torch.tensor([150.0, 157.0, 164.0, 171.0, 178.0, 185.0, 192.0], dtype=torch.float32)
        self.register_buffer("height_anchors_cm", anchors.float())
        self.register_buffer("height_mean", torch.tensor(170.0, dtype=torch.float32))
        self.register_buffer("height_std", torch.tensor(9.0, dtype=torch.float32))

        self.height_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_height_experts),
        )
        self.height_experts = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            for _ in range(n_height_experts)
        )
        self.height_direct = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
        self.height_logvar = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
        self.height_bin_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, height_bin_classes))
        self.weight_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
        self.age_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
        self.gender_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 2))
        self.quality_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1), nn.Sigmoid())

        self.loss_module = VocalMorphV5Loss(**dict(loss_cfg or {}))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight, gain=0.85)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        # Start direct height around zero-normalized residual; anchored experts carry the prior.
        final = self.height_direct[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        for expert in self.height_experts:
            expert_final = expert[-1]
            if isinstance(expert_final, nn.Linear):
                nn.init.zeros_(expert_final.weight)
                nn.init.zeros_(expert_final.bias)
        # Init height_bin_head with small weights
        for m in self.height_bin_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_target_stats(self, target_stats: Optional[Mapping[str, Mapping[str, float]]]) -> None:
        if not target_stats:
            return
        h = target_stats.get("height", {})
        self.height_mean.fill_(float(h.get("mean", float(self.height_mean.item()))))
        self.height_std.fill_(max(float(h.get("std", float(self.height_std.item()))), 1e-3))
        self.loss_module.set_target_stats(target_stats)

    def encode(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        domain: Optional[torch.Tensor] = None,
        clip_metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        is_training = self.training
        x = _finite(features.float())
        x = self.input_proj(self.input_norm(x))
        x = _mask_zero(x, padding_mask)
        for conv in self.conv_stem:
            x = conv(x, padding_mask)
        for block in self.blocks:
            x = block(x, padding_mask, is_training=is_training)
        pooled = self.pool(x, padding_mask)
        meta = self.meta(batch_size=x.size(0), device=x.device, domain=domain, clip_metadata=clip_metadata)
        return self.fuse(torch.cat([pooled, meta], dim=-1))

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        domain: Optional[torch.Tensor] = None,
        lambda_grl: float = 0.0,
        clip_metadata: Optional[Mapping[str, Any]] = None,
        targets: Optional[Mapping[str, torch.Tensor]] = None,
        current_epoch: Optional[int] = None,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del lambda_grl, current_epoch, return_aux
        emb = self.encode(features, padding_mask, domain=domain, clip_metadata=clip_metadata)
        gate = torch.softmax(self.height_gate(emb), dim=-1)
        anchors_norm = (self.height_anchors_cm.to(device=emb.device) - self.height_mean) / self.height_std.clamp_min(1e-3)
        deltas = torch.cat([expert(emb) for expert in self.height_experts], dim=-1)
        deltas = torch.tanh(deltas) * (self.expert_delta_cm / self.height_std.clamp_min(1e-3))
        expert_preds = anchors_norm.unsqueeze(0) + deltas
        moe_height = (gate * expert_preds).sum(dim=-1)
        direct_height = self.height_direct(emb).squeeze(-1)
        height = (1.0 - self.direct_blend) * moe_height + self.direct_blend * direct_height
        height_logvar = self.height_logvar(emb).squeeze(-1).clamp(-5.0, 3.0)
        height_var = torch.exp(height_logvar).clamp_min(1e-4)

        out: Dict[str, torch.Tensor] = {
            "height": height,
            "height_var": height_var,
            "height_logvar": height_logvar,
            "height_gate": gate,
            "height_expert_preds": expert_preds,
            "height_bin_logits": self.height_bin_head(emb),
            "weight": self.weight_head(emb).squeeze(-1),
            "weight_var": torch.ones_like(height_var),
            "age": self.age_head(emb).squeeze(-1),
            "age_var": torch.ones_like(height_var),
            "gender_logits": self.gender_head(emb),
            "quality_score": self.quality_head(emb).squeeze(-1).clamp(0.05, 1.0),
            "embedding": emb,
        }
        if targets is not None:
            out["losses"] = self.loss_module(out, targets)
        return out

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        domain: Optional[torch.Tensor] = None,
        speaker_ids: Optional[Sequence[str]] = None,
        clip_metadata: Optional[Mapping[str, Any]] = None,
        deterministic: bool = True,
        n_samples: int = 1,
        crop_size: Optional[int] = None,
        n_crops: int = 1,
    ) -> Dict[str, Any]:
        del speaker_ids, deterministic, crop_size, n_crops
        was_training = self.training
        if n_samples <= 1:
            self.eval()
            out = self.forward(features, padding_mask, domain=domain, clip_metadata=clip_metadata)
            if was_training:
                self.train()
            probs = torch.softmax(out["gender_logits"], dim=-1)
            return {
                "height": {"mean": out["height"], "var": out["height_var"]},
                "weight": {"mean": out["weight"], "var": out["weight_var"]},
                "age": {"mean": out["age"], "var": out["age_var"]},
                "gender": {"probs": probs, "pred": probs.argmax(dim=-1)},
                "utterance": {"quality_score": out["quality_score"]},
            }
        self.train()
        outputs = [self.forward(features, padding_mask, domain=domain, clip_metadata=clip_metadata) for _ in range(int(n_samples))]
        if not was_training:
            self.eval()
        height = torch.stack([o["height"] for o in outputs], dim=0)
        weight = torch.stack([o["weight"] for o in outputs], dim=0)
        age = torch.stack([o["age"] for o in outputs], dim=0)
        logits = torch.stack([o["gender_logits"] for o in outputs], dim=0)
        quality = torch.stack([o["quality_score"] for o in outputs], dim=0).mean(dim=0)
        probs = torch.softmax(logits, dim=-1).mean(dim=0)
        return {
            "height": {"mean": height.mean(dim=0), "var": height.var(dim=0, unbiased=False).clamp_min(1e-4)},
            "weight": {"mean": weight.mean(dim=0), "var": weight.var(dim=0, unbiased=False).clamp_min(1e-4)},
            "age": {"mean": age.mean(dim=0), "var": age.var(dim=0, unbiased=False).clamp_min(1e-4)},
            "gender": {"probs": probs, "pred": probs.argmax(dim=-1)},
            "utterance": {"quality_score": quality},
        }

    def aggregate_by_speaker(
        self,
        speaker_ids: Sequence[str],
        preds: Mapping[str, torch.Tensor],
        variances: Optional[Mapping[str, Optional[torch.Tensor]]] = None,
        quality: Optional[torch.Tensor] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        method: str = "legacy_inverse_variance",
    ) -> Dict[str, Any]:
        return aggregate_by_speaker(
            speaker_ids=speaker_ids,
            preds=preds,
            variances=variances,
            quality=quality,
            metadata=metadata,
            method=method,
            target_stats={
                "height": {"mean": float(self.height_mean.item()), "std": float(self.height_std.item())}
            },
        )


def build_v5_model(config: Mapping[str, Any]) -> VocalMorphV5:
    model_cfg = dict(config.get("model", {}).get("v5", {}))
    loss_cfg = dict(config.get("training", {}).get("loss", {}).get("v5", {}))
    return VocalMorphV5(
        input_dim=int(config.get("model", {}).get("input_dim", model_cfg.get("input_dim", 264))),
        d_model=int(model_cfg.get("d_model", 256)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        n_blocks=int(model_cfg.get("n_blocks", 8)),
        ff_mult=int(model_cfg.get("ff_mult", 4)),
        conv_kernel=int(model_cfg.get("conv_kernel", 15)),
        dropout=float(model_cfg.get("dropout", 0.20)),
        stochastic_depth=float(model_cfg.get("stochastic_depth", 0.05)),
        n_height_experts=int(model_cfg.get("n_height_experts", 7)),
        expert_delta_cm=float(model_cfg.get("expert_delta_cm", 12.0)),
        direct_blend=float(model_cfg.get("direct_blend", 0.15)),
        meta_dim=int(model_cfg.get("meta_dim", 80)),
        height_bin_classes=int(model_cfg.get("height_bin_classes", 5)),
        loss_cfg=loss_cfg,
    )


__all__ = ["VocalMorphV5", "VocalMorphV5Loss", "build_v5_model"]
