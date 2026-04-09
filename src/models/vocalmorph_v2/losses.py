"""Losses for VocalMorph V2."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    AblationToggles,
    AggregationConfig,
    DEFAULT_LOSS_WEIGHTS,
    DEFAULT_PHYSICS_CONSTANTS,
    LossWeights,
    PhysicsConstants,
    ReliabilityConfig,
    SpeakerAlignmentConfig,
)
from .reliability import omega_reliability_pool
from .utils import _denorm_tensor


class VocalTractSimulatorLossV2(nn.Module):
    """Multi-task loss with optional anthropometric targets and ablation toggles."""

    def __init__(
        self,
        vtl_height_ratio: float = 6.7,
        robust_huber_weight: float = 0.20,
        nll_floor: float = 1e-6,
        label_smoothing: float = 0.10,
        ranking_margin: float = 0.10,
        diversity_scale: float = 0.01,
        focal_after_epoch: int = 20,
        focal_ema_decay: float = 0.95,
        acoustic_physics_weight: float = 0.05,
        speaker_contrastive_temperature: float = 0.2,
        target_stats: Optional[Mapping[str, Mapping[str, float]]] = None,
        constants: PhysicsConstants = DEFAULT_PHYSICS_CONSTANTS,
        toggles: Optional[AblationToggles] = None,
        loss_weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
        aggregation_config: Optional[AggregationConfig] = None,
        reliability_config: Optional[ReliabilityConfig] = None,
        speaker_alignment: Optional[SpeakerAlignmentConfig] = None,
    ):
        super().__init__()
        self.constants = constants
        self.toggles = toggles or AblationToggles()
        self.loss_weights = loss_weights
        self.vtl_height_ratio = float(vtl_height_ratio)
        self.robust_huber_weight = float(robust_huber_weight)
        self.nll_floor = float(nll_floor)
        self.label_smoothing = float(label_smoothing)
        self.ranking_margin = float(ranking_margin)
        self.diversity_scale = float(diversity_scale)
        self.focal_after_epoch = int(focal_after_epoch)
        self.focal_ema_decay = float(focal_ema_decay)
        self.acoustic_physics_weight = float(acoustic_physics_weight)
        self.speaker_contrastive_temperature = float(speaker_contrastive_temperature)
        self.target_stats = target_stats
        self.aggregation_config = aggregation_config or AggregationConfig()
        self.reliability_config = reliability_config or ReliabilityConfig()
        self.speaker_alignment = speaker_alignment or SpeakerAlignmentConfig()
        self.current_epoch = 0
        self.running_height_mae = 1.0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def set_target_stats(
        self, target_stats: Optional[Mapping[str, Mapping[str, float]]]
    ) -> None:
        self.target_stats = target_stats

    def _get_target_stats(
        self, preds: Mapping[str, Any]
    ) -> Optional[Mapping[str, Mapping[str, float]]]:
        if self.target_stats is not None:
            return self.target_stats
        stats = preds.get("target_stats")
        if isinstance(stats, Mapping):
            return stats
        return None

    def _smooth_ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return torch.zeros((), device=logits.device)
        n_classes = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.full_like(
                logits, self.label_smoothing / max(1, n_classes - 1)
            )
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(true_dist * log_probs).sum(dim=-1).mean()

    def _pairwise_height_ranking_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if not self.toggles.use_ranking_loss:
            return torch.zeros((), device=preds["height_mu"].device)

        pred_height = preds["height_mu"]
        target_height_raw = targets.get("height_raw")
        if target_height_raw is None:
            target_height_raw = targets.get("height")
        if target_height_raw is None:
            return torch.zeros((), device=pred_height.device)

        target_height_raw = target_height_raw.to(pred_height.device)
        valid = torch.isfinite(target_height_raw)
        if valid.sum() < 2:
            return torch.zeros((), device=pred_height.device)

        pred_valid = pred_height[valid]
        true_valid = target_height_raw[valid]
        diff_true = true_valid.unsqueeze(0) - true_valid.unsqueeze(1)
        diff_pred = pred_valid.unsqueeze(0) - pred_valid.unsqueeze(1)
        pair_mask = torch.triu(
            torch.ones_like(diff_true, dtype=torch.bool), diagonal=1
        ) & (diff_true.abs() > self.constants.ranking_height_threshold_cm)
        if pair_mask.sum() == 0:
            return torch.zeros((), device=pred_height.device)

        signs = diff_true[pair_mask].sign()
        pairwise = F.relu(self.ranking_margin - signs * diff_pred[pair_mask])
        return pairwise.mean()

    def _probabilistic_regression_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        key: str,
        device: torch.device,
        *,
        mask_key: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target = targets.get(key)
        mu = preds.get(f"{key}_mu")
        logvar = preds.get(f"{key}_logvar")
        if target is None or mu is None or logvar is None:
            return torch.zeros((), device=device), torch.zeros((), device=device)

        target = target.to(device)
        valid = torch.isfinite(target)
        if mask_key is not None and targets.get(mask_key) is not None:
            valid = valid & (targets[mask_key].to(device) > 0.5)
        if valid.sum() == 0:
            return torch.zeros((), device=device), torch.zeros((), device=device)

        mu_valid = mu[valid]
        logvar_valid = logvar[valid].clamp(min=math.log(self.nll_floor), max=6.0)
        target_valid = target[valid]
        var_valid = logvar_valid.exp().clamp(min=self.nll_floor)
        residual = (target_valid - mu_valid).abs()
        nll = 0.5 * ((target_valid - mu_valid).pow(2) / var_valid + logvar_valid)
        huber = F.smooth_l1_loss(mu_valid, target_valid, reduction="none")
        per_sample = nll + self.robust_huber_weight * huber
        return per_sample.mean(), residual.mean()

    def _height_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss, mae = self._probabilistic_regression_loss(
            preds, targets, "height", device
        )
        target = targets.get("height")
        if target is None:
            return loss, mae

        target = target.to(device)
        valid = torch.isfinite(target)
        if valid.sum() == 0:
            return loss, mae

        mu = preds["height_mu"][valid]
        logvar = preds["height_logvar"][valid].clamp(
            min=math.log(self.nll_floor), max=6.0
        )
        target = target[valid]
        var = logvar.exp().clamp(min=self.nll_floor)
        residual = (target - mu).abs()
        nll = 0.5 * ((target - mu).pow(2) / var + logvar)
        huber = F.smooth_l1_loss(mu, target, reduction="none")
        per_sample = nll + self.robust_huber_weight * huber

        if torch.is_grad_enabled():
            batch_mae = (
                float(residual.mean().detach().item())
                if residual.numel() > 0
                else self.running_height_mae
            )
            self.running_height_mae = self.focal_ema_decay * self.running_height_mae + (
                1.0 - self.focal_ema_decay
            ) * max(batch_mae, 1e-3)

        current_epoch = int(
            targets.get("epoch", preds.get("current_epoch", self.current_epoch))
        )
        if current_epoch >= self.focal_after_epoch:
            denom = max(self.running_height_mae, 1e-3)
            weights = (residual.detach() / denom).clamp(1.0, 3.0)
            per_sample = per_sample * weights

        return per_sample.mean(), residual.mean()

    def _gender_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        target = targets.get("gender")
        logits = preds.get("gender_logits")
        if target is None or logits is None:
            return torch.zeros((), device=device)
        target_float = target.to(device=device, dtype=torch.float32)
        target_long = target_float.long()
        valid = (
            torch.isfinite(target_float)
            & (target_float >= 0)
            & (target_float == target_long.float())
            & (target_long < logits.size(-1))
        )
        if valid.sum() == 0:
            return torch.zeros((), device=device)
        return self._smooth_ce(logits[valid], target_long[valid])

    def _domain_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if not self.toggles.use_domain_adv:
            return torch.zeros((), device=device)
        logits = preds.get("domain_logits")
        target = targets.get("domain")
        if logits is None or target is None:
            return torch.zeros((), device=device)
        target_float = target.to(device=device, dtype=torch.float32)
        target_long = target_float.long()
        valid = (
            torch.isfinite(target_float)
            & (target_float >= 0)
            & (target_float == target_long.float())
            & (target_long < logits.size(-1))
        )
        if valid.sum() == 0:
            return torch.zeros((), device=device)
        return F.cross_entropy(logits[valid], target_long[valid])

    def _vtsl_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if not self.toggles.use_physics_branch:
            return torch.zeros((), device=device)

        vtl_pred = preds.get("vtl_pred")
        if vtl_pred is None:
            return torch.zeros((), device=device)

        target_height_raw = targets.get("height_raw")
        if target_height_raw is None:
            target_height = targets.get("height")
            if target_height is None:
                return torch.zeros((), device=device)
            stats = self._get_target_stats(preds)
            target_height_raw = _denorm_tensor(
                target_height.to(device), "height", stats
            )
        else:
            target_height_raw = target_height_raw.to(device)

        valid = torch.isfinite(target_height_raw) & torch.isfinite(vtl_pred)
        if valid.sum() == 0:
            return torch.zeros((), device=device)
        vtl_theory = target_height_raw[valid] / self.vtl_height_ratio
        supervision = F.mse_loss(vtl_pred[valid], vtl_theory)

        pred_height_raw = preds.get("height_cm")
        if pred_height_raw is None:
            stats = self._get_target_stats(preds)
            pred_height_raw = _denorm_tensor(preds["height_mu"], "height", stats)
        pred_height_raw = pred_height_raw.to(device)

        multi_formant_vtl = preds.get(
            "speaker_mean_multi_formant_vtl", preds.get("multi_formant_vtl", vtl_pred)
        )
        soft_height_target = self.vtl_height_ratio * multi_formant_vtl.to(device)
        consistency_valid = torch.isfinite(pred_height_raw) & torch.isfinite(
            soft_height_target
        )
        if consistency_valid.sum() == 0:
            return supervision

        consistency = F.smooth_l1_loss(
            pred_height_raw[consistency_valid],
            soft_height_target[consistency_valid],
            reduction="none",
        )
        vtl_confidence = preds.get(
            "speaker_mean_vtl_confidence", preds.get("vtl_confidence")
        )
        formant_stability = preds.get(
            "speaker_mean_formant_stability", preds.get("formant_stability_score")
        )
        if vtl_confidence is not None and formant_stability is not None:
            reliability = (
                torch.stack(
                    [
                        vtl_confidence.to(device=device),
                        formant_stability.to(device=device),
                    ],
                    dim=-1,
                )
                .mean(dim=-1)
                .clamp(min=0.0, max=1.0)
            )
            consistency = consistency * (0.5 + 0.5 * reliability[consistency_valid])

        return supervision + 0.35 * consistency.mean()

    def _physics_penalty(
        self, preds: Mapping[str, torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        if not self.toggles.use_physics_branch:
            return torch.zeros((), device=device)

        spacing = preds.get("formant_spacing_pred")
        vtl_pred = preds.get("vtl_pred")
        penalty = torch.zeros((), device=device)
        if spacing is not None:
            penalty = (
                penalty + F.relu(self.constants.spacing_min_hz - spacing).mean() * 0.01
            )
            penalty = (
                penalty + F.relu(spacing - self.constants.spacing_max_hz).mean() * 0.01
            )
        if vtl_pred is not None:
            penalty = (
                penalty + F.relu(self.constants.vtl_min_cm - vtl_pred).mean() * 0.05
            )
            penalty = (
                penalty + F.relu(vtl_pred - self.constants.vtl_max_cm).mean() * 0.05
            )
        for key in ("multi_formant_vtl", "vtl_from_f1_f4", "vtl_from_avg_spacing"):
            vtl_candidate = preds.get(key)
            if vtl_candidate is not None:
                penalty = (
                    penalty
                    + F.relu(self.constants.vtl_min_cm - vtl_candidate).mean() * 0.02
                )
                penalty = (
                    penalty
                    + F.relu(vtl_candidate - self.constants.vtl_max_cm).mean() * 0.02
                )

        derived_spacing = preds.get("derived_formant_spacing")
        if derived_spacing is not None:
            penalty = (
                penalty
                + F.relu(self.constants.spacing_min_hz - derived_spacing).mean() * 0.01
            )
            penalty = (
                penalty
                + F.relu(derived_spacing - self.constants.spacing_max_hz).mean() * 0.01
            )

        spacing_std = preds.get("spacing_std")
        if spacing_std is not None:
            penalty = penalty + F.relu(spacing_std - 140.0).mean() * 0.002

        formant_stability = preds.get("formant_stability_score")
        if formant_stability is not None:
            penalty = penalty + F.relu(0.55 - formant_stability).mean() * 0.05

        if self.toggles.use_acoustic_physics_consistency:
            aux_spacing = preds.get("aux_spacing_pred")
            obs_spacing = preds.get("imputed_formant_spacing")
            aux_vtl = preds.get("aux_vtl_pred")
            obs_vtl = preds.get("imputed_vtl")
            if aux_spacing is not None and obs_spacing is not None:
                valid = torch.isfinite(obs_spacing)
                if valid.sum() > 0:
                    penalty = penalty + self.acoustic_physics_weight * F.smooth_l1_loss(
                        aux_spacing[valid], obs_spacing[valid]
                    )
            if aux_vtl is not None and obs_vtl is not None:
                valid = torch.isfinite(obs_vtl)
                if valid.sum() > 0:
                    penalty = penalty + self.acoustic_physics_weight * F.smooth_l1_loss(
                        aux_vtl[valid], obs_vtl[valid]
                    )
        return penalty

    def _diversity_loss(
        self, preds: Mapping[str, torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        if not self.toggles.use_diversity_loss:
            return torch.zeros((), device=device)
        acoustic = preds.get("acoustic_embedding")
        physics_proj = preds.get("physics_diversity_projection")
        if acoustic is None or physics_proj is None:
            return torch.zeros((), device=device)
        cosine = F.cosine_similarity(acoustic, physics_proj, dim=-1, eps=1e-8)
        return cosine.abs().mean() * self.diversity_scale

    def _speaker_consistency_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if not self.toggles.use_speaker_consistency:
            return torch.zeros((), device=device)

        speaker_idx = targets.get("speaker_idx")
        features = preds.get("height_features")
        pred_height = preds.get("height_mu")
        if speaker_idx is None or features is None or pred_height is None:
            return torch.zeros((), device=device)

        speaker_idx = speaker_idx.to(device=device, dtype=torch.long)
        valid = (speaker_idx >= 0) & torch.isfinite(pred_height)
        if valid.sum() < 2:
            return torch.zeros((), device=device)

        speaker_idx = speaker_idx[valid]
        features = features[valid]
        pred_height = pred_height[valid]
        pair_mask = speaker_idx.unsqueeze(0).eq(speaker_idx.unsqueeze(1))
        pair_mask = torch.triu(pair_mask, diagonal=1)
        if pair_mask.sum() == 0:
            return torch.zeros((), device=device)

        height_stability = (
            (pred_height.unsqueeze(0) - pred_height.unsqueeze(1))
            .abs()[pair_mask]
            .mean()
        )

        features = F.normalize(features, dim=-1, eps=1e-6)
        logits = (
            torch.matmul(features, features.transpose(0, 1))
            / self.speaker_contrastive_temperature
        )
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        self_mask = torch.eye(logits.size(0), device=device, dtype=torch.bool)
        positive_mask = (
            speaker_idx.unsqueeze(0).eq(speaker_idx.unsqueeze(1)) & ~self_mask
        )
        valid_anchor = positive_mask.any(dim=1)
        if valid_anchor.any():
            exp_logits = torch.exp(logits) * (~self_mask)
            log_prob = logits - torch.log(
                exp_logits.sum(dim=1, keepdim=True).clamp(min=self.nll_floor)
            )
            contrastive = -(positive_mask * log_prob).sum(dim=1) / positive_mask.sum(
                dim=1
            ).clamp(min=1.0)
            contrastive = contrastive[valid_anchor].mean()
        else:
            contrastive = torch.zeros((), device=device)

        return height_stability + 0.25 * contrastive

    def _uncertainty_calibration_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if not self.toggles.use_uncertainty_calibration:
            return torch.zeros((), device=device)

        target = targets.get("height")
        mu = preds.get("height_mu")
        logvar = preds.get("height_logvar")
        if target is None or mu is None or logvar is None:
            return torch.zeros((), device=device)

        target = target.to(device)
        valid = torch.isfinite(target)
        if valid.sum() == 0:
            return torch.zeros((), device=device)

        sigma = torch.exp(
            0.5 * logvar[valid].clamp(min=math.log(self.nll_floor), max=6.0)
        )
        residual = (
            (target[valid] - mu[valid])
            .abs()
            .detach()
            .clamp(min=math.sqrt(self.nll_floor))
        )
        return F.smooth_l1_loss(sigma, residual)

    def _speaker_alignment_scale(self, epoch_value: int) -> float:
        start = int(self.speaker_alignment.warmup_start_epoch)
        end = int(self.speaker_alignment.warmup_end_epoch)
        if epoch_value < start:
            return 0.0
        if end <= start:
            return 1.0
        if epoch_value >= end:
            return 1.0
        return float(epoch_value - start + 1) / float(end - start + 1)

    def _clip_reliability(self, preds: Mapping[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        reliability = preds.get("clip_reliability_prior")
        if reliability is None:
            reliability = preds.get("quality_score")
        if reliability is None:
            height_mu = preds["height_mu"]
            reliability = torch.ones_like(height_mu, device=device)
        return reliability.to(device=device, dtype=torch.float32).clamp(
            min=float(self.reliability_config.min_weight), max=1.0
        )

    def _speaker_pooled_height_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if not self.speaker_alignment.enable_pooled_height:
            return torch.zeros((), device=device)

        speaker_idx = targets.get("speaker_idx")
        height_raw = targets.get("height_raw")
        if speaker_idx is None or height_raw is None:
            return torch.zeros((), device=device)

        speaker_idx = speaker_idx.to(device=device, dtype=torch.long)
        height_raw = height_raw.to(device=device, dtype=torch.float32)
        pred_height_cm = _denorm_tensor(preds["height_mu"].to(device), "height", self._get_target_stats(preds))
        pred_height_var = preds["height_var"].to(device=device, dtype=torch.float32)
        height_std_scale = (
            float(self._get_target_stats(preds).get("height", {}).get("std", 1.0))
            if self._get_target_stats(preds) is not None
            else 1.0
        )
        pred_height_var_cm = pred_height_var * (height_std_scale ** 2)
        clip_reliability = self._clip_reliability(preds, device)

        losses = []
        unique_speakers = torch.unique(speaker_idx[speaker_idx >= 0])
        for speaker in unique_speakers:
            mask = speaker_idx == speaker
            if int(mask.sum().item()) < 2:
                continue
            pooled = omega_reliability_pool(
                pred_height_cm[mask],
                clip_reliability=clip_reliability[mask],
                pred_var=pred_height_var_cm[mask],
                config=self.aggregation_config,
            )
            speaker_targets = height_raw[mask]
            speaker_targets = speaker_targets[torch.isfinite(speaker_targets)]
            if speaker_targets.numel() == 0:
                continue
            losses.append(
                F.smooth_l1_loss(
                    pooled["mean"].view(()),
                    speaker_targets.mean().view(()),
                )
            )
        if not losses:
            return torch.zeros((), device=device)
        return torch.stack(losses).mean()

    def _speaker_clip_consistency_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if not self.speaker_alignment.enable_consistency:
            return torch.zeros((), device=device)

        speaker_idx = targets.get("speaker_idx")
        if speaker_idx is None:
            return torch.zeros((), device=device)

        speaker_idx = speaker_idx.to(device=device, dtype=torch.long)
        pred_height_cm = _denorm_tensor(preds["height_mu"].to(device), "height", self._get_target_stats(preds))
        pred_height_std_cm = torch.sqrt(preds["height_var"].to(device=device, dtype=torch.float32).clamp(min=self.nll_floor))
        height_std_scale = (
            float(self._get_target_stats(preds).get("height", {}).get("std", 1.0))
            if self._get_target_stats(preds) is not None
            else 1.0
        )
        pred_height_std_cm = pred_height_std_cm * height_std_scale
        clip_reliability = self._clip_reliability(preds, device)

        pair_losses = []
        max_combined = float(self.speaker_alignment.consistency_max_combined_std_cm)
        for speaker in torch.unique(speaker_idx[speaker_idx >= 0]):
            idxs = torch.nonzero(speaker_idx == speaker, as_tuple=False).flatten()
            if idxs.numel() < 2:
                continue
            for i in range(idxs.numel()):
                for j in range(i + 1, idxs.numel()):
                    ii = idxs[i]
                    jj = idxs[j]
                    combined_std = pred_height_std_cm[ii] + pred_height_std_cm[jj]
                    if float(combined_std.item()) > max_combined:
                        continue
                    weight = (
                        0.5 * (clip_reliability[ii] + clip_reliability[jj])
                    ) / combined_std.clamp(min=1.0)
                    pair_losses.append(weight * (pred_height_cm[ii] - pred_height_cm[jj]).abs())
        if not pair_losses:
            return torch.zeros((), device=device)
        return torch.stack(pair_losses).mean()

    def _speaker_height_ranking_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if not self.speaker_alignment.enable_ranking:
            return torch.zeros((), device=device)

        speaker_idx = targets.get("speaker_idx")
        height_raw = targets.get("height_raw")
        if speaker_idx is None or height_raw is None:
            return torch.zeros((), device=device)

        speaker_idx = speaker_idx.to(device=device, dtype=torch.long)
        height_raw = height_raw.to(device=device, dtype=torch.float32)
        pred_height_cm = _denorm_tensor(preds["height_mu"].to(device), "height", self._get_target_stats(preds))
        pred_height_var = preds["height_var"].to(device=device, dtype=torch.float32)
        height_std_scale = (
            float(self._get_target_stats(preds).get("height", {}).get("std", 1.0))
            if self._get_target_stats(preds) is not None
            else 1.0
        )
        pred_height_var_cm = pred_height_var * (height_std_scale ** 2)
        clip_reliability = self._clip_reliability(preds, device)

        pooled_preds = []
        pooled_truth = []
        unique_speakers = torch.unique(speaker_idx[speaker_idx >= 0])
        for speaker in unique_speakers:
            mask = speaker_idx == speaker
            pooled = omega_reliability_pool(
                pred_height_cm[mask],
                clip_reliability=clip_reliability[mask],
                pred_var=pred_height_var_cm[mask],
                config=self.aggregation_config,
            )
            truths = height_raw[mask]
            truths = truths[torch.isfinite(truths)]
            if truths.numel() == 0:
                continue
            pooled_preds.append(pooled["mean"])
            pooled_truth.append(truths.mean())

        if len(pooled_preds) < 2:
            return torch.zeros((), device=device)

        pred_vec = torch.stack(pooled_preds)
        truth_vec = torch.stack(pooled_truth)
        pair_losses = []
        min_delta = float(self.speaker_alignment.ranking_min_height_delta_cm)
        margin = float(self.speaker_alignment.ranking_margin_cm)
        for i in range(pred_vec.numel()):
            for j in range(i + 1, pred_vec.numel()):
                true_delta = truth_vec[i] - truth_vec[j]
                if float(true_delta.abs().item()) < min_delta:
                    continue
                sign = true_delta.sign()
                pair_losses.append(F.relu(margin - sign * (pred_vec[i] - pred_vec[j])))
        if not pair_losses:
            return torch.zeros((), device=device)
        return torch.stack(pair_losses).mean()

    def _ensure_finite_scalar(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Loss component '{name}' must be a tensor, got {type(value).__name__}"
            )
        if value.ndim != 0:
            raise ValueError(
                f"Loss component '{name}' must be scalar, got shape {tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value)):
            raise FloatingPointError(
                f"Loss component '{name}' became non-finite: {value.detach().cpu().item()}"
            )
        return value

    def forward(
        self, preds: Dict[str, Any], targets: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        device = preds["height_mu"].device
        epoch_value = int(
            targets.get("epoch", preds.get("current_epoch", self.current_epoch))
        )
        speaker_alignment_scale = self._speaker_alignment_scale(epoch_value)
        height_loss, height_mae = self._height_loss(preds, targets, device)
        weight_loss, _ = self._probabilistic_regression_loss(
            preds, targets, "weight", device, mask_key="weight_mask"
        )
        age_loss, _ = self._probabilistic_regression_loss(preds, targets, "age", device)
        shoulder_loss, _ = self._probabilistic_regression_loss(
            preds, targets, "shoulder", device, mask_key="shoulder_mask"
        )
        waist_loss, _ = self._probabilistic_regression_loss(
            preds, targets, "waist", device, mask_key="waist_mask"
        )
        gender_loss = self._gender_loss(preds, targets, device)
        domain_loss = self._domain_loss(preds, targets, device)
        vtsl_loss = self._vtsl_loss(preds, targets, device)
        physics_penalty = self._physics_penalty(preds, device)
        diversity_loss = self._diversity_loss(preds, device)
        ranking_loss = self._pairwise_height_ranking_loss(preds, targets)
        speaker_consistency_loss = self._speaker_consistency_loss(
            preds, targets, device
        )
        uncertainty_calibration_loss = self._uncertainty_calibration_loss(
            preds, targets, device
        )
        speaker_pooled_height_loss = (
            self._speaker_pooled_height_loss(preds, targets, device)
            * speaker_alignment_scale
        )
        speaker_clip_consistency_loss = (
            self._speaker_clip_consistency_loss(preds, targets, device)
            * speaker_alignment_scale
        )
        speaker_height_ranking_loss = (
            self._speaker_height_ranking_loss(preds, targets, device)
            * speaker_alignment_scale
        )

        if not self.toggles.use_shoulder_head:
            shoulder_loss = torch.zeros((), device=device)
        if not self.toggles.use_waist_head:
            waist_loss = torch.zeros((), device=device)

        losses = {
            "height_nll": height_loss,
            "height_mae_proxy": height_mae,
            "weight": weight_loss,
            "age": age_loss,
            "shoulder": shoulder_loss,
            "waist": waist_loss,
            "gender": gender_loss,
            "vtsl": vtsl_loss,
            "physics_penalty": physics_penalty,
            "diversity": diversity_loss,
            "height_ranking": ranking_loss,
            "speaker_consistency": speaker_consistency_loss,
            "uncertainty_calibration": uncertainty_calibration_loss,
            "domain_adv": domain_loss,
            "speaker_pooled_height": speaker_pooled_height_loss,
            "speaker_clip_consistency": speaker_clip_consistency_loss,
            "speaker_height_ranking": speaker_height_ranking_loss,
            "speaker_alignment_scale": torch.tensor(
                float(speaker_alignment_scale), device=device
            ),
        }
        losses = {
            name: self._ensure_finite_scalar(name, value)
            for name, value in losses.items()
        }

        total = (
            self.loss_weights.height * losses["height_nll"]
            + self.loss_weights.weight * losses["weight"]
            + self.loss_weights.age * losses["age"]
            + self.loss_weights.shoulder * losses["shoulder"]
            + self.loss_weights.waist * losses["waist"]
            + self.loss_weights.gender * losses["gender"]
            + self.loss_weights.vtsl * losses["vtsl"]
            + self.loss_weights.physics_penalty * losses["physics_penalty"]
            + self.loss_weights.domain_adv * losses["domain_adv"]
            + self.loss_weights.ranking * losses["height_ranking"]
            + self.loss_weights.diversity * losses["diversity"]
            + self.loss_weights.speaker_consistency * losses["speaker_consistency"]
            + self.loss_weights.uncertainty_calibration
            * losses["uncertainty_calibration"]
            + float(self.speaker_alignment.pooled_height_weight_max)
            * losses["speaker_pooled_height"]
            + float(self.speaker_alignment.consistency_weight_max)
            * losses["speaker_clip_consistency"]
            + float(self.speaker_alignment.ranking_weight_max)
            * losses["speaker_height_ranking"]
        )
        losses["total"] = self._ensure_finite_scalar("total", total)
        return losses


__all__ = ["KendallMultiTaskLoss", "VocalTractSimulatorLossV2"]


class KendallMultiTaskLoss(nn.Module):
    """Uncertainty-weighted multi-task loss (Kendall et al., 2018).

    Learns per-task log-variance parameters to automatically balance
    task contributions. Each task loss L_i is weighted as:
        L_total = Σ_i exp(-s_i) * L_i + s_i
    where s_i = log(σ_i²) is learned during training.

    This replaces manual loss weight tuning.
    """

    # Tasks that get automatic weighting
    KENDALL_TASKS = (
        "height_nll",
        "weight",
        "age",
        "shoulder",
        "waist",
        "gender",
        "vtsl",
        "physics_penalty",
        "domain_adv",
        "height_ranking",
        "diversity",
        "speaker_consistency",
        "uncertainty_calibration",
    )

    def __init__(self, base_loss: VocalTractSimulatorLossV2):
        super().__init__()
        self.base_loss = base_loss

        # Initialize log-variance parameters: start at 0 (σ=1, equal weight)
        self.log_vars = nn.ParameterDict(
            {task: nn.Parameter(torch.zeros(1)) for task in self.KENDALL_TASKS}
        )

    def set_epoch(self, epoch: int) -> None:
        self.base_loss.set_epoch(epoch)

    def set_target_stats(
        self, target_stats: Optional[Mapping[str, Mapping[str, float]]]
    ) -> None:
        self.base_loss.set_target_stats(target_stats)

    def forward(
        self, preds: Dict[str, Any], targets: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        # Get individual losses from base loss module
        losses = self.base_loss(preds, targets)

        # Apply Kendall weighting
        total = torch.zeros((), device=preds["height_mu"].device)
        kendall_terms = {}

        for task in self.KENDALL_TASKS:
            if task not in losses:
                continue
            loss_val = losses[task]
            if loss_val.numel() != 1:
                continue

            log_var = self.log_vars[task].squeeze()
            # L_weighted = exp(-s) * L + s
            precision = torch.exp(-log_var)
            weighted = precision * loss_val + log_var
            total = total + weighted
            kendall_terms[f"kendall_weight_{task}"] = precision.detach()

        losses["total"] = total.squeeze()
        losses.update(kendall_terms)

        # Log learned weights for monitoring
        with torch.no_grad():
            for task in self.KENDALL_TASKS:
                if task in self.log_vars:
                    s = self.log_vars[task].item()
                    losses[f"kendall_logvar_{task}"] = torch.tensor(s)

        return losses
