"""Physics-aware feature extraction and branch modules."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .config import DEFAULT_PHYSICS_CONSTANTS, PhysicsConstants, PhysicsFeatureSpec
from .utils import _masked_feature_mean, _plausible_spacing


class PhysicsMLP(nn.Module):
    """Standard MLP for physics path — regular Linear + GELU + Dropout."""

    def __init__(
        self,
        in_dim: int = 19,
        hidden_dims: Sequence[int] = (512, 512, 256, 128),
        dropout: float = 0.10,
    ):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsPath(nn.Module):
    """Physics branch with explicit observed/derived/imputed bookkeeping."""

    def __init__(
        self,
        spec: PhysicsFeatureSpec,
        constants: PhysicsConstants = DEFAULT_PHYSICS_CONSTANTS,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.spec = spec
        self.constants = constants
        self.input_dim = 19
        self.reliability_dim = 8
        self.net = PhysicsMLP(
            in_dim=self.input_dim, hidden_dims=(512, 512, 256, 128), dropout=dropout
        )
        self.spacing_head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 1),
        )

    def _feature_or_zeros(self, features: torch.Tensor, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= features.size(-1):
            return torch.zeros(
                features.size(0),
                features.size(1),
                1,
                device=features.device,
                dtype=features.dtype,
            )
        return features[:, :, idx : idx + 1]

    def _valid_signal_mask(self, feature: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(feature.squeeze(-1)) & (
            feature.abs().squeeze(-1) > self.constants.signal_floor
        )

    def _normalize_physics_input(
        self,
        *,
        imputed_vtl: torch.Tensor,
        vtl_from_multi_formants: torch.Tensor,
        vtl_from_f1_f4: torch.Tensor,
        vtl_from_avg_spacing: torch.Tensor,
        spacing_std: torch.Tensor,
        f1m: torch.Tensor,
        f2m: torch.Tensor,
        f3m: torch.Tensor,
        f4m: torch.Tensor,
        formant_ratio_f2_f1: torch.Tensor,
        formant_ratio_f3_f2: torch.Tensor,
        formant_ratio_f4_f3: torch.Tensor,
        f0m: torch.Tensor,
        spacing_confidence: torch.Tensor,
        vtl_confidence: torch.Tensor,
        formant_stability_score: torch.Tensor,
        observed_spacing_rel: torch.Tensor,
        observed_vtl_rel: torch.Tensor,
        imputed_spacing: torch.Tensor,
        f0_rel: torch.Tensor,
    ) -> torch.Tensor:
        c = self.constants
        spacing = imputed_spacing.unsqueeze(-1).clamp(min=max(1.0, c.ratio_floor))
        relative_formants = (
            torch.cat([f1m, f2m, f3m, f4m], dim=-1).clamp(min=c.ratio_floor) / spacing
        )
        relative_formants = torch.log(relative_formants.clamp(min=c.ratio_floor))

        vtl_norms = torch.stack(
            [
                (imputed_vtl - c.default_vtl_cm) / max(c.vtl_range_cm, c.eps),
                (vtl_from_multi_formants - c.default_vtl_cm)
                / max(c.vtl_range_cm, c.eps),
                (vtl_from_f1_f4 - c.default_vtl_cm) / max(c.vtl_range_cm, c.eps),
                (vtl_from_avg_spacing - c.default_vtl_cm) / max(c.vtl_range_cm, c.eps),
            ],
            dim=-1,
        )
        spacing_std_norm = (spacing_std / max(c.spacing_range_hz, c.eps)).unsqueeze(-1)
        ratio_feats = torch.stack(
            [
                torch.log(formant_ratio_f2_f1.clamp(min=c.ratio_floor)),
                torch.log(formant_ratio_f3_f2.clamp(min=c.ratio_floor)),
                torch.log(formant_ratio_f4_f3.clamp(min=c.ratio_floor)),
            ],
            dim=-1,
        ).clamp(min=-2.5, max=2.5)
        f0_norm = torch.log((f0m.clamp(min=50.0)) / 140.0)
        reliability_feats = torch.stack(
            [
                spacing_confidence,
                vtl_confidence,
                formant_stability_score,
                observed_spacing_rel,
                observed_vtl_rel,
                f0_rel,
            ],
            dim=-1,
        )
        reliability_feats = reliability_feats.mul(2.0).sub(1.0)
        return torch.cat(
            [
                vtl_norms,
                spacing_std_norm,
                relative_formants,
                ratio_feats,
                f0_norm,
                reliability_feats,
            ],
            dim=-1,
        )

    def _extract_physics_features(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        s = self.spec
        c = self.constants

        f1 = self._feature_or_zeros(features, s.formant_freq_idx(0))
        f2 = self._feature_or_zeros(features, s.formant_freq_idx(1))
        f3 = self._feature_or_zeros(features, s.formant_freq_idx(2))
        f4 = self._feature_or_zeros(features, s.formant_freq_idx(3))
        f0 = self._feature_or_zeros(features, s.f0_idx)
        observed_spacing_feat = self._feature_or_zeros(features, s.spacing_idx)
        observed_vtl_feat = self._feature_or_zeros(features, s.vtl_idx)

        f1m, _, f1_rel = _masked_feature_mean(
            f1, padding_mask, validity_mask=self._valid_signal_mask(f1)
        )
        f2m, _, f2_rel = _masked_feature_mean(
            f2, padding_mask, validity_mask=self._valid_signal_mask(f2)
        )
        f3m, _, f3_rel = _masked_feature_mean(
            f3, padding_mask, validity_mask=self._valid_signal_mask(f3)
        )
        f4m, _, f4_rel = _masked_feature_mean(
            f4, padding_mask, validity_mask=self._valid_signal_mask(f4)
        )
        f0m, _, f0_rel = _masked_feature_mean(
            f0, padding_mask, validity_mask=self._valid_signal_mask(f0)
        )

        observed_spacing, observed_spacing_available, observed_spacing_rel = (
            _masked_feature_mean(
                observed_spacing_feat,
                padding_mask,
                validity_mask=self._valid_signal_mask(observed_spacing_feat),
                default=c.default_spacing_hz,
            )
        )
        observed_vtl, observed_vtl_available, observed_vtl_rel = _masked_feature_mean(
            observed_vtl_feat,
            padding_mask,
            validity_mask=self._valid_signal_mask(observed_vtl_feat),
            default=c.default_vtl_cm,
        )

        pair_rel = torch.stack(
            [
                torch.minimum(f1_rel, f2_rel),
                torch.minimum(f2_rel, f3_rel),
                torch.minimum(f3_rel, f4_rel),
            ],
            dim=-1,
        )
        pair_diffs = torch.stack(
            [
                (f2m - f1m).squeeze(-1),
                (f3m - f2m).squeeze(-1),
                (f4m - f3m).squeeze(-1),
            ],
            dim=-1,
        ).abs()
        pair_weight_sum = pair_rel.sum(dim=-1)
        avg_spacing_default = torch.full_like(pair_weight_sum, c.default_spacing_hz)
        avg_spacing = torch.where(
            pair_weight_sum > 0,
            pair_diffs.mean(dim=-1),
            avg_spacing_default,
        )
        derived_spacing = (pair_diffs * pair_rel).sum(dim=-1) / pair_weight_sum.clamp(
            min=c.eps
        )
        derived_spacing = torch.where(
            pair_weight_sum > 0,
            derived_spacing,
            avg_spacing_default,
        )
        derived_spacing = derived_spacing.abs().clamp(
            min=c.spacing_min_hz, max=c.spacing_max_hz
        )
        avg_spacing = avg_spacing.abs().clamp(
            min=c.spacing_min_hz, max=c.spacing_max_hz
        )
        derived_spacing_rel = pair_rel.mean(dim=-1).clamp(min=0.0, max=1.0)
        spacing_var = (
            (pair_diffs - derived_spacing.unsqueeze(-1)).pow(2) * pair_rel
        ).sum(dim=-1) / pair_weight_sum.clamp(min=c.eps)
        spacing_std = torch.where(
            pair_weight_sum > 0,
            torch.sqrt(spacing_var.clamp(min=0.0)),
            torch.zeros_like(derived_spacing),
        )

        observed_spacing_value = (
            observed_spacing.squeeze(-1)
            .abs()
            .clamp(min=c.spacing_min_hz, max=c.spacing_max_hz)
        )
        formant_ratio_f2_f1 = (
            f2m.clamp(min=c.ratio_floor) / f1m.clamp(min=c.ratio_floor)
        ).squeeze(-1)
        formant_ratio_f3_f2 = (
            f3m.clamp(min=c.ratio_floor) / f2m.clamp(min=c.ratio_floor)
        ).squeeze(-1)
        formant_ratio_f4_f3 = (
            f4m.clamp(min=c.ratio_floor) / f3m.clamp(min=c.ratio_floor)
        ).squeeze(-1)

        f1f4_spacing = ((f4m - f1m).squeeze(-1).abs() / 3.0).clamp(
            min=c.spacing_min_hz, max=c.spacing_max_hz
        )
        f1f4_rel = torch.minimum(f1_rel, f4_rel)
        vtl_from_avg_spacing = (
            c.speed_of_sound_cm_per_s / (2.0 * avg_spacing.clamp(min=1.0))
        ).clamp(min=c.vtl_min_cm, max=c.vtl_max_cm)
        vtl_from_weighted_spacing = (
            c.speed_of_sound_cm_per_s / (2.0 * derived_spacing.clamp(min=1.0))
        ).clamp(
            min=c.vtl_min_cm,
            max=c.vtl_max_cm,
        )
        vtl_from_f1_f4 = (
            c.speed_of_sound_cm_per_s / (2.0 * f1f4_spacing.clamp(min=1.0))
        ).clamp(
            min=c.vtl_min_cm,
            max=c.vtl_max_cm,
        )
        multi_vtl_weights = torch.stack(
            [
                derived_spacing_rel,
                derived_spacing_rel,
                f1f4_rel,
            ],
            dim=-1,
        )
        multi_vtl_candidates = torch.stack(
            [
                vtl_from_avg_spacing,
                vtl_from_weighted_spacing,
                vtl_from_f1_f4,
            ],
            dim=-1,
        )
        vtl_from_multi_formants = (multi_vtl_candidates * multi_vtl_weights).sum(
            dim=-1
        ) / multi_vtl_weights.sum(dim=-1).clamp(min=c.eps)
        vtl_from_multi_formants = torch.where(
            multi_vtl_weights.sum(dim=-1) > 0,
            vtl_from_multi_formants,
            torch.full_like(vtl_from_avg_spacing, c.default_vtl_cm),
        ).clamp(min=c.vtl_min_cm, max=c.vtl_max_cm)

        spacing_variation_score = 1.0 - (
            spacing_std / derived_spacing.clamp(min=1.0)
        ).clamp(min=0.0, max=1.0)
        vtl_agreement = 1.0 - (
            (
                (vtl_from_avg_spacing - vtl_from_weighted_spacing).abs()
                + (vtl_from_f1_f4 - vtl_from_weighted_spacing).abs()
            )
            / 2.0
            / c.vtl_range_cm
        ).clamp(min=0.0, max=1.0)
        formant_stability_score = (
            torch.stack(
                [derived_spacing_rel, f1f4_rel, spacing_variation_score, vtl_agreement],
                dim=-1,
            )
            .mean(dim=-1)
            .clamp(min=0.0, max=1.0)
        )
        spacing_confidence = (
            torch.stack(
                [
                    derived_spacing_rel,
                    spacing_variation_score,
                    pair_rel.max(dim=-1).values.clamp(min=0.0, max=1.0),
                ],
                dim=-1,
            )
            .mean(dim=-1)
            .clamp(min=0.0, max=1.0)
        )
        vtl_confidence = (
            torch.stack(
                [
                    spacing_confidence,
                    vtl_agreement,
                    torch.maximum(observed_vtl_rel, f1f4_rel),
                ],
                dim=-1,
            )
            .mean(dim=-1)
            .clamp(min=0.0, max=1.0)
        )

        imputed_spacing = torch.where(
            observed_spacing_available, observed_spacing_value, derived_spacing
        )
        spacing_reliability = torch.where(
            observed_spacing_available, observed_spacing_rel, spacing_confidence
        )
        spacing_source = torch.where(
            observed_spacing_available,
            torch.zeros_like(spacing_reliability, dtype=torch.long),
            torch.ones_like(spacing_reliability, dtype=torch.long),
        )

        derived_vtl = vtl_from_multi_formants
        derived_vtl_rel = vtl_confidence

        observed_vtl_value = (
            observed_vtl.squeeze(-1).abs().clamp(min=c.vtl_min_cm, max=c.vtl_max_cm)
        )
        imputed_vtl = torch.where(
            observed_vtl_available, observed_vtl_value, derived_vtl
        )
        vtl_reliability = torch.where(
            observed_vtl_available, observed_vtl_rel, vtl_confidence
        )
        vtl_source = torch.where(
            observed_vtl_available,
            torch.zeros_like(vtl_reliability, dtype=torch.long),
            torch.ones_like(vtl_reliability, dtype=torch.long),
        )

        raw_physics_input = torch.cat(
            [
                imputed_vtl.unsqueeze(-1),
                vtl_from_multi_formants.unsqueeze(-1),
                vtl_from_f1_f4.unsqueeze(-1),
                vtl_from_avg_spacing.unsqueeze(-1),
                spacing_std.unsqueeze(-1),
                f1m,
                f2m,
                f3m,
                f4m,
                formant_ratio_f2_f1.unsqueeze(-1),
                formant_ratio_f3_f2.unsqueeze(-1),
                formant_ratio_f4_f3.unsqueeze(-1),
                f0m,
                spacing_confidence.unsqueeze(-1),
                vtl_confidence.unsqueeze(-1),
                formant_stability_score.unsqueeze(-1),
                observed_spacing_rel.unsqueeze(-1),
                observed_vtl_rel.unsqueeze(-1),
                f0_rel.unsqueeze(-1),
            ],
            dim=-1,
        )
        normalized_physics_input = self._normalize_physics_input(
            imputed_vtl=imputed_vtl,
            vtl_from_multi_formants=vtl_from_multi_formants,
            vtl_from_f1_f4=vtl_from_f1_f4,
            vtl_from_avg_spacing=vtl_from_avg_spacing,
            spacing_std=spacing_std,
            f1m=f1m,
            f2m=f2m,
            f3m=f3m,
            f4m=f4m,
            formant_ratio_f2_f1=formant_ratio_f2_f1,
            formant_ratio_f3_f2=formant_ratio_f3_f2,
            formant_ratio_f4_f3=formant_ratio_f4_f3,
            f0m=f0m,
            spacing_confidence=spacing_confidence,
            vtl_confidence=vtl_confidence,
            formant_stability_score=formant_stability_score,
            observed_spacing_rel=observed_spacing_rel,
            observed_vtl_rel=observed_vtl_rel,
            imputed_spacing=imputed_spacing,
            f0_rel=f0_rel,
        )

        spacing_consistency = torch.where(
            observed_spacing_available,
            1.0
            - (
                (observed_spacing_value - derived_spacing).abs() / c.spacing_range_hz
            ).clamp(min=0.0, max=1.0),
            spacing_confidence,
        )
        vtl_consistency = torch.where(
            observed_vtl_available,
            1.0
            - ((observed_vtl_value - derived_vtl).abs() / c.vtl_range_cm).clamp(
                min=0.0, max=1.0
            ),
            vtl_confidence,
        )
        physics_confidence = (
            torch.stack(
                [
                    spacing_confidence,
                    vtl_confidence,
                    formant_stability_score,
                    f0_rel,
                    spacing_consistency,
                    vtl_consistency,
                ],
                dim=-1,
            )
            .mean(dim=-1)
            .clamp(min=0.0, max=1.0)
        )
        physics_residual_confidence = (
            torch.stack(
                [formant_stability_score, spacing_confidence, vtl_confidence],
                dim=-1,
            )
            .mean(dim=-1)
            .clamp(min=0.0, max=1.0)
        )
        return {
            "physics_input": normalized_physics_input,
            "physics_input_raw": raw_physics_input,
            "physics_input_normalized": normalized_physics_input,
            "formant_spacing_observed": observed_spacing_value,
            "formant_spacing_derived": derived_spacing,
            "formant_spacing_imputed": imputed_spacing,
            "formant_spacing_reliability": spacing_reliability,
            "formant_spacing_observed_reliability": observed_spacing_rel,
            "formant_spacing_derived_reliability": spacing_confidence,
            "formant_spacing_source": spacing_source,
            "formant_spacing_used_observed": observed_spacing_available,
            "vtl_observed": observed_vtl_value,
            "vtl_derived": derived_vtl,
            "vtl_imputed": imputed_vtl,
            "vtl_reliability": vtl_reliability,
            "vtl_observed_reliability": observed_vtl_rel,
            "vtl_derived_reliability": vtl_confidence,
            "vtl_source": vtl_source,
            "vtl_used_observed": observed_vtl_available,
            "formant_spacing_consistency": spacing_consistency,
            "vtl_consistency": vtl_consistency,
            "physics_confidence": physics_confidence,
            "vtl_from_multi_formants": vtl_from_multi_formants,
            "multi_formant_vtl": vtl_from_multi_formants,
            "vtl_from_f1_f4": vtl_from_f1_f4,
            "vtl_from_avg_spacing": vtl_from_avg_spacing,
            "spacing_std": spacing_std,
            "formant_ratio_f2_f1": formant_ratio_f2_f1,
            "formant_ratio_f3_f2": formant_ratio_f3_f2,
            "formant_ratio_f4_f3": formant_ratio_f4_f3,
            "spacing_confidence": spacing_confidence,
            "vtl_confidence": vtl_confidence,
            "formant_stability_score": formant_stability_score,
            "physics_residual_confidence": physics_residual_confidence,
            "physics_reliability": torch.stack(
                [
                    spacing_confidence,
                    vtl_confidence,
                    formant_stability_score,
                    spacing_reliability,
                    vtl_reliability,
                    observed_spacing_rel,
                    observed_vtl_rel,
                    f0_rel,
                ],
                dim=-1,
            ),
        }

    def forward(
        self, features: torch.Tensor, padding_mask: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        stats = self._extract_physics_features(features, padding_mask)
        physics_emb = self.net(stats["physics_input"])
        spacing_pred = _plausible_spacing(
            self.spacing_head(physics_emb).squeeze(-1), self.constants
        )
        return {
            "physics_input": stats["physics_input"],
            "physics_embedding": physics_emb,
            "formant_spacing_pred": spacing_pred,
            "physics_input_raw": stats["physics_input_raw"],
            "physics_input_normalized": stats["physics_input_normalized"],
            "observed_formant_spacing": stats["formant_spacing_observed"],
            "derived_formant_spacing": stats["formant_spacing_derived"],
            "imputed_formant_spacing": stats["formant_spacing_imputed"],
            "formant_spacing_reliability": stats["formant_spacing_reliability"],
            "observed_formant_spacing_reliability": stats[
                "formant_spacing_observed_reliability"
            ],
            "derived_formant_spacing_reliability": stats[
                "formant_spacing_derived_reliability"
            ],
            "formant_spacing_source": stats["formant_spacing_source"],
            "formant_spacing_used_observed": stats["formant_spacing_used_observed"],
            "observed_vtl": stats["vtl_observed"],
            "derived_vtl": stats["vtl_derived"],
            "imputed_vtl": stats["vtl_imputed"],
            "vtl_reliability": stats["vtl_reliability"],
            "observed_vtl_reliability": stats["vtl_observed_reliability"],
            "derived_vtl_reliability": stats["vtl_derived_reliability"],
            "vtl_source": stats["vtl_source"],
            "vtl_used_observed": stats["vtl_used_observed"],
            "formant_spacing_consistency": stats["formant_spacing_consistency"],
            "vtl_consistency": stats["vtl_consistency"],
            "physics_confidence": stats["physics_confidence"],
            "vtl_from_multi_formants": stats["vtl_from_multi_formants"],
            "multi_formant_vtl": stats["multi_formant_vtl"],
            "vtl_from_f1_f4": stats["vtl_from_f1_f4"],
            "vtl_from_avg_spacing": stats["vtl_from_avg_spacing"],
            "spacing_std": stats["spacing_std"],
            "formant_ratio_f2_f1": stats["formant_ratio_f2_f1"],
            "formant_ratio_f3_f2": stats["formant_ratio_f3_f2"],
            "formant_ratio_f4_f3": stats["formant_ratio_f4_f3"],
            "spacing_confidence": stats["spacing_confidence"],
            "vtl_confidence": stats["vtl_confidence"],
            "formant_stability_score": stats["formant_stability_score"],
            "physics_residual_confidence": stats["physics_residual_confidence"],
            "physics_reliability": stats["physics_reliability"],
        }


__all__ = ["PhysicsMLP", "PhysicsPath"]
