"""Main VocalMorph V2 model orchestration."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import torch
import torch.nn as nn

from .acoustic import AcousticPathECAPAConformer
from .config import (
    AblationToggles,
    AggregationConfig,
    DEFAULT_LOSS_WEIGHTS,
    DEFAULT_PHYSICS_CONSTANTS,
    EPS,
    LossWeights,
    ModelHyperparameters,
    PhysicsConstants,
    PhysicsFeatureSpec,
    ReliabilityConfig,
    SpeakerAlignmentConfig,
    dataclass_from_mapping,
)
from .heads import (
    AcousticPhysicsConsistencyHead,
    BayesianHeightHead,
    HeightFeatureAdapter,
    HeightPriorHead,
    ProbabilisticRegressionHead,
    ReliabilityAdaptivePhysicsGate,
)
from .layers import (
    AttentiveStatsPooling,
    ConditionalLayerNorm,
    GradientReversalLayer,
    MCDropout,
    PositiveLinear,
    SinusoidalPositionalEncoding,
    SqueezeExcitationVector,
    CrossAttentionFusion,
)
from .losses import KendallMultiTaskLoss, VocalTractSimulatorLossV2
from .physics import PhysicsPath
from .reliability import MetadataReliabilityTower, compose_clip_reliability
from .types import RegressionHeadOutput, RegressionUncertaintySummary
from .utils import (
    _clone_tensor_mapping,
    _denorm_tensor,
    _validate_batch_axis,
    _validate_class_labels,
    _validate_sequence_inputs,
    _zero_regression_output,
    aggregate_by_speaker,
    build_multi_crops,
)


class VocalMorphV2(nn.Module):
    """Publication-grade dual-path anthropometric prediction model."""

    expects_domain = True
    regression_targets: Tuple[str, ...] = (
        "height",
        "weight",
        "age",
        "shoulder",
        "waist",
    )
    denorm_output_names: Dict[str, str] = {
        "height": "height_cm",
        "weight": "weight_kg",
        "age": "age_years",
        "shoulder": "shoulder_cm",
        "waist": "waist_cm",
    }

    def __init__(
        self,
        input_dim: int,
        feature_spec: Optional[PhysicsFeatureSpec] = None,
        ecapa_channels: int = 512,
        ecapa_scale: int = 8,
        conformer_d_model: int = 256,
        conformer_heads: int = 8,
        conformer_blocks: int = 4,
        dropout: float = 0.30,
        target_stats: Optional[Mapping[str, Mapping[str, float]]] = None,
        use_feature_mixup: bool = True,
        mixup_alpha: float = 0.2,
        focal_after_epoch: int = 20,
        ranking_margin: float = 0.10,
        constants: PhysicsConstants = DEFAULT_PHYSICS_CONSTANTS,
        toggles: Optional[AblationToggles] = None,
        loss_weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
        hyperparameters: Optional[ModelHyperparameters] = None,
        aggregation_config: Optional[AggregationConfig] = None,
        reliability_config: Optional[ReliabilityConfig] = None,
        speaker_alignment_config: Optional[SpeakerAlignmentConfig] = None,
    ):
        super().__init__()
        self.feature_spec = feature_spec or PhysicsFeatureSpec()
        min_required_input_dim = self.feature_spec.minimum_input_dim
        if input_dim < min_required_input_dim:
            raise ValueError(
                f"VocalMorphV2 expected input_dim >= {min_required_input_dim} based on the declared feature layout, got input_dim={input_dim}"
            )
        self.target_stats = target_stats
        self.constants = constants

        toggle_config = toggles or AblationToggles()
        if toggle_config.use_feature_mixup != bool(use_feature_mixup):
            toggle_dict = asdict(toggle_config)
            toggle_dict["use_feature_mixup"] = bool(use_feature_mixup)
            toggle_config = AblationToggles(**toggle_dict)
        self.toggles = toggle_config
        self.loss_weights = loss_weights
        self.aggregation_config = aggregation_config or AggregationConfig()
        self.reliability_config = reliability_config or ReliabilityConfig()
        self.speaker_alignment_config = (
            speaker_alignment_config or SpeakerAlignmentConfig()
        )

        self.hyperparameters = hyperparameters or ModelHyperparameters(
            ecapa_channels=int(ecapa_channels),
            ecapa_scale=int(ecapa_scale),
            conformer_d_model=int(conformer_d_model),
            conformer_heads=int(conformer_heads),
            conformer_blocks=int(conformer_blocks),
            dropout=float(dropout),
            mixup_alpha=float(mixup_alpha),
            focal_after_epoch=int(focal_after_epoch),
            ranking_margin=float(ranking_margin),
        )
        self.domain_classes = self.hyperparameters.domain_classes
        self.gender_classes = self.hyperparameters.gender_classes
        self.use_feature_mixup = (
            bool(use_feature_mixup) and self.toggles.use_feature_mixup
        )
        self.mixup_alpha = float(self.hyperparameters.mixup_alpha)
        self.focal_after_epoch = int(self.hyperparameters.focal_after_epoch)
        self.ranking_margin = float(self.hyperparameters.ranking_margin)

        if self.hyperparameters.conformer_blocks < 1:
            raise ValueError(
                f"conformer_blocks must be >= 1, got {self.hyperparameters.conformer_blocks}"
            )
        if self.hyperparameters.conformer_heads < 1:
            raise ValueError(
                f"conformer_heads must be >= 1, got {self.hyperparameters.conformer_heads}"
            )
        if (
            self.hyperparameters.conformer_d_model
            % self.hyperparameters.conformer_heads
            != 0
        ):
            raise ValueError(
                "conformer_d_model "
                f"({self.hyperparameters.conformer_d_model}) must be divisible by conformer_heads "
                f"({self.hyperparameters.conformer_heads})"
            )
        if not 0.0 <= float(self.hyperparameters.dropout) < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {self.hyperparameters.dropout}"
            )
        if not 0.0 <= float(self.hyperparameters.drop_path_rate) < 1.0:
            raise ValueError(
                f"drop_path_rate must be in [0, 1), got {self.hyperparameters.drop_path_rate}"
            )
        if self.use_feature_mixup and self.mixup_alpha <= 0.0:
            raise ValueError(
                f"mixup_alpha must be > 0 when feature mixup is enabled, got {self.mixup_alpha}"
            )
        if self.hyperparameters.pooling_hidden_dim < 1:
            raise ValueError(
                f"pooling_hidden_dim must be >= 1, got {self.hyperparameters.pooling_hidden_dim}"
            )
        if self.hyperparameters.feature_norm_eps <= 0.0:
            raise ValueError(
                f"feature_norm_eps must be > 0, got {self.hyperparameters.feature_norm_eps}"
            )
        if self.hyperparameters.layer_scale_init <= 0.0:
            raise ValueError(
                f"layer_scale_init must be > 0, got {self.hyperparameters.layer_scale_init}"
            )
        if self.hyperparameters.rel_pos_max_distance < 1:
            raise ValueError(
                f"rel_pos_max_distance must be >= 1, got {self.hyperparameters.rel_pos_max_distance}"
            )
        if self.hyperparameters.cross_rel_pos_max_distance < 1:
            raise ValueError(
                f"cross_rel_pos_max_distance must be >= 1, got {self.hyperparameters.cross_rel_pos_max_distance}"
            )
        if self.hyperparameters.height_adapter_hidden_dim < 1:
            raise ValueError(
                f"height_adapter_hidden_dim must be >= 1, got {self.hyperparameters.height_adapter_hidden_dim}"
            )
        if self.hyperparameters.height_adapter_scale < 0.0:
            raise ValueError(
                f"height_adapter_scale must be >= 0, got {self.hyperparameters.height_adapter_scale}"
            )

        self.physics_embedding_dim = self.hyperparameters.physics_embedding_dim
        self.physics_fusion_dim = self.hyperparameters.physics_fusion_dim
        self.fusion_dim = self.hyperparameters.fusion_dim

        self.acoustic_path = AcousticPathECAPAConformer(
            input_dim=input_dim,
            ecapa_channels=self.hyperparameters.ecapa_channels,
            ecapa_scale=self.hyperparameters.ecapa_scale,
            conformer_d_model=self.hyperparameters.conformer_d_model,
            conformer_heads=self.hyperparameters.conformer_heads,
            conformer_blocks=self.hyperparameters.conformer_blocks,
            dropout=self.hyperparameters.branch_dropout,
            drop_path=self.hyperparameters.drop_path_rate,
            layer_scale_init=self.hyperparameters.layer_scale_init,
            rel_pos_max_distance=self.hyperparameters.rel_pos_max_distance,
            pooling_hidden_dim=self.hyperparameters.pooling_hidden_dim,
            use_feature_normalization=self.toggles.use_feature_normalization,
            feature_norm_eps=self.hyperparameters.feature_norm_eps,
        )
        self.physics_path = PhysicsPath(
            spec=self.feature_spec,
            constants=self.constants,
            dropout=self.hyperparameters.branch_dropout,
        )

        self.physics_to_attn = nn.Sequential(
            nn.LayerNorm(self.physics_embedding_dim),
            nn.Linear(
                self.physics_embedding_dim, self.hyperparameters.conformer_d_model
            ),
        )
        self.cross_attention = CrossAttentionFusion(
            dim=self.hyperparameters.conformer_d_model,
            n_heads=self.hyperparameters.conformer_heads,
            dropout=self.hyperparameters.branch_dropout,
            drop_path=self.hyperparameters.drop_path_rate,
            layer_scale_init=self.hyperparameters.layer_scale_init,
            rel_pos_max_distance=self.hyperparameters.cross_rel_pos_max_distance,
        )
        self.cross_pool = AttentiveStatsPooling(
            self.hyperparameters.conformer_d_model,
            hidden_dim=self.hyperparameters.pooling_hidden_dim,
        )
        self.cross_pool_proj = nn.Sequential(
            nn.LayerNorm(self.hyperparameters.conformer_d_model * 2),
            nn.Linear(
                self.hyperparameters.conformer_d_model * 2,
                self.hyperparameters.conformer_d_model,
            ),
        )

        self.physics_fusion_proj = nn.Sequential(
            nn.LayerNorm(self.physics_embedding_dim),
            nn.Linear(self.physics_embedding_dim, self.physics_fusion_dim),
        )
        self.physics_gate = ReliabilityAdaptivePhysicsGate(
            acoustic_dim=self.hyperparameters.conformer_d_model,
            physics_dim=self.physics_embedding_dim,
            reliability_dim=self.physics_path.reliability_dim,
            hidden_dim=self.hyperparameters.physics_gate_hidden_dim,
            dropout=self.hyperparameters.branch_dropout,
            gate_floor=self.hyperparameters.physics_gate_floor,
        )
        self.fusion_se = SqueezeExcitationVector(dim=self.fusion_dim, reduction=4)
        self.conditional_ln = ConditionalLayerNorm(
            dim=self.fusion_dim, n_domains=self.domain_classes
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.fusion_dim, self.hyperparameters.fused_dim),
            nn.GELU(),
            nn.Dropout(self.hyperparameters.dropout),
        )
        self.fusion_out_norm = nn.LayerNorm(self.hyperparameters.fused_dim)

        self.height_head = BayesianHeightHead(
            in_dim=self.hyperparameters.fused_dim,
            hidden_dim=self.hyperparameters.regression_hidden_dim,
            dropout=self.hyperparameters.dropout,
            var_scale=self.hyperparameters.regression_var_scale,
            min_var=self.constants.eps,
        )
        self.height_adapter = HeightFeatureAdapter(
            fused_dim=self.hyperparameters.fused_dim,
            physics_dim=self.physics_embedding_dim,
            hidden_dim=self.hyperparameters.height_adapter_hidden_dim,
            dropout=self.hyperparameters.branch_dropout,
            adapter_scale=self.hyperparameters.height_adapter_scale,
        )
        self.physics_height_residual = nn.Sequential(
            nn.LayerNorm(self.physics_embedding_dim),
            nn.Linear(self.physics_embedding_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.weight_head = ProbabilisticRegressionHead(
            in_dim=self.hyperparameters.fused_dim,
            hidden_dim=self.hyperparameters.regression_hidden_dim,
            dropout=self.hyperparameters.dropout,
            var_scale=self.hyperparameters.regression_var_scale,
            min_var=self.constants.eps,
        )
        self.age_head = ProbabilisticRegressionHead(
            in_dim=self.hyperparameters.fused_dim,
            hidden_dim=self.hyperparameters.regression_hidden_dim,
            dropout=self.hyperparameters.dropout,
            var_scale=self.hyperparameters.regression_var_scale,
            min_var=self.constants.eps,
        )
        self.shoulder_head = (
            ProbabilisticRegressionHead(
                in_dim=self.hyperparameters.fused_dim,
                hidden_dim=self.hyperparameters.regression_hidden_dim,
                dropout=self.hyperparameters.dropout,
                var_scale=self.hyperparameters.regression_var_scale,
                min_var=self.constants.eps,
            )
            if self.toggles.use_shoulder_head
            else None
        )
        self.waist_head = (
            ProbabilisticRegressionHead(
                in_dim=self.hyperparameters.fused_dim,
                hidden_dim=self.hyperparameters.regression_hidden_dim,
                dropout=self.hyperparameters.dropout,
                var_scale=self.hyperparameters.regression_var_scale,
                min_var=self.constants.eps,
            )
            if self.toggles.use_waist_head
            else None
        )
        self.gender_head = nn.Linear(
            self.hyperparameters.fused_dim, self.gender_classes
        )
        self.height_prior_head = HeightPriorHead(
            physics_dim=self.physics_embedding_dim,
            domain_emb_dim=8,
            hidden_dim=96,
            dropout=self.hyperparameters.branch_dropout,
            constants=self.constants,
        )

        self.grl = GradientReversalLayer(lambda_init=0.0)
        self.domain_head = nn.Sequential(
            nn.Linear(self.hyperparameters.fused_dim, 128),
            nn.GELU(),
            nn.Dropout(self.hyperparameters.branch_dropout),
            nn.Linear(128, self.domain_classes),
        )

        self.diversity_proj = nn.Sequential(
            nn.LayerNorm(self.physics_embedding_dim),
            nn.Linear(
                self.physics_embedding_dim, self.hyperparameters.conformer_d_model
            ),
        )
        self.reliability_tower = MetadataReliabilityTower(
            hidden_dim=self.reliability_config.hidden_dim,
            embedding_dim=self.reliability_config.embedding_dim,
            dropout=self.hyperparameters.branch_dropout,
        )
        self.acoustic_physics_head = AcousticPhysicsConsistencyHead(
            in_dim=self.hyperparameters.conformer_d_model,
            hidden_dim=96,
            dropout=self.hyperparameters.branch_dropout,
            constants=self.constants,
        )

        self.vtsl_domain_raw = nn.Parameter(torch.zeros(self.domain_classes))
        self.loss_module = VocalTractSimulatorLossV2(
            target_stats=target_stats,
            focal_after_epoch=self.focal_after_epoch,
            ranking_margin=self.ranking_margin,
            nll_floor=float(self.hyperparameters.nll_floor),
            constants=self.constants,
            toggles=self.toggles,
            loss_weights=self.loss_weights,
            aggregation_config=self.aggregation_config,
            reliability_config=self.reliability_config,
            speaker_alignment=self.speaker_alignment_config,
        )
        # Wrap with Kendall auto-weighting if enabled
        if self.toggles.use_kendall_weights:
            self.loss_module = KendallMultiTaskLoss(self.loss_module)
        self._init_weights()

    def set_target_stats(
        self, target_stats: Optional[Mapping[str, Mapping[str, float]]]
    ) -> None:
        self.target_stats = target_stats
        self.loss_module.set_target_stats(target_stats)

    def _init_weights(self) -> None:
        skip_norm_types = (
            nn.MultiheadAttention,
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            ConditionalLayerNorm,
            SinusoidalPositionalEncoding,
        )
        for name, module in self.named_modules():
            if isinstance(module, PositiveLinear):
                continue
            if isinstance(module, skip_norm_types):
                continue
            if isinstance(module, nn.Linear):
                if name.endswith("mha.out_proj") or name.endswith("attn.out_proj"):
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def config_dict(self) -> Dict[str, Any]:
        """Return a serialization-friendly snapshot of the model configuration."""
        return {
            "hyperparameters": asdict(self.hyperparameters),
            "constants": asdict(self.constants),
            "toggles": asdict(self.toggles),
            "loss_weights": asdict(self.loss_weights),
            "aggregation": asdict(self.aggregation_config),
            "reliability": asdict(self.reliability_config),
            "speaker_alignment": asdict(self.speaker_alignment_config),
            "feature_spec": asdict(self.feature_spec),
            "target_stats": self.target_stats,
        }

    def parameter_summary(self) -> Dict[str, Any]:
        """Summarize parameter counts by major subsystem for quick inspection."""
        groups = {
            "acoustic_path": [self.acoustic_path],
            "physics_path": [self.physics_path],
            "fusion": [
                self.physics_to_attn,
                self.cross_attention,
                self.cross_pool,
                self.cross_pool_proj,
                self.physics_fusion_proj,
                self.physics_gate,
                self.fusion_se,
                self.conditional_ln,
                self.fusion_proj,
                self.fusion_out_norm,
            ],
            "heads": [
                self.height_head,
                self.physics_height_residual,
                self.weight_head,
                self.age_head,
                self.shoulder_head,
                self.waist_head,
                self.gender_head,
                self.height_adapter,
                self.height_prior_head,
                self.domain_head,
                self.diversity_proj,
                self.reliability_tower,
                self.acoustic_physics_head,
            ],
        }
        summary: Dict[str, Any] = {}
        total = 0
        for name, modules in groups.items():
            params = sum(
                p.numel()
                for module in modules
                if module is not None
                for p in module.parameters()
                if p.requires_grad
            )
            summary[name] = params
            total += params
        summary["total_trainable"] = total
        return summary

    def _domain_indices(
        self, domain: Optional[torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if domain is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        _validate_class_labels(domain, "domain", self.domain_classes)
        if domain.ndim != 1 or domain.size(0) != batch_size:
            raise ValueError(
                f"domain must have shape ({batch_size},), got {tuple(domain.shape)}"
            )
        return domain.to(device=device, dtype=torch.long)

    def _domain_correction(
        self, domain: Optional[torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        domain_idx = self._domain_indices(domain, batch_size, device)
        return (
            self.constants.domain_correction_base
            + self.constants.domain_correction_range
            * torch.sigmoid(self.vtsl_domain_raw[domain_idx])
        )

    def _quality_from_mask(
        self,
        padding_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        time_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if padding_mask is None:
            valid_value = 1.0 if time_steps is None else float(time_steps)
            valid_frames = torch.full((batch_size,), valid_value, device=device)
            return valid_frames, torch.ones(batch_size, device=device)
        valid_frames = (~padding_mask).sum(dim=1).float()
        quality = torch.sqrt(valid_frames.clamp(min=1.0))
        quality = quality / quality.max().clamp(min=1.0)
        return valid_frames, quality

    def _prepare_clip_metadata(
        self,
        clip_metadata: Optional[Mapping[str, Any]],
        *,
        batch_size: int,
        device: torch.device,
        valid_frames: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        keys = (
            "duration_s",
            "speech_ratio",
            "snr_db_estimate",
            "capture_quality_score",
            "voiced_ratio",
            "clipped_ratio",
            "distance_cm_estimate",
            "distance_confidence",
            "quality_ok",
            "feature_drift_zscore",
            "ood_zscore",
        )
        prepared: Dict[str, torch.Tensor] = {
            "valid_frames": valid_frames.to(device=device, dtype=torch.float32)
        }
        if clip_metadata is None:
            return prepared
        for key in keys:
            value = clip_metadata.get(key)
            if value is None:
                continue
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if tensor.ndim == 0:
                tensor = tensor.expand(batch_size)
            elif tensor.ndim > 1:
                tensor = tensor.reshape(batch_size, -1)[:, 0]
            _validate_batch_axis(tensor, batch_size, name=f"clip_metadata['{key}']", expected_ndim=1)
            prepared[key] = tensor.to(device=device, dtype=torch.float32)
        return prepared

    def _denorm_prediction(self, value: torch.Tensor, key: str) -> torch.Tensor:
        return _denorm_tensor(value, key, self.target_stats)

    def _run_regression_head(
        self,
        head: Optional[ProbabilisticRegressionHead],
        fused: torch.Tensor,
    ) -> RegressionHeadOutput:
        if head is None:
            return _zero_regression_output(fused, floor=self.constants.eps)
        return head(fused)

    def _append_regression_output(
        self,
        outputs: Dict[str, Any],
        key: str,
        prediction: RegressionHeadOutput,
    ) -> None:
        outputs[key] = prediction["mu"]
        outputs[f"{key}_mu"] = prediction["mu"]
        outputs[f"{key}_var"] = prediction["var"].clamp(min=self.constants.eps)
        outputs[f"{key}_logvar"] = prediction["logvar"].clamp(
            min=math.log(self.constants.eps), max=6.0
        )
        denorm_name = self.denorm_output_names.get(key)
        if denorm_name is not None:
            outputs[denorm_name] = self._denorm_prediction(prediction["mu"], key)

    def _build_diagnostics(
        self,
        preds: Mapping[str, Any],
        *,
        attention_maps: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        diagnostics = {
            "quality_score": preds["quality_score"],
            "quality_score_legacy": preds.get("quality_score_legacy"),
            "clip_reliability_prior": preds.get("clip_reliability_prior"),
            "usable_clip_probability": preds.get("usable_clip_probability"),
            "valid_frames": preds["valid_frames"],
            "physics_confidence": preds["physics_confidence"],
            "physics_gate": preds["physics_gate"],
            "physics_reliability": preds["physics_reliability"],
            "spacing_confidence": preds["spacing_confidence"],
            "vtl_confidence": preds["vtl_confidence"],
            "formant_stability_score": preds["formant_stability_score"],
            "formant_spacing_reliability": preds["formant_spacing_reliability"],
            "vtl_reliability": preds["vtl_reliability"],
            "formant_spacing_source": preds["formant_spacing_source"],
            "vtl_source": preds["vtl_source"],
            "formant_spacing_consistency": preds["formant_spacing_consistency"],
            "vtl_consistency": preds["vtl_consistency"],
            "formant_spacing_consistency_error": (
                preds["formant_spacing_pred"] - preds["imputed_formant_spacing"]
            ).abs(),
            "vtl_consistency_error": (preds["vtl_pred"] - preds["imputed_vtl"]).abs(),
        }
        if attention_maps is not None:
            diagnostics["cross_attention_maps"] = attention_maps
        return diagnostics

    def _empty_mixup_info(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]:
        return {
            "applied": torch.zeros(batch_size, device=device, dtype=torch.bool),
            "pair_index": torch.full(
                (batch_size,), -1, device=device, dtype=torch.long
            ),
            "lambda": torch.ones(batch_size, device=device, dtype=dtype),
        }

    def _resolve_current_epoch(
        self,
        current_epoch: Optional[int],
        targets: Optional[Mapping[str, Any]],
    ) -> int:
        if current_epoch is not None:
            return int(current_epoch)
        if targets is None or "epoch" not in targets or targets["epoch"] is None:
            return 0
        epoch_value = targets["epoch"]
        if isinstance(epoch_value, torch.Tensor):
            if epoch_value.numel() != 1:
                raise ValueError(
                    f"targets['epoch'] must be scalar if provided as a tensor, got shape {tuple(epoch_value.shape)}"
                )
            epoch_value = epoch_value.item()
        return int(epoch_value)

    def _validate_targets(
        self,
        targets: Optional[Mapping[str, Any]],
        batch_size: int,
        device: torch.device,
        domain: Optional[torch.Tensor] = None,
    ) -> None:
        if targets is None:
            return

        per_sample_keys = (
            "height",
            "weight",
            "age",
            "shoulder",
            "waist",
            "height_raw",
            "weight_raw",
            "age_raw",
            "shoulder_raw",
            "waist_raw",
            "f0_mean",
            "formant_spacing_mean",
            "vtl_mean",
            "weight_mask",
            "shoulder_mask",
            "waist_mask",
            "gender",
            "domain",
            "speaker_idx",
        )
        for key in per_sample_keys:
            value = targets.get(key)
            if isinstance(value, torch.Tensor):
                _validate_batch_axis(
                    value, batch_size, name=f"targets['{key}']", expected_ndim=1
                )
                if key.endswith("_mask") and not bool(torch.isfinite(value).all()):
                    raise ValueError(
                        f"targets['{key}'] must contain only finite values"
                    )

        if "gender" in targets and targets["gender"] is not None:
            _validate_class_labels(targets["gender"], "gender", self.gender_classes)

        if "domain" in targets and targets["domain"] is not None:
            target_domain = targets["domain"]
            _validate_class_labels(target_domain, "domain", self.domain_classes)
            target_domain_idx = self._domain_indices(target_domain, batch_size, device)
            if domain is not None:
                arg_domain_idx = self._domain_indices(domain, batch_size, device)
                if not torch.equal(target_domain_idx, arg_domain_idx):
                    raise ValueError(
                        "domain argument and targets['domain'] must match when both are provided"
                    )

    def _validate_branch_outputs(
        self,
        acoustic: Mapping[str, torch.Tensor],
        physics: Mapping[str, torch.Tensor],
        *,
        batch_size: int,
        time_steps: int,
    ) -> None:
        acoustic_sequence = acoustic["sequence"]
        sequence_attention = acoustic["sequence_attention"]
        acoustic_embedding = acoustic["acoustic_embedding"]
        physics_embedding = physics["physics_embedding"]
        physics_confidence = physics["physics_confidence"]

        if acoustic_sequence.shape != (
            batch_size,
            time_steps,
            self.acoustic_path.conformer_d_model,
        ):
            raise ValueError(
                "Unexpected acoustic sequence shape: "
                f"expected {(batch_size, time_steps, self.acoustic_path.conformer_d_model)}, got {tuple(acoustic_sequence.shape)}"
            )
        if sequence_attention.shape != (batch_size, time_steps):
            raise ValueError(
                f"Unexpected acoustic attention shape: expected {(batch_size, time_steps)}, got {tuple(sequence_attention.shape)}"
            )
        if acoustic_embedding.shape != (
            batch_size,
            self.acoustic_path.conformer_d_model,
        ):
            raise ValueError(
                "Unexpected acoustic embedding shape: "
                f"expected {(batch_size, self.acoustic_path.conformer_d_model)}, got {tuple(acoustic_embedding.shape)}"
            )
        if physics_embedding.shape != (batch_size, self.physics_embedding_dim):
            raise ValueError(
                f"Unexpected physics embedding shape: expected {(batch_size, self.physics_embedding_dim)}, got {tuple(physics_embedding.shape)}"
            )
        if physics_confidence.shape != (batch_size,):
            raise ValueError(
                f"Unexpected physics confidence shape: expected {(batch_size,)}, got {tuple(physics_confidence.shape)}"
            )

    def _coerce_speaker_index(
        self,
        speaker_idx: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if speaker_idx is None:
            return None
        _validate_batch_axis(
            speaker_idx, batch_size, name="speaker_idx", expected_ndim=1
        )
        if not bool(torch.isfinite(speaker_idx.to(dtype=torch.float32)).all()):
            raise ValueError("speaker_idx must contain only finite values")
        return speaker_idx.to(device=device, dtype=torch.long)

    def _aggregate_speaker_physics(
        self,
        *,
        speaker_idx: Optional[torch.Tensor],
        quality_score: torch.Tensor,
        physics: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        aggregated = {
            "multi_formant_vtl": physics["multi_formant_vtl"].clone(),
            "mean_spacing": physics["derived_formant_spacing"].clone(),
            "formant_stability": physics["formant_stability_score"].clone(),
            "vtl_confidence": physics["vtl_confidence"].clone(),
            "spacing_confidence": physics["spacing_confidence"].clone(),
            "residual_confidence": physics["physics_residual_confidence"].clone(),
        }
        if speaker_idx is None:
            return aggregated

        valid = speaker_idx >= 0
        if valid.sum() == 0:
            return aggregated

        weights = (quality_score * physics["physics_confidence"]).clamp(
            min=self.constants.eps
        )
        for speaker in torch.unique(speaker_idx[valid]):
            mask = speaker_idx == speaker
            speaker_weights = weights[mask]
            denom = speaker_weights.sum().clamp(min=self.constants.eps)
            for key, value in aggregated.items():
                speaker_mean = (value[mask] * speaker_weights).sum() / denom
                aggregated[key][mask] = speaker_mean
        return aggregated

    def _mix_targets(
        self, a: torch.Tensor, b: torch.Tensor, lam: torch.Tensor
    ) -> torch.Tensor:
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            return a
        return lam * a + (1.0 - lam) * b

    def _sample_mixup_partners(
        self, gender: torch.Tensor, domain_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        compatibility = (gender.unsqueeze(0) == gender.unsqueeze(1)) & (
            domain_idx.unsqueeze(0) == domain_idx.unsqueeze(1)
        )
        compatibility.fill_diagonal_(False)
        has_partner = compatibility.any(dim=1)
        random_scores = torch.rand(
            compatibility.shape, device=gender.device, dtype=torch.float32
        )
        random_scores = random_scores.masked_fill(~compatibility, -1.0)
        partners = random_scores.argmax(dim=1)
        fallback = torch.arange(gender.size(0), device=gender.device)
        partners = torch.where(has_partner, partners, fallback)
        return partners, has_partner

    def _apply_feature_mixup(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        domain: Optional[torch.Tensor],
        targets: Optional[MutableMapping[str, Any]],
        enabled: bool,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[MutableMapping[str, Any]],
        Dict[str, torch.Tensor],
    ]:
        """Mix feature sequences within same-gender / same-domain groups."""
        batch_size = features.size(0)
        device = features.device
        if padding_mask is not None:
            _validate_sequence_inputs(
                features,
                padding_mask,
                expected_feature_dim=features.size(-1),
                name="mixup_features",
            )
        if not enabled or not self.training or targets is None or batch_size < 2:
            return (
                features,
                padding_mask,
                targets,
                self._empty_mixup_info(batch_size, device, features.dtype),
            )

        gender = targets.get("gender")
        if gender is None:
            return (
                features,
                padding_mask,
                targets,
                self._empty_mixup_info(batch_size, device, features.dtype),
            )
        _validate_class_labels(gender, "gender", self.gender_classes)
        _validate_batch_axis(
            gender, batch_size, name="targets['gender']", expected_ndim=1
        )

        domain_idx = self._domain_indices(
            domain if domain is not None else targets.get("domain"), batch_size, device
        )
        gender = gender.to(device=device)

        mixed_features = features.clone()
        mixed_mask = padding_mask.clone() if padding_mask is not None else None
        mixed_targets = _clone_tensor_mapping(targets)
        if mixed_targets is None:
            raise RuntimeError(
                "targets unexpectedly became None during mixup preparation"
            )
        mixup_info = self._empty_mixup_info(batch_size, device, features.dtype)

        partners, applied = self._sample_mixup_partners(
            gender.to(dtype=torch.long), domain_idx
        )
        if not bool(applied.any()):
            return features, padding_mask, mixed_targets, mixup_info

        beta = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
        lam = beta.sample((batch_size,)).to(device=device, dtype=features.dtype)
        lam_feat = lam.view(batch_size, 1, 1)
        partner_features = features.index_select(0, partners)
        continuous_keys = [
            "height",
            "weight",
            "age",
            "shoulder",
            "waist",
            "height_raw",
            "weight_raw",
            "age_raw",
            "shoulder_raw",
            "waist_raw",
            "f0_mean",
            "formant_spacing_mean",
            "vtl_mean",
        ]

        source_a = features
        source_b = partner_features
        if padding_mask is not None:
            valid_a = (~padding_mask).to(dtype=features.dtype).unsqueeze(-1)
            valid_b = (
                (~padding_mask.index_select(0, partners))
                .to(dtype=features.dtype)
                .unsqueeze(-1)
            )
            source_a = source_a * valid_a
            source_b = source_b * valid_b
        mixed_values = lam_feat * source_a + (1.0 - lam_feat) * source_b
        mixed_features = torch.where(
            applied.view(batch_size, 1, 1), mixed_values, mixed_features
        )

        if mixed_mask is not None and padding_mask is not None:
            partner_mask = padding_mask.index_select(0, partners)
            mixed_mask = torch.where(
                applied.view(batch_size, 1), padding_mask & partner_mask, mixed_mask
            )

        for key in continuous_keys:
            if key in mixed_targets and isinstance(mixed_targets[key], torch.Tensor):
                source = mixed_targets[key]
                partner_source = source.index_select(0, partners)
                valid_pair = (
                    applied & torch.isfinite(source) & torch.isfinite(partner_source)
                )
                mixed_targets[key] = torch.where(
                    valid_pair, self._mix_targets(source, partner_source, lam), source
                )

        for mask_key in ("weight_mask", "shoulder_mask", "waist_mask"):
            if mask_key in mixed_targets and isinstance(
                mixed_targets[mask_key], torch.Tensor
            ):
                source_mask = mixed_targets[mask_key]
                partner_mask = source_mask.index_select(0, partners)
                combined_mask = torch.minimum(
                    source_mask.to(features.dtype), partner_mask.to(features.dtype)
                ).to(source_mask.dtype)
                mixed_targets[mask_key] = torch.where(
                    applied, combined_mask, source_mask
                )
        if "speaker_idx" in mixed_targets and isinstance(
            mixed_targets["speaker_idx"], torch.Tensor
        ):
            invalid_speaker_idx = torch.full_like(mixed_targets["speaker_idx"], -1)
            mixed_targets["speaker_idx"] = torch.where(
                applied, invalid_speaker_idx, mixed_targets["speaker_idx"]
            )

        mixup_info["pair_index"] = torch.where(
            applied, partners, mixup_info["pair_index"]
        )
        mixup_info["lambda"] = torch.where(applied, lam, mixup_info["lambda"])
        mixup_info["applied"] = applied

        return mixed_features, mixed_mask, mixed_targets, mixup_info

    def _forward_once(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        domain: Optional[torch.Tensor] = None,
        speaker_idx: Optional[torch.Tensor] = None,
        clip_metadata: Optional[Mapping[str, Any]] = None,
        lambda_grl: Optional[float] = None,
        return_diagnostics: bool = False,
        return_attention_maps: bool = False,
    ) -> Dict[str, Any]:
        """Single deterministic forward pass used by training and MC inference."""
        batch_size = features.size(0)
        device = features.device
        mask = _validate_sequence_inputs(
            features,
            padding_mask,
            expected_feature_dim=self.acoustic_path.input_dim,
            name="model_features",
        )
        _validate_class_labels(domain, "domain", self.domain_classes)
        speaker_idx = self._coerce_speaker_index(speaker_idx, batch_size, device)
        valid_frames, frame_quality = self._quality_from_mask(
            mask, batch_size, device, time_steps=features.size(1)
        )
        normalized_clip_metadata = self._prepare_clip_metadata(
            clip_metadata,
            batch_size=batch_size,
            device=device,
            valid_frames=valid_frames,
        )
        reliability_outputs = compose_clip_reliability(
            config=self.reliability_config,
            metadata=normalized_clip_metadata,
            valid_frames=valid_frames.to(device=device, dtype=features.dtype),
            crop_frames=features.size(1),
            tower=self.reliability_tower
            if self.reliability_config.mode == "learned"
            else None,
        )
        quality_score = reliability_outputs["clip_reliability_prior"].to(
            device=device, dtype=features.dtype
        )
        usable_clip_probability = torch.sigmoid(
            reliability_outputs["usable_clip_logit"].to(device=device, dtype=features.dtype)
        )
        reliability_embedding = reliability_outputs["reliability_embedding"].to(
            device=device, dtype=features.dtype
        )

        acoustic = self.acoustic_path(features, padding_mask=mask)
        physics = self.physics_path(features, padding_mask=mask)
        self._validate_branch_outputs(
            acoustic, physics, batch_size=batch_size, time_steps=features.size(1)
        )

        acoustic_sequence = acoustic["sequence"]
        acoustic_embedding = acoustic["acoustic_embedding"]
        raw_physics_embedding = physics["physics_embedding"]
        use_physics_branch = self.toggles.use_physics_branch
        use_cross_attention = self.toggles.use_cross_attention and use_physics_branch

        if use_physics_branch:
            if self.toggles.use_reliability_gate:
                physics_gate = self.physics_gate(
                    acoustic_embedding=acoustic_embedding,
                    physics_embedding=raw_physics_embedding,
                    physics_reliability=physics["physics_reliability"],
                    physics_confidence=physics["physics_confidence"],
                    quality_score=quality_score,
                )
                # Physics curriculum: force gate high early, let it learn to reduce later
                curriculum_epochs = self.hyperparameters.physics_gate_curriculum_epochs
                if curriculum_epochs > 0 and hasattr(self, "_current_epoch"):
                    epoch = self._current_epoch
                    if epoch <= curriculum_epochs:
                        # Linear ramp from floor to full freedom
                        progress = epoch / curriculum_epochs
                        floor = self.hyperparameters.physics_gate_floor
                        # Blend: early epochs use high gate, later epochs use learned gate
                        curriculum_gate = 1.0 - (1.0 - floor) * (1.0 - progress)
                        physics_gate = (
                            curriculum_gate * torch.ones_like(physics_gate)
                            + (1.0 - curriculum_gate) * physics_gate
                        )
            else:
                physics_gate = torch.ones(
                    batch_size, device=device, dtype=raw_physics_embedding.dtype
                )
            physics_embedding = raw_physics_embedding * physics_gate.unsqueeze(-1)
        else:
            physics_gate = torch.zeros(
                batch_size, device=device, dtype=raw_physics_embedding.dtype
            )
            physics_embedding = torch.zeros_like(raw_physics_embedding)
        speaker_physics = self._aggregate_speaker_physics(
            speaker_idx=speaker_idx,
            quality_score=quality_score,
            physics=physics,
        )
        height_prior_summary = torch.stack(
            [
                (speaker_physics["multi_formant_vtl"] - self.constants.default_vtl_cm)
                / max(self.constants.vtl_range_cm, self.constants.eps),
                speaker_physics["vtl_confidence"],
                speaker_physics["formant_stability"],
            ],
            dim=-1,
        )
        physics_token = self.physics_to_attn(physics_embedding).unsqueeze(1)

        cross_attention_maps: Optional[List[torch.Tensor]] = None
        if use_cross_attention:
            if return_attention_maps:
                refined_sequence, cross_attention_maps = self.cross_attention(
                    acoustic_tokens=acoustic_sequence,
                    physics_tokens=physics_token,
                    acoustic_mask=mask,
                    physics_mask=None,
                    return_attention=True,
                )
            else:
                refined_sequence = self.cross_attention(
                    acoustic_tokens=acoustic_sequence,
                    physics_tokens=physics_token,
                    acoustic_mask=mask,
                    physics_mask=None,
                )
            cross_pooled, cross_attn = self.cross_pool(
                refined_sequence, padding_mask=mask
            )
            cross_out = self.cross_pool_proj(cross_pooled)
        else:
            refined_sequence = acoustic_sequence
            cross_out = acoustic_embedding
            cross_attn = acoustic["sequence_attention"]
            if return_attention_maps:
                cross_attention_maps = []

        physics_fused = self.physics_fusion_proj(physics_embedding)
        fusion_vec = torch.cat([cross_out, physics_fused], dim=-1)
        if fusion_vec.size(-1) != self.fusion_dim:
            raise ValueError(
                f"Fusion dimension must be exactly {self.fusion_dim}, got {fusion_vec.size(-1)}"
            )

        fusion_vec = self.fusion_se(fusion_vec)
        fusion_vec = self.conditional_ln(fusion_vec, domain=domain)
        fused = self.fusion_out_norm(self.fusion_proj(fusion_vec))
        if self.toggles.use_height_adapter:
            height_features = self.height_adapter(
                fused_embedding=fused,
                physics_embedding=physics_embedding,
                physics_gate=physics_gate,
                quality_score=quality_score,
            )
        else:
            height_features = fused

        weight_pred = self._run_regression_head(self.weight_head, fused)
        age_pred = self._run_regression_head(self.age_head, fused)
        shoulder_pred = self._run_regression_head(self.shoulder_head, fused)
        waist_pred = self._run_regression_head(self.waist_head, fused)
        gender_logits = self.gender_head(fused)

        if self.toggles.use_height_prior and use_physics_branch:
            prior_height = physics_gate * self.height_prior_head(
                physics_embedding,
                age_pred["mu"],
                gender_logits,
                domain=domain,
                physics_summary=height_prior_summary,
            )
        else:
            prior_height = torch.zeros(batch_size, device=device, dtype=fused.dtype)
        if use_physics_branch:
            residual_reliability = 0.5 + 0.5 * torch.stack(
                [
                    physics["formant_stability_score"],
                    physics["spacing_confidence"],
                    physics["vtl_confidence"],
                    speaker_physics["residual_confidence"],
                ],
                dim=-1,
            ).mean(dim=-1).clamp(min=0.0, max=1.0)
            physics_residual = (
                physics_gate
                * residual_reliability
                * self.constants.physics_residual_tanh_scale
                * torch.tanh(
                    self.physics_height_residual(physics_embedding).squeeze(-1)
                )
            )
        else:
            physics_residual = torch.zeros(batch_size, device=device, dtype=fused.dtype)
        height_out = self.height_head(
            height_features,
            physics_residual=physics_residual if use_physics_branch else None,
            prior_residual=prior_height if self.toggles.use_height_prior else None,
        )

        if self.toggles.use_domain_adv:
            adv_in = self.grl(fused, lambda_override=lambda_grl)
            domain_logits = self.domain_head(adv_in)
        else:
            domain_logits = torch.zeros(
                batch_size, self.domain_classes, device=device, dtype=fused.dtype
            )

        corr = self._domain_correction(domain, batch_size, device)
        if use_physics_branch:
            spacing_pred = physics["formant_spacing_pred"]
            vtl_pred = corr * (
                self.constants.speed_of_sound_cm_per_s
                / (2.0 * spacing_pred.clamp(min=1.0))
            )
            vtl_pred = vtl_pred.clamp(
                min=self.constants.vtl_min_cm, max=self.constants.vtl_max_cm
            )
        else:
            spacing_pred = physics["imputed_formant_spacing"]
            vtl_pred = physics["imputed_vtl"]

        if self.toggles.use_acoustic_physics_consistency:
            aux_physics = self.acoustic_physics_head(acoustic_embedding)
        else:
            aux_physics = {
                "spacing": physics["imputed_formant_spacing"].detach(),
                "vtl": physics["imputed_vtl"].detach(),
            }

        physics_div_proj = (
            self.diversity_proj(physics_embedding)
            if use_physics_branch
            else torch.zeros(
                batch_size,
                self.hyperparameters.conformer_d_model,
                device=device,
                dtype=fused.dtype,
            )
        )

        output: Dict[str, Any] = {
            "gender_logits": gender_logits,
            "gender_probs": torch.softmax(gender_logits, dim=-1),
            "domain_logits": domain_logits,
            "formant_spacing_pred": spacing_pred,
            "vtl_pred": vtl_pred,
            "vtsl_correction": corr,
            "acoustic_embedding": acoustic_embedding,
            "acoustic_sequence": acoustic_sequence,
            "cross_attended_sequence": refined_sequence,
            "cross_output": cross_out,
            "cross_pool_attention": cross_attn,
            "physics_embedding": physics_embedding,
            "physics_embedding_raw": raw_physics_embedding,
            "physics_gate": physics_gate,
            "physics_confidence": physics["physics_confidence"],
            "physics_embedding_fusion": physics_fused,
            "physics_diversity_projection": physics_div_proj,
            "physics_input": physics["physics_input"],
            "physics_input_raw": physics["physics_input_raw"],
            "physics_input_normalized": physics["physics_input_normalized"],
            "multi_formant_vtl": physics["multi_formant_vtl"],
            "vtl_from_multi_formants": physics["vtl_from_multi_formants"],
            "vtl_from_f1_f4": physics["vtl_from_f1_f4"],
            "vtl_from_avg_spacing": physics["vtl_from_avg_spacing"],
            "spacing_std": physics["spacing_std"],
            "formant_ratio_f2_f1": physics["formant_ratio_f2_f1"],
            "formant_ratio_f3_f2": physics["formant_ratio_f3_f2"],
            "formant_ratio_f4_f3": physics["formant_ratio_f4_f3"],
            "spacing_confidence": physics["spacing_confidence"],
            "vtl_confidence": physics["vtl_confidence"],
            "formant_stability_score": physics["formant_stability_score"],
            "physics_residual_confidence": physics["physics_residual_confidence"],
            "observed_formant_spacing": physics["observed_formant_spacing"],
            "derived_formant_spacing": physics["derived_formant_spacing"],
            "imputed_formant_spacing": physics["imputed_formant_spacing"],
            "formant_spacing_reliability": physics["formant_spacing_reliability"],
            "observed_formant_spacing_reliability": physics[
                "observed_formant_spacing_reliability"
            ],
            "derived_formant_spacing_reliability": physics[
                "derived_formant_spacing_reliability"
            ],
            "formant_spacing_consistency": physics["formant_spacing_consistency"],
            "formant_spacing_source": physics["formant_spacing_source"],
            "formant_spacing_used_observed": physics["formant_spacing_used_observed"],
            "observed_vtl": physics["observed_vtl"],
            "derived_vtl": physics["derived_vtl"],
            "imputed_vtl": physics["imputed_vtl"],
            "vtl_reliability": physics["vtl_reliability"],
            "observed_vtl_reliability": physics["observed_vtl_reliability"],
            "derived_vtl_reliability": physics["derived_vtl_reliability"],
            "vtl_consistency": physics["vtl_consistency"],
            "vtl_source": physics["vtl_source"],
            "vtl_used_observed": physics["vtl_used_observed"],
            "physics_reliability": physics["physics_reliability"],
            "speaker_mean_multi_formant_vtl": speaker_physics["multi_formant_vtl"],
            "speaker_mean_formant_spacing": speaker_physics["mean_spacing"],
            "speaker_mean_formant_stability": speaker_physics["formant_stability"],
            "speaker_mean_vtl_confidence": speaker_physics["vtl_confidence"],
            "speaker_mean_spacing_confidence": speaker_physics["spacing_confidence"],
            "speaker_mean_physics_residual_confidence": speaker_physics[
                "residual_confidence"
            ],
            "height_prior_summary": height_prior_summary,
            "fused_embedding": fused,
            "height_features": height_features,
            "height_prior": prior_height,
            "height_physics_residual": physics_residual,
            "aux_spacing_pred": aux_physics["spacing"],
            "aux_vtl_pred": aux_physics["vtl"],
            "quality_score": quality_score,
            "quality_score_legacy": frame_quality,
            "clip_reliability_prior": quality_score,
            "usable_clip_probability": usable_clip_probability,
            "usable_clip_logit": reliability_outputs["usable_clip_logit"].to(
                device=device, dtype=features.dtype
            ),
            "reliability_embedding": reliability_embedding,
            "reliability_features": reliability_outputs["features"].to(
                device=device, dtype=features.dtype
            ),
            "clip_reliability_capture": reliability_outputs["capture"].to(
                device=device, dtype=features.dtype
            ),
            "clip_reliability_speech": reliability_outputs["speech"].to(
                device=device, dtype=features.dtype
            ),
            "clip_reliability_snr": reliability_outputs["snr"].to(
                device=device, dtype=features.dtype
            ),
            "clip_reliability_clipping": reliability_outputs["clipping"].to(
                device=device, dtype=features.dtype
            ),
            "clip_reliability_distance": reliability_outputs["distance"].to(
                device=device, dtype=features.dtype
            ),
            "clip_reliability_voiced": reliability_outputs["voiced"].to(
                device=device, dtype=features.dtype
            ),
            "clip_reliability_duration": reliability_outputs["duration"].to(
                device=device, dtype=features.dtype
            ),
            "clip_reliability_frames": reliability_outputs["frames"].to(
                device=device, dtype=features.dtype
            ),
            "clip_reliability_uncertainty": reliability_outputs["uncertainty"].to(
                device=device, dtype=features.dtype
            ),
            "clip_reliability_drift": reliability_outputs["drift"].to(
                device=device, dtype=features.dtype
            ),
            "valid_frames": valid_frames,
            "clip_metadata": normalized_clip_metadata,
            "target_stats": self.target_stats,
        }
        self._append_regression_output(
            output,
            "height",
            {
                "mu": height_out["mu"],
                "var": height_out["var"],
                "logvar": height_out["logvar"],
            },
        )
        output["height_mu_base"] = height_out["mu_base"]
        self._append_regression_output(output, "weight", weight_pred)
        self._append_regression_output(output, "age", age_pred)
        self._append_regression_output(output, "shoulder", shoulder_pred)
        self._append_regression_output(output, "waist", waist_pred)
        if return_attention_maps:
            output["cross_attention_maps"] = cross_attention_maps
        if return_diagnostics:
            output["diagnostics"] = self._build_diagnostics(
                output, attention_maps=cross_attention_maps
            )
        return output

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        domain: Optional[torch.Tensor] = None,
        speaker_idx: Optional[torch.Tensor] = None,
        clip_metadata: Optional[Mapping[str, Any]] = None,
        lambda_grl: Optional[float] = None,
        targets: Optional[MutableMapping[str, Any]] = None,
        current_epoch: Optional[int] = None,
        enable_mixup: Optional[bool] = None,
        return_aux: bool = True,
        return_diagnostics: bool = False,
        return_attention_maps: bool = False,
    ) -> Dict[str, Any]:
        batch_size = features.size(0)
        mask = _validate_sequence_inputs(
            features,
            padding_mask,
            expected_feature_dim=self.acoustic_path.input_dim,
            name="model_features",
        )
        if domain is not None:
            self._domain_indices(domain, batch_size, features.device)
        self._validate_targets(targets, batch_size, features.device, domain=domain)
        speaker_idx_tensor = self._coerce_speaker_index(
            speaker_idx, batch_size, features.device
        )
        if (
            targets is not None
            and "speaker_idx" in targets
            and targets["speaker_idx"] is not None
        ):
            target_speaker_idx = self._coerce_speaker_index(
                targets["speaker_idx"], batch_size, features.device
            )
            if speaker_idx_tensor is not None and not torch.equal(
                speaker_idx_tensor, target_speaker_idx
            ):
                raise ValueError(
                    "speaker_idx argument and targets['speaker_idx'] must match when both are provided"
                )
            speaker_idx_tensor = target_speaker_idx

        mixup_enabled = (
            self.use_feature_mixup if enable_mixup is None else bool(enable_mixup)
        )
        local_targets = _clone_tensor_mapping(targets)
        if (
            local_targets is not None
            and domain is not None
            and "domain" not in local_targets
        ):
            local_targets["domain"] = domain.detach().clone()
        if (
            local_targets is not None
            and speaker_idx_tensor is not None
            and "speaker_idx" not in local_targets
        ):
            local_targets["speaker_idx"] = speaker_idx_tensor.detach().clone()

        resolved_epoch = self._resolve_current_epoch(current_epoch, local_targets)
        if local_targets is not None:
            local_targets["epoch"] = resolved_epoch

        # Store epoch for physics curriculum gating
        self._current_epoch = resolved_epoch

        features_in, mask_in, mixed_targets, mixup_info = self._apply_feature_mixup(
            features,
            mask,
            domain=domain,
            targets=local_targets,
            enabled=mixup_enabled,
        )

        preds = self._forward_once(
            features_in,
            padding_mask=mask_in,
            domain=domain,
            speaker_idx=mixed_targets.get("speaker_idx")
            if mixed_targets is not None
            else speaker_idx_tensor,
            clip_metadata=clip_metadata,
            lambda_grl=lambda_grl,
            return_diagnostics=return_diagnostics,
            return_attention_maps=return_attention_maps,
        )
        preds["mixup"] = mixup_info
        preds["current_epoch"] = resolved_epoch

        aux = {
            "acoustic_embedding": preds["acoustic_embedding"],
            "physics_embedding": preds["physics_embedding"],
            "fused_embedding": preds["fused_embedding"],
            "quality_score": preds["quality_score"],
            "clip_reliability_prior": preds.get("clip_reliability_prior"),
        }
        if "diagnostics" in preds:
            aux["diagnostics"] = preds["diagnostics"]
        if return_aux:
            preds["aux"] = aux

        if mixed_targets is not None:
            self.loss_module.set_epoch(resolved_epoch)
            losses = self.loss_module(preds, mixed_targets)
            preds["losses"] = losses

        return preds

    def clip_gradients(self) -> Dict[str, Any]:
        assigned: Set[int] = set()
        diagnostics: Dict[str, Any] = {}

        group_specs: List[Tuple[str, float, List[Optional[nn.Module]]]] = [
            (
                "height_head",
                0.5,
                [
                    self.height_head,
                    self.height_adapter,
                    self.physics_height_residual,
                    self.height_prior_head,
                ],
            ),
            ("weight_head", 1.0, [self.weight_head]),
            ("age_head", 1.0, [self.age_head]),
            ("shoulder_head", 1.0, [self.shoulder_head]),
            ("waist_head", 1.0, [self.waist_head]),
            ("gender_head", 1.0, [self.gender_head]),
            ("domain_head", 1.0, [self.domain_head]),
            (
                "trunk",
                1.0,
                [
                    self.acoustic_path,
                    self.physics_path,
                    self.physics_to_attn,
                    self.cross_attention,
                    self.cross_pool,
                    self.cross_pool_proj,
                    self.physics_fusion_proj,
                    self.fusion_se,
                    self.conditional_ln,
                    self.fusion_proj,
                    self.fusion_out_norm,
                    self.diversity_proj,
                    self.acoustic_physics_head,
                ],
            ),
        ]

        def collect_params(
            modules: Iterable[Optional[nn.Module]],
        ) -> List[torch.nn.Parameter]:
            params: List[torch.nn.Parameter] = []
            for module in modules:
                if module is None:
                    continue
                for param in module.parameters():
                    if not param.requires_grad or id(param) in assigned:
                        continue
                    assigned.add(id(param))
                    params.append(param)
            return params

        for group_name, max_norm, modules in group_specs:
            params = collect_params(modules)
            params = [param for param in params if param.grad is not None]
            if not params:
                diagnostics[group_name] = {
                    "requested_max_norm": max_norm,
                    "total_norm_before": 0.0,
                    "n_params": 0,
                }
                continue
            total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
            diagnostics[group_name] = {
                "requested_max_norm": max_norm,
                "total_norm_before": float(total_norm),
                "n_params": len(params),
            }

        leftover = []
        for _, param in self.named_parameters():
            if param.requires_grad and id(param) not in assigned:
                if param.grad is not None:
                    leftover.append(param)
                assigned.add(id(param))
        if leftover:
            total_norm = torch.nn.utils.clip_grad_norm_(leftover, max_norm=1.0)
            diagnostics["unassigned_fallback"] = {
                "requested_max_norm": 1.0,
                "total_norm_before": float(total_norm),
                "n_params": len(leftover),
            }
        else:
            diagnostics["unassigned_fallback"] = {
                "requested_max_norm": 1.0,
                "total_norm_before": 0.0,
                "n_params": 0,
            }

        diagnostics["total_trainable_params"] = sum(
            1 for param in self.parameters() if param.requires_grad
        )
        return diagnostics

    def named_parameters_for_ema(self) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param

    def ema_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.detach().clone()
            for name, param in self.named_parameters_for_ema()
        }

    def _set_dropout_sampling(self, enabled: bool) -> None:
        """Enable stochasticity only for `MCDropout` modules during uncertainty inference."""
        for module in self.modules():
            if isinstance(module, MCDropout):
                module.mc_enabled = enabled
                module.eval()
            elif enabled and isinstance(module, nn.Dropout):
                module.eval()
            elif enabled and isinstance(
                module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
            ):
                module.eval()

    def _summarize_regression_passes(
        self,
        outputs: Sequence[Mapping[str, Any]],
        key: str,
    ) -> RegressionUncertaintySummary:
        means = torch.stack([out[f"{key}_mu"] for out in outputs], dim=0)
        logvars = torch.stack(
            [
                out[f"{key}_logvar"].clamp(min=math.log(self.constants.eps), max=6.0)
                for out in outputs
            ],
            dim=0,
        )
        epistemic = means.var(dim=0, unbiased=False)
        aleatoric = logvars.exp().mean(dim=0).clamp(min=self.constants.eps)
        total_var = (epistemic + aleatoric).clamp(min=self.constants.eps)
        return {
            "mean": means.mean(dim=0),
            "var": total_var,
            "std": torch.sqrt(total_var),
            "epistemic_var": epistemic,
            "aleatoric_var": aleatoric,
        }

    def _summarize_stochastic_passes(
        self, outputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not outputs:
            raise ValueError(
                "outputs must contain at least one stochastic forward pass"
            )
        gender_logits = torch.stack([out["gender_logits"] for out in outputs], dim=0)
        gender_probs = torch.softmax(gender_logits, dim=-1).mean(dim=0)
        gender_probs = gender_probs / gender_probs.sum(dim=-1, keepdim=True).clamp(
            min=EPS
        )

        ref = outputs[0]
        summary = {
            "gender": {"probs": gender_probs, "pred": gender_probs.argmax(dim=-1)},
            "quality_score": ref["quality_score"],
            "valid_frames": ref["valid_frames"],
        }
        for key in self.regression_targets:
            summary[key] = self._summarize_regression_passes(outputs, key)
        return summary

    def _merge_crop_predictions(self, crops: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not crops:
            raise ValueError("crops must contain at least one prediction dictionary")
        if len(crops) == 1:
            return crops[0]

        device = crops[0][self.regression_targets[0]]["mean"].device
        crop_quality = torch.stack([crop["quality_score"] for crop in crops], dim=0).to(
            device=device
        )
        if not bool(torch.isfinite(crop_quality).all()):
            raise ValueError("crop quality scores must be finite")

        def aggregate_regression(key: str) -> Dict[str, torch.Tensor]:
            means = torch.stack([crop[key]["mean"] for crop in crops], dim=0)
            vars_ = torch.stack([crop[key]["var"] for crop in crops], dim=0).clamp(
                min=1e-6
            )
            weights = (crop_quality / vars_).clamp(min=1e-6)
            mean = (means * weights).sum(dim=0) / weights.sum(dim=0).clamp(min=1e-6)
            var = 1.0 / weights.sum(dim=0).clamp(min=1e-6)
            epistemic = torch.stack(
                [
                    crop[key].get("epistemic_var", torch.zeros_like(var))
                    for crop in crops
                ],
                dim=0,
            )
            aleatoric = torch.stack(
                [crop[key].get("aleatoric_var", crop[key]["var"]) for crop in crops],
                dim=0,
            )
            epistemic_mean = (epistemic * weights).sum(dim=0) / weights.sum(
                dim=0
            ).clamp(min=1e-6)
            aleatoric_mean = (aleatoric * weights).sum(dim=0) / weights.sum(
                dim=0
            ).clamp(min=1e-6)
            return {
                "mean": mean,
                "var": var,
                "std": torch.sqrt(var),
                "epistemic_var": epistemic_mean.clamp(min=0.0),
                "aleatoric_var": aleatoric_mean.clamp(min=self.constants.eps),
            }

        gender_probs = torch.stack([crop["gender"]["probs"] for crop in crops], dim=0)
        gender_weights = crop_quality.unsqueeze(-1)
        gender_mean = (gender_probs * gender_weights).sum(dim=0) / gender_weights.sum(
            dim=0
        ).clamp(min=1e-6)
        gender_mean = gender_mean / gender_mean.sum(dim=-1, keepdim=True).clamp(min=EPS)

        quality_mean = crop_quality.mean(dim=0)
        valid_frames = (
            torch.stack([crop["valid_frames"] for crop in crops], dim=0)
            .max(dim=0)
            .values
        )

        merged = {
            "gender": {"probs": gender_mean, "pred": gender_mean.argmax(dim=-1)},
            "quality_score": quality_mean,
            "valid_frames": valid_frames,
        }
        for key in self.regression_targets:
            merged[key] = aggregate_regression(key)
        return merged

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        domain: Optional[torch.Tensor] = None,
        speaker_ids: Optional[Sequence[str]] = None,
        quality: Optional[torch.Tensor] = None,
        clip_metadata: Optional[Mapping[str, Any]] = None,
        n_samples: int = 10,
        deterministic: bool = False,
        crop_size: Optional[int] = None,
        n_crops: int = 1,
        aggregation: Optional[str] = None,
    ) -> Dict[str, Any]:
        if int(n_samples) != n_samples:
            raise ValueError(f"n_samples must be an integer, got {n_samples}")
        if int(n_samples) < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        was_training = self.training
        self.eval()
        mask = _validate_sequence_inputs(
            features,
            padding_mask,
            expected_feature_dim=self.acoustic_path.input_dim,
            name="uncertainty_features",
        )
        if domain is not None:
            self._domain_indices(domain, features.size(0), features.device)
        if quality is not None:
            _validate_batch_axis(
                quality, features.size(0), name="quality", expected_ndim=1
            )
            if not bool(torch.isfinite(quality).all()):
                raise ValueError("quality must contain only finite values")
        sample_count = int(n_samples)
        aggregation_method = aggregation or self.aggregation_config.method
        speaker_idx = None
        if speaker_ids is not None:
            speaker_map: Dict[str, int] = {}
            speaker_idx = torch.tensor(
                [
                    speaker_map.setdefault(str(speaker_id), len(speaker_map))
                    for speaker_id in speaker_ids
                ],
                device=features.device,
                dtype=torch.long,
            )

        try:
            crop_results: List[Dict[str, Any]] = []
            for crop_features, crop_mask, _ in build_multi_crops(
                features, mask, crop_size=crop_size, n_crops=n_crops
            ):
                if deterministic:
                    out = self.forward(
                        crop_features,
                        padding_mask=crop_mask,
                        domain=domain,
                        speaker_idx=speaker_idx,
                        clip_metadata=clip_metadata,
                        lambda_grl=0.0,
                        return_aux=False,
                    )
                    gender_probs = torch.softmax(out["gender_logits"], dim=-1)
                    crop_result: Dict[str, Any] = {
                        "gender": {
                            "probs": gender_probs,
                            "pred": gender_probs.argmax(dim=-1),
                        },
                        "quality_score": out["quality_score"],
                        "valid_frames": out["valid_frames"],
                    }
                    zero_epi = torch.zeros_like(out["height_mu"])
                    for key in self.regression_targets:
                        var = out[f"{key}_var"].clamp(min=self.constants.eps)
                        crop_result[key] = {
                            "mean": out[f"{key}_mu"],
                            "var": var,
                            "std": torch.sqrt(var),
                            "epistemic_var": zero_epi.clone(),
                            "aleatoric_var": var,
                        }
                else:
                    self._set_dropout_sampling(True)
                    try:
                        outputs = [
                            self.forward(
                                crop_features,
                                padding_mask=crop_mask,
                                domain=domain,
                                speaker_idx=speaker_idx,
                                clip_metadata=clip_metadata,
                                lambda_grl=0.0,
                                return_aux=False,
                            )
                            for _ in range(sample_count)
                        ]
                    finally:
                        self._set_dropout_sampling(False)
                    crop_result = self._summarize_stochastic_passes(outputs)
                crop_results.append(crop_result)

            utterance_result = self._merge_crop_predictions(crop_results)

            utterance_quality = (
                quality.to(device=features.device, dtype=torch.float32)
                if quality is not None
                else utterance_result["quality_score"]
            )
            variances: Dict[str, Optional[torch.Tensor]] = {"gender_probs": None}
            for key in self.regression_targets:
                variances[key] = utterance_result[key]["var"]

            speaker_result = None
            if speaker_ids is not None:
                speaker_preds: Dict[str, torch.Tensor] = {
                    "gender_probs": utterance_result["gender"]["probs"]
                }
                for key in self.regression_targets:
                    speaker_preds[key] = utterance_result[key]["mean"]
                speaker_result = aggregate_by_speaker(
                    speaker_ids=speaker_ids,
                    preds=speaker_preds,
                    variances=variances,
                    quality=utterance_quality,
                    metadata=clip_metadata,
                    method=aggregation_method,
                    aggregation_config=self.aggregation_config,
                    reliability_config=self.reliability_config,
                    target_stats=self.target_stats,
                )

            result = {
                "gender": utterance_result["gender"],
                "utterance": utterance_result,
                "speaker": speaker_result,
            }
            for key in self.regression_targets:
                result[key] = utterance_result[key]
            return result
        finally:
            if was_training:
                self.train()
            else:
                self.eval()

    def aggregate_by_speaker(
        self,
        speaker_ids: Sequence[str],
        preds: Mapping[str, torch.Tensor],
        variances: Optional[Mapping[str, Optional[torch.Tensor]]] = None,
        quality: Optional[torch.Tensor] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        method: str = "inverse_variance",
    ) -> Dict[str, Any]:
        return aggregate_by_speaker(
            speaker_ids=speaker_ids,
            preds=preds,
            variances=variances,
            quality=quality,
            metadata=metadata,
            method=method,
            aggregation_config=self.aggregation_config,
            reliability_config=self.reliability_config,
            target_stats=self.target_stats,
        )


def build_vocalmorph_v2(config: dict) -> VocalMorphV2:
    """Backward-compatible config builder for VocalMorph V2."""
    model_cfg = config.get("model", {})
    v2_cfg = model_cfg.get("v2", {})
    feat_cfg = config.get("features", {})

    n_mfcc = feat_cfg.get("mfcc", {}).get("n_mfcc", 40)
    include_delta = feat_cfg.get("mfcc", {}).get("include_delta", True)
    include_delta_delta = feat_cfg.get("mfcc", {}).get("include_delta_delta", True)
    n_formants = feat_cfg.get("formants", {}).get("n_formants", 4)

    target_stats = config.get("target_stats")
    if target_stats is None:
        target_stats = config.get("data", {}).get("target_stats")

    spec = PhysicsFeatureSpec(
        n_mfcc=n_mfcc,
        include_delta=include_delta,
        include_delta_delta=include_delta_delta,
        n_formants=n_formants,
    )

    legacy_toggle_keys = (
        "use_physics_branch",
        "use_cross_attention",
        "use_reliability_gate",
        "use_height_prior",
        "use_height_adapter",
        "use_domain_adv",
        "use_diversity_loss",
        "use_feature_mixup",
        "use_feature_normalization",
        "use_acoustic_physics_consistency",
        "use_ranking_loss",
        "use_speaker_consistency",
        "use_uncertainty_calibration",
        "use_shoulder_head",
        "use_waist_head",
    )
    toggle_cfg = dict(v2_cfg.get("toggles", {}))
    for key in legacy_toggle_keys:
        if key in v2_cfg and key not in toggle_cfg:
            toggle_cfg[key] = v2_cfg[key]

    constants = dataclass_from_mapping(PhysicsConstants, v2_cfg.get("constants"))
    toggles = dataclass_from_mapping(AblationToggles, toggle_cfg)
    loss_weights = dataclass_from_mapping(LossWeights, v2_cfg.get("loss_weights"))
    hyperparameters = dataclass_from_mapping(ModelHyperparameters, v2_cfg)
    aggregation_config = dataclass_from_mapping(
        AggregationConfig, v2_cfg.get("aggregation")
    )
    reliability_config = dataclass_from_mapping(
        ReliabilityConfig, v2_cfg.get("reliability")
    )
    speaker_alignment_config = dataclass_from_mapping(
        SpeakerAlignmentConfig,
        config.get("training", {}).get("speaker_alignment"),
    )

    input_dim = int(model_cfg.get("input_dim", 136))
    return VocalMorphV2(
        input_dim=input_dim,
        feature_spec=spec,
        ecapa_channels=hyperparameters.ecapa_channels,
        ecapa_scale=hyperparameters.ecapa_scale,
        conformer_d_model=hyperparameters.conformer_d_model,
        conformer_heads=hyperparameters.conformer_heads,
        conformer_blocks=hyperparameters.conformer_blocks,
        dropout=hyperparameters.dropout,
        target_stats=target_stats,
        use_feature_mixup=toggles.use_feature_mixup,
        mixup_alpha=hyperparameters.mixup_alpha,
        focal_after_epoch=hyperparameters.focal_after_epoch,
        ranking_margin=hyperparameters.ranking_margin,
        constants=constants,
        toggles=toggles,
        loss_weights=loss_weights,
        hyperparameters=hyperparameters,
        aggregation_config=aggregation_config,
        reliability_config=reliability_config,
        speaker_alignment_config=speaker_alignment_config,
    )


__all__ = ["VocalMorphV2", "build_vocalmorph_v2"]
