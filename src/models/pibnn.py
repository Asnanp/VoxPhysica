"""
VocalMorph - Physics-Informed Bayesian Neural Network (PIBNN)

Includes:
- Transformer encoder with MC dropout
- Multi-task heads for height/weight/age/gender
- Optional physics constraint loss
- Model factory supporting PIBNN and ECAPA-TDNN backbones
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for transformer input."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MCDropout(nn.Dropout):
    """Dropout that stays active during inference for MC sampling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, self.p, training=True, inplace=self.inplace)


class BayesianTransformerEncoder(nn.Module):
    """Transformer encoder with MC dropout enabled at inference."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self._replace_dropout_with_mc(dropout)

        self.norm = nn.LayerNorm(d_model)
        self.mc_dropout = MCDropout(p=dropout)

    def _replace_dropout_with_mc(self, p: float):
        for name, module in self.named_modules():
            if isinstance(module, nn.Dropout) and not isinstance(module, MCDropout):
                parent = self
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], MCDropout(p=p))

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            x = x.mean(dim=1)

        return self.mc_dropout(x)


class RegressionHead(nn.Module):
    """MLP regression head with MC dropout between layers."""

    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims[:-1]:
            layers += [nn.Linear(prev, h), nn.GELU(), MCDropout(p=dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ClassificationHead(nn.Module):
    """MLP classification head for gender prediction."""

    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], n_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims[:-1]:
            layers += [nn.Linear(prev, h), nn.GELU(), MCDropout(p=dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsConstraintLoss(nn.Module):
    """Differentiable physics penalties embedded in training loss."""

    def __init__(
        self,
        vtl_height_ratio: float = 6.7,
        speed_of_sound: float = 34000.0,
        vtl_weight: float = 0.1,
        formant_weight: float = 0.1,
        f0_gender_weight: float = 0.05,
    ):
        super().__init__()
        self.vtl_height_ratio = vtl_height_ratio
        self.speed_of_sound = speed_of_sound
        self.vtl_weight = vtl_weight
        self.formant_weight = formant_weight
        self.f0_gender_weight = f0_gender_weight

    def vtl_height_penalty(self, pred_height_cm: torch.Tensor, vtl_estimated: Optional[torch.Tensor]) -> torch.Tensor:
        if vtl_estimated is None:
            return torch.tensor(0.0, device=pred_height_cm.device)
        return F.mse_loss(pred_height_cm / self.vtl_height_ratio, vtl_estimated)

    def formant_vtl_penalty(self, pred_height_cm: torch.Tensor, formant_spacing: Optional[torch.Tensor]) -> torch.Tensor:
        if formant_spacing is None:
            return torch.tensor(0.0, device=pred_height_cm.device)
        vtl = (pred_height_cm / self.vtl_height_ratio).clamp(min=1.0)
        expected_delta_f = self.speed_of_sound / (2.0 * vtl)
        return F.mse_loss(expected_delta_f, formant_spacing)

    def f0_gender_penalty(self, gender_logits: torch.Tensor, f0_mean: Optional[torch.Tensor]) -> torch.Tensor:
        if f0_mean is None:
            return torch.tensor(0.0, device=gender_logits.device)
        probs = torch.softmax(gender_logits, dim=-1)
        p_female = probs[:, 0]
        p_male = probs[:, 1]
        penalty = p_male * F.relu(f0_mean - 180.0) + p_female * F.relu(165.0 - f0_mean)
        return penalty.mean()

    def forward(
        self,
        pred_height: torch.Tensor,
        gender_logits: torch.Tensor,
        vtl_estimated: Optional[torch.Tensor] = None,
        formant_spacing: Optional[torch.Tensor] = None,
        f0_mean: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses = {
            "vtl_height": self.vtl_weight * self.vtl_height_penalty(pred_height, vtl_estimated),
            "formant_vtl": self.formant_weight * self.formant_vtl_penalty(pred_height, formant_spacing),
            "f0_gender": self.f0_gender_weight * self.f0_gender_penalty(gender_logits, f0_mean),
        }
        losses["total_physics"] = sum(losses.values())
        return losses


class VocalMorphPIBNN(nn.Module):
    """Full PIBNN for VocalMorph."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.2,
        head_hidden: Tuple[int, ...] = (128, 64),
        physics_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.encoder = BayesianTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

        self.height_head = RegressionHead(d_model, head_hidden, dropout)
        self.weight_head = RegressionHead(d_model, head_hidden, dropout)
        self.age_head = RegressionHead(d_model, head_hidden, dropout)
        self.gender_head = ClassificationHead(d_model, head_hidden, n_classes=2, dropout=dropout)

        pc = physics_config or {}
        self.physics_loss = PhysicsConstraintLoss(
            vtl_height_ratio=pc.get("vtl_height_ratio", 6.7),
            speed_of_sound=pc.get("speed_of_sound", 34000.0),
            vtl_weight=pc.get("vtl_weight", 0.1),
            formant_weight=pc.get("formant_weight", 0.1),
            f0_gender_weight=pc.get("f0_gender_weight", 0.05),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        enc = self.encoder(features, src_key_padding_mask=padding_mask)
        return {
            "height": self.height_head(enc),
            "weight": self.weight_head(enc),
            "age": self.age_head(enc),
            "gender_logits": self.gender_head(enc),
        }

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        n_samples: int = 50,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        self.train()
        preds = {k: [] for k in ["height", "weight", "age", "gender_logits"]}

        for _ in range(n_samples):
            out = self.forward(features, padding_mask)
            for k in preds:
                preds[k].append(out[k].unsqueeze(0))

        self.eval()
        stacked = {k: torch.cat(v, dim=0) for k, v in preds.items()}
        gender_probs = torch.softmax(stacked["gender_logits"], dim=-1).mean(0)

        return {
            "height": {"mean": stacked["height"].mean(0), "std": stacked["height"].std(0)},
            "weight": {"mean": stacked["weight"].mean(0), "std": stacked["weight"].std(0)},
            "age": {"mean": stacked["age"].mean(0), "std": stacked["age"].std(0)},
            "gender": {"probs": gender_probs, "pred": gender_probs.argmax(-1)},
        }


class VocalMorphLoss(nn.Module):
    """Combined multi-task loss with optional weight masking + class weights."""

    def __init__(
        self,
        height_weight: float = 1.0,
        weight_weight: float = 1.0,
        age_weight: float = 1.0,
        gender_weight: float = 2.0,
        physics_weight: float = 0.2,
        gender_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.w = {
            "height": height_weight,
            "weight": weight_weight,
            "age": age_weight,
            "gender": gender_weight,
        }
        self.physics_weight = physics_weight
        self.mse = nn.MSELoss()
        self.gender_class_weights = gender_class_weights

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        physics_inputs: Optional[Dict[str, torch.Tensor]] = None,
        physics_module: Optional[PhysicsConstraintLoss] = None,
    ) -> Dict[str, torch.Tensor]:
        weight_mask = targets.get("weight_mask")
        if weight_mask is None:
            weight_loss = self.mse(preds["weight"], targets["weight"])
        else:
            mask = weight_mask.float()
            diff = (preds["weight"] - targets["weight"]) ** 2
            denom = mask.sum().clamp(min=1.0)
            weight_loss = (diff * mask).sum() / denom

        ce_weight = None
        if self.gender_class_weights is not None:
            ce_weight = self.gender_class_weights.to(preds["gender_logits"].device)

        losses = {
            "height": self.w["height"] * self.mse(preds["height"], targets["height"]),
            "weight": self.w["weight"] * weight_loss,
            "age": self.w["age"] * self.mse(preds["age"], targets["age"]),
            "gender": self.w["gender"] * F.cross_entropy(preds["gender_logits"], targets["gender"], weight=ce_weight),
        }

        if physics_module is not None and physics_inputs is not None:
            phys = physics_module(
                pred_height=preds["height"],
                gender_logits=preds["gender_logits"],
                vtl_estimated=physics_inputs.get("vtl_estimated"),
                formant_spacing=physics_inputs.get("formant_spacing"),
                f0_mean=physics_inputs.get("f0_mean"),
            )
            losses["physics"] = self.physics_weight * phys["total_physics"]
        else:
            losses["physics"] = torch.tensor(0.0, device=preds["height"].device)

        losses["total"] = losses["height"] + losses["weight"] + losses["age"] + losses["gender"] + losses["physics"]
        return losses


def build_model(config: dict) -> nn.Module:
    """Build model from config dict."""
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "PIBNN")).lower()

    feat_cfg = config.get("features", {})
    input_dim = int(model_cfg.get("input_dim", _compute_input_dim(feat_cfg)))

    if "vocalmorphv2" in model_name or "vocalmorph_v2" in model_name:
        from src.models.vocalmorphv2 import build_vocalmorph_v2

        cfg = dict(config)
        cfg.setdefault("model", {})
        cfg["model"]["input_dim"] = input_dim
        return build_vocalmorph_v2(cfg)

    if "ecapa" in model_name:
        from src.models.ecapa import ECAPAMultiTask

        ecapa_cfg = model_cfg.get("ecapa", {})
        return ECAPAMultiTask(
            input_dim=input_dim,
            channels=ecapa_cfg.get("channels", 512),
            emb_dim=ecapa_cfg.get("embedding_dim", 192),
            scale=ecapa_cfg.get("scale", 8),
            dropout=ecapa_cfg.get("dropout", 0.2),
        )

    enc_cfg = model_cfg.get("encoder", {})
    bay_cfg = model_cfg.get("bayesian", {})
    phys_cfg = config.get("physics", {})

    physics_config = {
        "vtl_height_ratio": phys_cfg.get("vtl_height_constraint", {}).get("ratio", 6.7),
        "speed_of_sound": 34000.0,
        "vtl_weight": phys_cfg.get("vtl_height_constraint", {}).get("penalty_weight", 0.1),
        "formant_weight": phys_cfg.get("formant_vtl_constraint", {}).get("penalty_weight", 0.1),
        "f0_gender_weight": phys_cfg.get("f0_gender_constraint", {}).get("penalty_weight", 0.05),
    }

    return VocalMorphPIBNN(
        input_dim=input_dim,
        d_model=enc_cfg.get("d_model", 256),
        n_heads=enc_cfg.get("n_heads", 8),
        n_layers=enc_cfg.get("n_layers", 4),
        feedforward_dim=enc_cfg.get("feedforward_dim", 1024),
        dropout=bay_cfg.get("dropout_rate", 0.2),
        physics_config=physics_config,
    )


def _compute_input_dim(feat_cfg: dict) -> int:
    """Estimate feature vector dimensionality from config and extractor output."""
    mfcc_cfg = feat_cfg.get("mfcc", {})
    n_mfcc = mfcc_cfg.get("n_mfcc", 40)
    delta_mult = 1 + int(mfcc_cfg.get("include_delta", True)) + int(mfcc_cfg.get("include_delta_delta", True))
    mfcc_dim = n_mfcc * delta_mult

    formant_dim = feat_cfg.get("formants", {}).get("n_formants", 4) * 2  # freq + bandwidth
    spectral_dim = 5  # centroid, rolloff, flux, bandwidth, contrast
    f0_dim = 1
    formant_spacing_dim = 1
    vtl_dim = 1

    return mfcc_dim + spectral_dim + formant_dim + f0_dim + formant_spacing_dim + vtl_dim


if __name__ == "__main__":
    b, t, d = 4, 100, 136
    model = VocalMorphPIBNN(input_dim=d, d_model=128, n_heads=4, n_layers=2, feedforward_dim=256)
    x = torch.randn(b, t, d)
    out = model(x)
    for k, v in out.items():
        print(k, tuple(v.shape))
