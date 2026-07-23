#!/usr/bin/env python
"""Evaluate a VocalMorph V4 checkpoint on val/test splits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_v4 import EMAWeights, compute_height_metrics, configure_cuda, seed_everything
from src.models.vocalmorph_v4 import build_v4_model
from src.preprocessing.dataset import VocalMorphDataset, collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained V4 checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-ema", action="store_true")
    return parser.parse_args()


def _resolve(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metadata_features(batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    source_id = batch.get("source_id")
    if source_id is None:
        source_oh = torch.zeros((batch["sequence"].shape[0], 3), device=device)
    else:
        source_oh = F.one_hot(source_id.to(device).clamp(min=0, max=2).long(), num_classes=3).float()

    def clean(key: str, scale: float = 1.0, clamp_min=None, clamp_max=None, transform=None):
        value = batch.get(key)
        if value is None:
            value = torch.zeros(batch["sequence"].shape[0], device=device)
        else:
            value = value.to(device)
        value = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if clamp_min is not None or clamp_max is not None:
            value = value.clamp(
                min=-float("inf") if clamp_min is None else clamp_min,
                max=float("inf") if clamp_max is None else clamp_max,
            )
        if transform == "log1p":
            value = torch.log1p(value.clamp(min=0.0))
        return (value / scale).unsqueeze(1)

    scalar = torch.cat(
        [
            clean("f0_mean", 250.0, 0.0, 500.0),
            clean("formant_spacing_mean", 1200.0, 300.0, 2000.0),
            clean("vtl_mean", 20.0, 5.0, 40.0),
            clean("duration_s", 2.5, 0.0, 20.0, transform="log1p"),
            clean("voiced_ratio", 1.0, 0.0, 1.0),
            clean("speech_ratio", 1.0, 0.0, 1.0),
            clean("snr_db_estimate", 40.0, -10.0, 80.0),
            clean("capture_quality_score", 1.0, 0.0, 1.0),
            clean("clipped_ratio", 0.10, 0.0, 0.50),
            clean("distance_cm_estimate", 100.0, 0.0, 300.0),
        ],
        dim=1,
    )
    return torch.cat([scalar, source_oh], dim=1)


@torch.no_grad()
def evaluate_split(model, loader, target_stats, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_preds, all_targets, all_speaker_ids = [], [], []
    use_meta = int(getattr(model, "meta_dim", 0)) > 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        metadata = _metadata_features(batch, device) if use_meta else None
        preds = model(batch["sequence"], padding_mask=batch.get("padding_mask"), metadata=metadata)
        all_preds.append(preds["height"].detach().cpu())
        all_targets.append(batch["height"].detach().cpu())
        all_speaker_ids.append(batch["speaker_id"])
    return compute_height_metrics(all_preds, all_targets, all_speaker_ids, target_stats)


def main() -> int:
    args = parse_args()
    config_path = _resolve(args.config)
    ckpt_path = _resolve(args.checkpoint)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tc = config.get("training", {})
    seed_everything(int(tc.get("seed", 42)))
    configure_cuda(bool(tc.get("allow_tf32", True)))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    feat_dir = _resolve(config.get("data", {}).get("features_dir", "data/features_v4"))
    stats_path = feat_dir / "target_stats.json"
    target_stats = _load_json(stats_path) if stats_path.exists() else None

    ssl_info_path = feat_dir / "ssl_info.json"
    if ssl_info_path.exists():
        config.setdefault("model", {})["input_dim"] = int(_load_json(ssl_info_path).get("input_dim", config["model"].get("input_dim", 136)))

    max_len = int(tc.get("max_feature_frames", 640))
    eval_crop_mode = str(tc.get("eval_crop_mode", "center"))
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "collate_fn": collate_fn,
        "pin_memory": device.type == "cuda",
    }
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loaders = {
        split: DataLoader(
            VocalMorphDataset(
                str(feat_dir / split),
                max_len=max_len,
                target_stats=target_stats,
                crop_mode=eval_crop_mode,
            ),
            **loader_kwargs,
        )
        for split in ("val", "test")
    }

    model = build_v4_model(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if not args.no_ema and "ema_state" in ckpt:
        ema = EMAWeights(model)
        ema.load_state_dict(ckpt["ema_state"])
        ema.swap_in()
        print(f"[eval-v4] using EMA weights from {ckpt_path}")
    else:
        print(f"[eval-v4] using raw model weights from {ckpt_path}")

    out = {"checkpoint": str(ckpt_path), "epoch": int(ckpt.get("epoch", -1)), "splits": {}}
    for split, loader in loaders.items():
        metrics = evaluate_split(model, loader, target_stats, device)
        out["splits"][split] = metrics
        print(
            f"[eval-v4] {split}: spk={metrics['height_mae_speaker']:.3f} "
            f"bal={metrics['height_mae_speaker_balanced']:.3f} "
            f"short={metrics.get('height_mae_short_speaker', float('nan')):.3f} "
            f"med={metrics.get('height_mae_medium_speaker', float('nan')):.3f} "
            f"tall={metrics.get('height_mae_tall_speaker', float('nan')):.3f}"
        )

    output_path = _resolve(args.output) if args.output else ckpt_path.parent.parent / "checkpoint_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[eval-v4] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
