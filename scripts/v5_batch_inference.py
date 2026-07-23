#!/usr/bin/env python
"""
V5 Batch Inference: Generate clip-level predictions for push_toward_3cm.

Reads the V5 architecture checkpoint, runs inference on val/test splits,
and writes clip-level CSVs with:
  - speaker_id, height_pred_cm, height_var_norm, source, gender, height_cm

These CSVs are the inputs to push_toward_3cm.py's KNN blending ensemble.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.vocalmorph_v5 import build_v5_model
from src.preprocessing.dataset import VocalMorphDataset, collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V5 Batch Inference for push_toward_3cm")
    parser.add_argument("--config", default="configs/pibnn_rtx3060_v5_3cm_architecture.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints_v5_3cm_architecture/best.ckpt")
    parser.add_argument("--output-dir", default="outputs/v5_3cm_architecture_predictions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-ema", action="store_true", default=True)
    return parser.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _metadata_features(batch: dict, device: torch.device) -> Optional[torch.Tensor]:
    """Build metadata tensor matching what V5 expects."""
    source_id = batch.get("source_id")
    if source_id is None:
        source_oh = torch.zeros((batch["sequence"].shape[0], 4), device=device)
    else:
        source_oh = F.one_hot(
            source_id.to(device).clamp(min=0, max=3).long(), num_classes=4
        ).float()

    def clean(key: str, scale: float = 1.0, default: float = 0.0):
        value = batch.get(key)
        if value is None:
            value = torch.zeros(batch["sequence"].shape[0], device=device)
        else:
            value = value.to(device)
        value = torch.nan_to_num(value.float(), nan=default, posinf=default, neginf=default)
        return (value / scale).unsqueeze(1)

    scalar = torch.cat(
        [
            clean("duration_s", 10.0),
            clean("speech_ratio", 1.0),
            clean("snr_db_estimate", 35.0),
            clean("capture_quality_score", 1.0),
            clean("voiced_ratio", 1.0),
            clean("clipped_ratio", 0.10),
            clean("distance_cm_estimate", 100.0),
            clean("distance_confidence", 1.0),
        ],
        dim=1,
    )
    return torch.cat([scalar, source_oh], dim=1)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_stats: Optional[dict],
) -> tuple:
    """Run model inference on all clips, returning per-clip data."""
    model.eval()
    all_speaker_ids: List[str] = []
    all_heights_cm: List[float] = []
    all_preds_cm: List[float] = []
    all_vars: List[float] = []
    all_sources: List[str] = []
    all_genders: List[int] = []

    height_mean = float(target_stats.get("height", {}).get("mean", 170.0)) if target_stats else 170.0
    height_std = float(target_stats.get("height", {}).get("std", 9.0)) if target_stats else 9.0

    for batch_idx, batch in enumerate(loader):
        seq = batch["sequence"].to(device)
        mask = batch.get("padding_mask")
        if mask is not None:
            mask = mask.to(device)

        # V5 expects clip_metadata for quality metrics
        clip_meta = {
            "duration_s": batch.get("duration_s"),
            "speech_ratio": batch.get("speech_ratio"),
            "snr_db_estimate": batch.get("snr_db_estimate"),
            "capture_quality_score": batch.get("capture_quality_score"),
            "voiced_ratio": batch.get("voiced_ratio"),
            "clipped_ratio": batch.get("clipped_ratio"),
            "quality_ok": batch.get("quality_ok"),
            "distance_cm_estimate": batch.get("distance_cm_estimate"),
            "distance_confidence": batch.get("distance_confidence"),
            "valid_frames": batch.get("valid_frames"),
        }
        # Clean metadata - replace None with empty
        clip_meta = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in clip_meta.items()}

        out = model(seq, padding_mask=mask, clip_metadata=clip_meta)
        pred_norm = out["height"].cpu().numpy().astype(np.float32)
        pred_cm = pred_norm * height_std + height_mean
        var_norm = out.get("height_var", torch.ones_like(out["height"])).cpu().numpy().astype(np.float32)

        speaker_ids = batch.get("speaker_id", [f"unknown_{i}" for i in range(len(pred_cm))])
        height_raw = batch.get("height_raw")
        if height_raw is not None:
            heights = height_raw.cpu().numpy().astype(np.float32)
        else:
            heights = batch["height"].cpu().numpy().astype(np.float32) * height_std + height_mean

        sources = batch.get("source", [""] * len(pred_cm))
        genders = batch.get("gender", torch.zeros(len(pred_cm))).cpu().numpy().astype(int)

        for i in range(len(pred_cm)):
            all_speaker_ids.append(str(speaker_ids[i]))
            all_heights_cm.append(float(heights[i]))
            all_preds_cm.append(float(pred_cm[i]))
            all_vars.append(float(var_norm[i]))
            src = sources[i] if isinstance(sources[i], str) else str(sources[i])
            all_sources.append(src)
            all_genders.append(int(genders[i]))

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(loader)} batches", flush=True)

    return all_speaker_ids, all_heights_cm, all_preds_cm, all_vars, all_sources, all_genders


def speaker_aggregate(
    speaker_ids: List[str],
    heights_cm: List[float],
    preds_cm: List[float],
    vars_: List[float],
    sources: List[str],
    genders: List[int],
) -> tuple:
    """
    Aggregate clip predictions to speaker level.
    Returns both clip-level and speaker-level DataFrames.
    """
    import pandas as pd

    clip_df = pd.DataFrame({
        "speaker_id": speaker_ids,
        "height_cm": heights_cm,
        "height_pred_cm": preds_cm,
        "height_var_norm": vars_,
        "source": sources,
        "gender": genders,
    })

    # Speaker-level aggregation
    speaker_rows = []
    for sid, group in clip_df.groupby("speaker_id"):
        group_preds = group["height_pred_cm"].to_numpy(dtype=np.float32)
        group_vars = group["height_var_norm"].to_numpy(dtype=np.float32)
        group_heights = group["height_cm"].to_numpy(dtype=np.float32)

        # Inverse variance weighting
        vars_clamped = np.clip(group_vars, 1e-6, None)
        weights = 1.0 / vars_clamped
        var_weighted_pred = float(np.average(group_preds, weights=weights))
        simple_avg_pred = float(np.mean(group_preds))
        true_height = float(np.mean(group_heights))

        # Source from majority
        src = group["source"].mode().iloc[0] if len(group) > 0 else ""
        gender = int(group["gender"].mode().iloc[0]) if len(group) > 0 else -1

        speaker_rows.append({
            "speaker_id": sid,
            "height_cm": true_height,
            "source": src,
            "gender": gender,
            "n_clips": len(group),
            "v5_var_weighted_cm": var_weighted_pred,
            "v5_simple_avg_cm": simple_avg_pred,
        })

    speaker_df = pd.DataFrame(speaker_rows)
    return clip_df, speaker_df


def main() -> int:
    args = parse_args()
    config_path = _resolve(args.config)
    ckpt_path = _resolve(args.checkpoint)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[V5-Infer] Device: {device}")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    tc = config.get("training", {})
    dc = config.get("data", {})

    # Features path
    feat_dir = _resolve(dc.get("features_dir", "data/features_vtl_ssl"))
    stats_path = feat_dir / "target_stats.json"
    target_stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    if target_stats:
        print(f"[V5-Infer] Height stats: mean={target_stats['height']['mean']:.1f}, std={target_stats['height']['std']:.1f}")

    # Build model
    input_dim = int(config.get("model", {}).get("input_dim", 264))
    config.setdefault("model", {})["input_dim"] = input_dim

    model = build_v5_model(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[V5-Infer] Loaded checkpoint from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    # EMA if available
    if args.use_ema and "ema_state" in ckpt:
        from scripts.train_v4 import EMAWeights
        ema = EMAWeights(model)
        ema.load_state_dict(ckpt["ema_state"])
        ema.swap_in()
        print("[V5-Infer] Using EMA weights")

    # Data loaders
    max_len = int(tc.get("max_feature_frames", 640))
    eval_crop = str(tc.get("eval_crop_mode", "center"))
    batch_size = int(args.batch_size)
    nw = int(args.num_workers)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": nw,
        "collate_fn": collate_fn,
        "pin_memory": device.type == "cuda",
    }
    if nw > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loaders = {}
    for split in ("val", "test"):
        split_dir = feat_dir / split
        if not split_dir.exists():
            print(f"[V5-Infer] WARNING: {split_dir} does not exist, skipping")
            continue
        dataset = VocalMorphDataset(
            str(split_dir),
            max_len=max_len,
            target_stats=target_stats,
            crop_mode=eval_crop,
        )
        loaders[split] = DataLoader(dataset, **loader_kwargs)
        print(f"[V5-Infer] {split}: {len(dataset)} clips")

    # Run inference
    all_results = {}
    for split, loader in loaders.items():
        print(f"\n[V5-Infer] Running inference on {split}...")
        speaker_ids, heights, preds, vars_, sources, genders = run_inference(
            model, loader, device, target_stats
        )
        clip_df, speaker_df = speaker_aggregate(speaker_ids, heights, preds, vars_, sources, genders)
        all_results[split] = {
            "clip_df": clip_df,
            "speaker_df": speaker_df,
        }

        # Write clip-level CSV
        clip_path = output_dir / f"{split}_clip_predictions.csv"
        clip_df.to_csv(clip_path, index=False)
        print(f"  Wrote {clip_path} ({len(clip_df)} clips)")

        # Write speaker-level CSV
        speaker_path = output_dir / f"{split}_speaker_predictions.csv"
        speaker_df.to_csv(speaker_path, index=False)
        print(f"  Wrote {speaker_path} ({len(speaker_df)} speakers)")

        # Print summary
        if "height_cm" in speaker_df.columns:
            y = speaker_df["height_cm"].to_numpy(dtype=np.float32)
            for label, col in [("V5 Variance-Weighted", "v5_var_weighted_cm"),
                               ("V5 Simple Avg", "v5_simple_avg_cm")]:
                if col in speaker_df.columns:
                    pred = speaker_df[col].to_numpy(dtype=np.float32)
                    mae = float(np.mean(np.abs(pred - y)))
                    print(f"  {label} MAE: {mae:.3f}cm")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
