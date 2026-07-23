#!/usr/bin/env python
"""Export clip and speaker predictions from a VocalMorph checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.pibnn import build_model  # noqa: E402
from src.preprocessing.dataset import VocalMorphDataset, collate_fn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export checkpoint predictions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=640)
    parser.add_argument("--use-ema", action="store_true")
    return parser.parse_args()


def resolve(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(args: argparse.Namespace, checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    if args.config:
        with open(resolve(args.config), "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    cfg = checkpoint.get("config")
    if not isinstance(cfg, dict):
        raise RuntimeError("Checkpoint has no embedded config; pass --config")
    return cfg


def denorm(values: np.ndarray, stats: Mapping[str, Any], key: str) -> np.ndarray:
    s = stats.get(key, {})
    return values * float(s.get("std", 1.0)) + float(s.get("mean", 0.0))


def copy_ema_to_model(model: torch.nn.Module, ema_state: Mapping[str, torch.Tensor]) -> None:
    if not ema_state:
        return
    with torch.no_grad():
        params = dict(model.named_parameters())
        for name, value in ema_state.items():
            if name in params:
                params[name].copy_(value.to(device=params[name].device, dtype=params[name].dtype))


def metadata_from_batch(batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    meta: Dict[str, torch.Tensor] = {}
    for key in (
        "duration_s",
        "speech_ratio",
        "snr_db_estimate",
        "capture_quality_score",
        "voiced_ratio",
        "clipped_ratio",
        "distance_cm_estimate",
        "distance_confidence",
        "quality_ok",
    ):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            meta[key] = value
    padding_mask = batch.get("padding_mask")
    if isinstance(padding_mask, torch.Tensor):
        meta["valid_frames"] = (~padding_mask).sum(dim=1).float()
    return meta


def mae(y: Sequence[float], p: Sequence[float]) -> float:
    y_arr = np.asarray(y, dtype=np.float32)
    p_arr = np.asarray(p, dtype=np.float32)
    return float(np.mean(np.abs(y_arr - p_arr))) if y_arr.size else float("nan")


@torch.no_grad()
def export_split(
    *,
    split: str,
    model: torch.nn.Module,
    dataset: VocalMorphDataset,
    output_dir: Path,
    device: torch.device,
    target_stats: Mapping[str, Any],
    batch_size: int,
) -> Dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    model.eval()
    clip_rows = []
    speaker_ids = []
    pred_tensors = {"height": [], "weight": [], "age": [], "gender_probs": []}
    var_tensors = {"height": [], "weight": [], "age": []}
    quality_tensors = []
    meta_tensors: Dict[str, list[torch.Tensor]] = defaultdict(list)
    truth: Dict[str, Dict[str, Any]] = {}

    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        meta = metadata_from_batch(batch)
        out = model(
            batch["sequence"],
            padding_mask=batch.get("padding_mask"),
            domain=batch.get("source_id") if getattr(model, "expects_domain", False) else None,
            clip_metadata=meta,
        )
        probs = torch.softmax(out["gender_logits"], dim=-1)
        height_norm = out["height"].detach().float()
        height_cm = denorm(height_norm.cpu().numpy(), target_stats, "height")
        weight_cm = denorm(out["weight"].detach().float().cpu().numpy(), target_stats, "weight")
        age_year = denorm(out["age"].detach().float().cpu().numpy(), target_stats, "age")
        height_true = batch["height_raw"].detach().cpu().numpy()
        weight_true = batch["weight_raw"].detach().cpu().numpy()
        age_true = batch["age_raw"].detach().cpu().numpy()
        source_id = batch.get("source_id", torch.zeros_like(batch["height"]).long()).detach().cpu().numpy()
        gender = batch["gender"].detach().cpu().numpy()

        pred_tensors["height"].append(out["height"].detach().cpu())
        pred_tensors["weight"].append(out["weight"].detach().cpu())
        pred_tensors["age"].append(out["age"].detach().cpu())
        pred_tensors["gender_probs"].append(probs.detach().cpu())
        var_tensors["height"].append(out.get("height_var", torch.ones_like(out["height"])).detach().cpu())
        var_tensors["weight"].append(out.get("weight_var", torch.ones_like(out["height"])).detach().cpu())
        var_tensors["age"].append(out.get("age_var", torch.ones_like(out["height"])).detach().cpu())
        quality_tensors.append(out.get("quality_score", torch.ones_like(out["height"])).detach().cpu())
        for key, value in meta.items():
            meta_tensors[key].append(value.detach().cpu())

        for idx, sid in enumerate(batch["speaker_id"]):
            sid = str(sid)
            speaker_ids.append(sid)
            truth.setdefault(
                sid,
                {
                    "height_cm": float(height_true[idx]),
                    "weight_kg": float(weight_true[idx]),
                    "age": float(age_true[idx]),
                    "gender": int(gender[idx]),
                    "source_id": int(source_id[idx]),
                },
            )
            clip_rows.append(
                {
                    "speaker_id": sid,
                    "source_id": int(source_id[idx]),
                    "gender": int(gender[idx]),
                    "height_cm": float(height_true[idx]),
                    "height_pred_cm": float(height_cm[idx]),
                    "height_abs_error_cm": abs(float(height_true[idx]) - float(height_cm[idx])),
                    "weight_kg": float(weight_true[idx]),
                    "weight_pred_kg": float(weight_cm[idx]),
                    "age": float(age_true[idx]),
                    "age_pred": float(age_year[idx]),
                    "height_var_norm": float(var_tensors["height"][-1][idx].item()),
                    "quality_score": float(quality_tensors[-1][idx].item()),
                }
            )

    preds = {key: torch.cat(value).float() for key, value in pred_tensors.items()}
    variances = {key: torch.cat(value).float() for key, value in var_tensors.items()}
    quality = torch.cat(quality_tensors).float()
    metadata = {key: torch.cat(value).float() for key, value in meta_tensors.items() if value}
    legacy = model.aggregate_by_speaker(speaker_ids, preds, variances, quality, metadata, method="legacy_inverse_variance")
    omega = model.aggregate_by_speaker(speaker_ids, preds, variances, quality, metadata, method="omega_robust_reliability_pool")

    speaker_rows = []
    for sid, entry in legacy.get("speaker", {}).items():
        t = truth[str(sid)]
        height_pred = float(denorm(entry["height"].view(1).cpu().numpy(), target_stats, "height")[0])
        omega_entry = omega.get("speaker", {}).get(str(sid), entry)
        omega_height_pred = float(denorm(omega_entry["height"].view(1).cpu().numpy(), target_stats, "height")[0])
        speaker_rows.append(
            {
                "speaker_id": sid,
                "source_id": int(t["source_id"]),
                "gender": int(t["gender"]),
                "height_cm": float(t["height_cm"]),
                "height_pred_cm": height_pred,
                "height_abs_error_cm": abs(float(t["height_cm"]) - height_pred),
                "height_pred_omega_cm": omega_height_pred,
                "height_abs_error_omega_cm": abs(float(t["height_cm"]) - omega_height_pred),
                "weight_kg": float(t["weight_kg"]),
                "age": float(t["age"]),
                "n_clips": int(entry.get("count", 0)),
                "height_std_norm": float(entry.get("height_std", torch.tensor(float("nan"))).item()),
                "quality": float(entry.get("quality", torch.tensor(float("nan"))).item()),
            }
        )

    def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / f"{split}_clip_predictions.csv", clip_rows)
    write_csv(output_dir / f"{split}_speaker_predictions.csv", speaker_rows)
    summary = {
        "clip_height_mae": mae([r["height_cm"] for r in clip_rows], [r["height_pred_cm"] for r in clip_rows]),
        "speaker_height_mae": mae([r["height_cm"] for r in speaker_rows], [r["height_pred_cm"] for r in speaker_rows]),
        "speaker_height_mae_omega": mae([r["height_cm"] for r in speaker_rows], [r["height_pred_omega_cm"] for r in speaker_rows]),
        "n_clips": float(len(clip_rows)),
        "n_speakers": float(len(speaker_rows)),
    }
    return summary


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(resolve(args.checkpoint), map_location="cpu", weights_only=False)
    config = load_config(args, checkpoint)
    features_dir = resolve(args.features_dir or config.get("data", {}).get("features_dir", "data/features_vtl_ssl"))
    with open(features_dir / "target_stats.json", "r", encoding="utf-8") as handle:
        target_stats = json.load(handle)
    sample = np.load(next((features_dir / "train").glob("*.npz")), allow_pickle=True)
    config.setdefault("model", {})["input_dim"] = int(sample["sequence"].shape[1])
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    if hasattr(model, "set_target_stats"):
        model.set_target_stats(target_stats)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if args.use_ema and checkpoint.get("ema_state_dict") is not None:
        copy_ema_to_model(model, checkpoint["ema_state_dict"])

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, Dict[str, float]] = {}
    for split in [part.strip() for part in args.splits.split(",") if part.strip()]:
        dataset = VocalMorphDataset(
            str(features_dir / split),
            max_len=int(args.max_frames),
            target_stats=target_stats,
            crop_mode="center",
            augment=False,
        )
        summaries[split] = export_split(
            split=split,
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            device=device,
            target_stats=target_stats,
            batch_size=int(args.batch_size),
        )
        print(f"[export] {split}: {summaries[split]}", flush=True)
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
