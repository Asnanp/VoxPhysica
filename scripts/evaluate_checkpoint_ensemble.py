#!/usr/bin/env python
"""Evaluate a true checkpoint ensemble on val/test splits.

The script loads multiple trained checkpoints, runs each on the same canonical
val/test dataloaders, then compares several ensemble strategies selected by
validation speaker-balanced MAE:

- uniform mean
- validation-weighted mean
- rank-weighted mean
- prediction median
- trimmed mean
- inverse-variance mean when checkpoints expose variance
- each strategy with mean, median, and trimmed speaker pooling
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    parser = argparse.ArgumentParser(description="Evaluate checkpoint ensemble.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--output-dir", default="outputs/checkpoint_ensemble_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument("--tta-crops", type=int, default=1)
    parser.add_argument("--tta-samples", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--weight-temp", type=float, default=0.85)
    parser.add_argument(
        "--include-top-k-checkpoints",
        action="store_true",
        help="Expand each completed manifest member to its saved epoch_*.ckpt top-k checkpoints.",
    )
    parser.add_argument(
        "--max-checkpoints-per-member",
        type=int,
        default=6,
        help="Maximum epoch checkpoints to use per completed seed when --include-top-k-checkpoints is set.",
    )
    return parser.parse_args()


def resolve(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_target_stats(features_dir: Path) -> Optional[Dict[str, Any]]:
    path = features_dir / "target_stats.json"
    if not path.exists():
        return None
    return load_json(path)


def denorm(values: np.ndarray, key: str, target_stats: Optional[Mapping[str, Mapping[str, float]]]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if not target_stats:
        return arr
    stats = target_stats.get(key, {})
    return arr * float(stats.get("std", 1.0)) + float(stats.get("mean", 0.0))


def make_loader(config: Mapping[str, Any], split: str, target_stats: Optional[Mapping[str, Any]], args: argparse.Namespace) -> DataLoader:
    tc = config.get("training", {})
    features_dir = resolve(config.get("data", {}).get("features_dir", "data/features_audited"))
    dataset = VocalMorphDataset(
        str(features_dir / split),
        max_len=tc.get("max_feature_frames"),
        target_stats=target_stats,
        crop_mode=str(tc.get("eval_crop_mode", "center")),
        augment=False,
    )
    kwargs: Dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "collate_fn": collate_fn,
        "pin_memory": str(args.device).startswith("cuda") and torch.cuda.is_available(),
    }
    if int(args.num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def clip_metadata(batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    metadata: Dict[str, torch.Tensor] = {}
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
        "feature_drift_zscore",
        "ood_zscore",
    ):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            metadata[key] = value
    padding_mask = batch.get("padding_mask")
    if isinstance(padding_mask, torch.Tensor):
        metadata["valid_frames"] = (~padding_mask).sum(dim=1).to(dtype=torch.float32)
    return metadata


def copy_ema_to_model(model: torch.nn.Module, ema_state: Mapping[str, torch.Tensor]) -> bool:
    if not ema_state:
        return False
    copied = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in ema_state:
                param.copy_(ema_state[name].to(device=param.device, dtype=param.dtype))
                copied += 1
    return copied > 0


def load_model(config_path: Path, checkpoint_path: Path, input_dim: int, target_stats: Optional[Mapping[str, Any]], device: torch.device, use_ema: bool):
    config = load_yaml(config_path)
    config.setdefault("model", {})["input_dim"] = int(input_dim)
    config["target_stats"] = target_stats
    model = build_model(config).to(device)
    if hasattr(model, "set_target_stats"):
        model.set_target_stats(target_stats)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    ema_used = False
    if use_ema and checkpoint.get("ema_state_dict") is not None:
        ema_used = copy_ema_to_model(model, checkpoint["ema_state_dict"])
    model.eval()
    return model, config, int(checkpoint.get("epoch", -1)), ema_used


@torch.no_grad()
def infer_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    tta_crops: int,
    tta_samples: int,
    crop_size: int,
) -> Dict[str, Any]:
    preds = {"height": [], "weight": [], "age": [], "height_var": [], "gender": []}
    targets = {"height": [], "weight": [], "age": [], "gender": [], "source_id": []}
    speaker_ids: List[str] = []
    for batch in loader:
        batch = to_device(batch, device)
        if int(tta_crops) > 1 and hasattr(model, "predict_with_uncertainty"):
            result = model.predict_with_uncertainty(
                batch["sequence"],
                padding_mask=batch.get("padding_mask"),
                domain=batch.get("source_id") if getattr(model, "expects_domain", False) else None,
                speaker_ids=batch.get("speaker_id"),
                clip_metadata=clip_metadata(batch),
                deterministic=True,
                n_samples=int(tta_samples),
                crop_size=int(crop_size),
                n_crops=int(tta_crops),
            )
            height = result["height"]["mean"]
            weight = result["weight"]["mean"]
            age = result["age"]["mean"]
            height_var = result["height"].get("var", torch.ones_like(height))
            gender = result["gender"]["pred"]
        else:
            kwargs = {"padding_mask": batch.get("padding_mask"), "clip_metadata": clip_metadata(batch)}
            if getattr(model, "expects_domain", False):
                kwargs["domain"] = batch.get("source_id")
                kwargs["lambda_grl"] = 0.0
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                out = model(batch["sequence"], **kwargs)
            height = out["height"]
            weight = out["weight"]
            age = out["age"]
            height_var = out.get("height_var", torch.ones_like(height))
            gender = out["gender_logits"].argmax(-1)

        preds["height"].append(height.detach().cpu().float())
        preds["weight"].append(weight.detach().cpu().float())
        preds["age"].append(age.detach().cpu().float())
        preds["height_var"].append(height_var.detach().cpu().float())
        preds["gender"].append(gender.detach().cpu().long())
        targets["height"].append(batch["height_raw"].detach().cpu().float())
        targets["weight"].append(batch["weight_raw"].detach().cpu().float())
        targets["age"].append(batch["age_raw"].detach().cpu().float())
        targets["gender"].append(batch["gender"].detach().cpu().long())
        source = batch.get("source_id")
        if isinstance(source, torch.Tensor):
            targets["source_id"].append(source.detach().cpu().long())
        else:
            targets["source_id"].append(torch.zeros_like(batch["gender"].detach().cpu().long()))
        speaker_ids.extend([str(x) for x in batch.get("speaker_id", [])])

    out: Dict[str, Any] = {
        "speaker_id": np.asarray(speaker_ids, dtype=object),
        "height_pred_norm": torch.cat(preds["height"]).numpy().astype(np.float32),
        "weight_pred_norm": torch.cat(preds["weight"]).numpy().astype(np.float32),
        "age_pred_norm": torch.cat(preds["age"]).numpy().astype(np.float32),
        "height_var": torch.cat(preds["height_var"]).numpy().astype(np.float32),
        "gender_pred": torch.cat(preds["gender"]).numpy().astype(np.int64),
        "height_true": torch.cat(targets["height"]).numpy().astype(np.float32),
        "weight_true": torch.cat(targets["weight"]).numpy().astype(np.float32),
        "age_true": torch.cat(targets["age"]).numpy().astype(np.float32),
        "gender_true": torch.cat(targets["gender"]).numpy().astype(np.int64),
        "source_id": torch.cat(targets["source_id"]).numpy().astype(np.int64),
    }
    return out


def height_bin(height: float) -> str:
    if float(height) < 160.0:
        return "short"
    if float(height) < 175.0:
        return "medium"
    return "tall"


def pool_values(values: Sequence[float], mode: str) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return float("nan")
    if mode == "median":
        return float(np.median(arr))
    if mode == "trimmed" and arr.size >= 5:
        lo = int(math.floor(arr.size * 0.10))
        hi = int(math.ceil(arr.size * 0.90))
        return float(np.mean(np.sort(arr)[lo:hi]))
    return float(np.mean(arr))


def speaker_height_metrics(
    speaker_ids: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    source_id: np.ndarray,
    gender: np.ndarray,
    *,
    pool: str,
) -> Dict[str, float]:
    groups: Dict[str, Dict[str, Any]] = {}
    for idx, speaker in enumerate(speaker_ids):
        item = groups.setdefault(
            str(speaker),
            {"pred": [], "true": float(y_true[idx]), "source": int(source_id[idx]), "gender": int(gender[idx])},
        )
        item["pred"].append(float(y_pred[idx]))
    true_vals = []
    pred_vals = []
    source_vals = []
    gender_vals = []
    for item in groups.values():
        true_vals.append(float(item["true"]))
        pred_vals.append(pool_values(item["pred"], pool))
        source_vals.append(int(item["source"]))
        gender_vals.append(int(item["gender"]))
    true = np.asarray(true_vals, dtype=np.float32)
    pred = np.asarray(pred_vals, dtype=np.float32)
    source = np.asarray(source_vals, dtype=np.int64)
    gender_arr = np.asarray(gender_vals, dtype=np.int64)
    err = pred - true
    ae = np.abs(err)
    metrics: Dict[str, float] = {
        "height_mae_speaker": float(np.mean(ae)),
        "height_rmse_speaker": float(np.sqrt(np.mean(err * err))),
        "height_median_ae_speaker": float(np.median(ae)),
        "height_within_3cm_speaker": float(np.mean(ae <= 3.0)),
        "height_within_5cm_speaker": float(np.mean(ae <= 5.0)),
        "n_speakers": float(len(true)),
    }
    bin_maes = []
    for label in ("short", "medium", "tall"):
        mask = np.asarray([height_bin(v) == label for v in true], dtype=bool)
        if mask.any():
            val = float(np.mean(ae[mask]))
            metrics[f"height_{label}_speaker_mae"] = val
            metrics[f"height_{label}_speaker_n"] = float(mask.sum())
            bin_maes.append(val)
    if bin_maes:
        metrics["height_mae_speaker_balanced"] = float(np.mean(bin_maes))
    for sid_value, label in ((0, "timit"), (1, "nisp"), (2, "external")):
        mask = source == sid_value
        if mask.any():
            metrics[f"height_source_{label}_speaker_mae"] = float(np.mean(ae[mask]))
    for gid, label in ((0, "female"), (1, "male")):
        mask = gender_arr == gid
        if mask.any():
            metrics[f"height_gender_{label}_speaker_mae"] = float(np.mean(ae[mask]))
    return metrics


def clip_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = np.asarray(y_pred, dtype=np.float32) - np.asarray(y_true, dtype=np.float32)
    ae = np.abs(err)
    return {
        "height_mae_clip": float(np.mean(ae)),
        "height_rmse_clip": float(np.sqrt(np.mean(err * err))),
        "height_median_ae_clip": float(np.median(ae)),
    }


def evaluate_prediction(
    base: Mapping[str, Any],
    pred_height_cm: np.ndarray,
    *,
    pool: str,
) -> Dict[str, float]:
    metrics = clip_metrics(base["height_true"], pred_height_cm)
    metrics.update(
        speaker_height_metrics(
            base["speaker_id"],
            base["height_true"],
            pred_height_cm,
            base["source_id"],
            base["gender_true"],
            pool=pool,
        )
    )
    return metrics


def member_score(metrics: Mapping[str, float]) -> float:
    return float(metrics.get("height_mae_speaker_balanced", metrics.get("height_mae_speaker", 999.0)))


def model_weights(val_member_metrics: Sequence[Mapping[str, float]], temp: float, mode: str) -> np.ndarray:
    scores = np.asarray([member_score(m) for m in val_member_metrics], dtype=np.float32)
    if mode == "rank":
        order = np.argsort(scores)
        weights = np.zeros_like(scores)
        for rank, idx in enumerate(order):
            weights[idx] = 1.0 / float(rank + 1)
        return weights / max(float(weights.sum()), 1e-8)
    centered = scores - float(np.nanmin(scores))
    weights = np.exp(-centered / max(float(temp), 1e-3))
    return weights / max(float(weights.sum()), 1e-8)


def trimmed_mean_stack(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 4:
        return values.mean(axis=0)
    ordered = np.sort(values, axis=0)
    return ordered[1:-1].mean(axis=0)


def ensemble_arrays(
    pred_norm: np.ndarray,
    var: np.ndarray,
    *,
    weights: Optional[np.ndarray],
    method: str,
) -> np.ndarray:
    if method == "median":
        return np.median(pred_norm, axis=0)
    if method == "trimmed_mean":
        return trimmed_mean_stack(pred_norm)
    if method == "inverse_var":
        inv = 1.0 / np.maximum(var, 1e-4)
        return np.sum(pred_norm * inv, axis=0) / np.maximum(np.sum(inv, axis=0), 1e-8)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        return np.sum(pred_norm * w, axis=0)
    return pred_norm.mean(axis=0)


def write_prediction_csv(path: Path, base: Mapping[str, Any], pred_height_cm: np.ndarray, pred_std_cm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, Dict[str, Any]] = {}
    for idx, speaker in enumerate(base["speaker_id"]):
        item = grouped.setdefault(
            str(speaker),
            {
                "speaker_id": str(speaker),
                "height_cm": float(base["height_true"][idx]),
                "source_id": int(base["source_id"][idx]),
                "gender": int(base["gender_true"][idx]),
                "preds": [],
                "stds": [],
            },
        )
        item["preds"].append(float(pred_height_cm[idx]))
        item["stds"].append(float(pred_std_cm[idx]))
    fields = ["speaker_id", "source_id", "gender", "height_cm", "ensemble_pred_cm", "abs_error_cm", "mean_clip_std_cm", "n_clips"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in grouped.values():
            pred = float(np.mean(item["preds"]))
            writer.writerow(
                {
                    "speaker_id": item["speaker_id"],
                    "source_id": item["source_id"],
                    "gender": item["gender"],
                    "height_cm": f"{float(item['height_cm']):.6f}",
                    "ensemble_pred_cm": f"{pred:.6f}",
                    "abs_error_cm": f"{abs(pred - float(item['height_cm'])):.6f}",
                    "mean_clip_std_cm": f"{float(np.mean(item['stds'])):.6f}",
                    "n_clips": len(item["preds"]),
                }
            )


def _metric_from_checkpoint_name(path: Path) -> float:
    match = re.search(r"_metric_([0-9.]+)\.ckpt$", path.name)
    if not match:
        return float("inf")
    try:
        return float(match.group(1))
    except ValueError:
        return float("inf")


def _epoch_from_checkpoint_name(path: Path) -> int:
    match = re.search(r"epoch_(\d+)_", path.name)
    if not match:
        return -1
    return int(match.group(1))


def expand_top_k_checkpoints(members: Sequence[Mapping[str, Any]], max_per_member: int) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    limit = max(1, int(max_per_member))
    for member in members:
        ckpt_dir = Path(str(member.get("checkpoint_dir", "")))
        if not ckpt_dir.exists():
            ckpt_dir = Path(str(member.get("best_checkpoint", ""))).parent
        epoch_ckpts = sorted(
            ckpt_dir.glob("epoch_*_metric_*.ckpt"),
            key=lambda path: (_metric_from_checkpoint_name(path), _epoch_from_checkpoint_name(path)),
        )
        chosen = epoch_ckpts[:limit]
        if not chosen and Path(str(member.get("best_checkpoint", ""))).exists():
            chosen = [Path(str(member["best_checkpoint"]))]
        for rank, ckpt_path in enumerate(chosen):
            item = dict(member)
            item["best_checkpoint"] = str(ckpt_path)
            item["checkpoint_rank_within_seed"] = rank
            item["checkpoint_metric_from_name"] = _metric_from_checkpoint_name(ckpt_path)
            expanded.append(item)
    # Avoid evaluating duplicate files if best/epoch links point to the same path.
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in expanded:
        key = str(resolve(item["best_checkpoint"]).resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def manifest_members(args: argparse.Namespace) -> Tuple[Path, List[Dict[str, Any]]]:
    if args.manifest:
        manifest_path = resolve(args.manifest)
        manifest = load_json(manifest_path)
        completed = manifest.get("completed_members", [])
        members = [m for m in completed if Path(m.get("best_checkpoint", "")).exists()]
        if not members:
            members = [
                m
                for m in manifest.get("members", [])
                if m.get("status") in {"ok", "skipped_existing"} and Path(m.get("best_checkpoint", "")).exists()
            ]
        if not members:
            members = [m for m in manifest.get("members", []) if Path(m.get("best_checkpoint", "")).exists()]
        if args.include_top_k_checkpoints:
            members = expand_top_k_checkpoints(members, int(args.max_checkpoints_per_member))
        return Path(manifest.get("base_config", members[0]["config"] if members else "")), members
    if not args.config or not args.checkpoints:
        raise SystemExit("Provide --manifest or both --config and --checkpoints.")
    config = resolve(args.config)
    members = [
        {"seed": i, "config": str(config), "best_checkpoint": str(resolve(path))}
        for i, path in enumerate(args.checkpoints)
        if resolve(path).exists()
    ]
    return config, members


def main() -> int:
    args = parse_args()
    device = torch.device(str(args.device) if str(args.device) == "cpu" or torch.cuda.is_available() else "cpu")
    if str(args.device).startswith("cuda") and device.type != "cuda":
        raise SystemExit("CUDA requested but unavailable.")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config_path, members = manifest_members(args)
    if not members:
        raise SystemExit("No checkpoints found for ensemble evaluation.")
    base_config = load_yaml(resolve(members[0].get("config", str(base_config_path))))
    features_dir = resolve(base_config.get("data", {}).get("features_dir", "data/features_audited"))
    target_stats = load_target_stats(features_dir)
    loaders = {split: make_loader(base_config, split, target_stats, args) for split in ("val", "test")}
    input_dim = loaders["val"].dataset.infer_input_dim()
    print(f"[ensemble-eval] checkpoints={len(members)} input_dim={input_dim} device={device}", flush=True)

    split_model_outputs: Dict[str, List[Dict[str, Any]]] = {"val": [], "test": []}
    member_rows: List[Dict[str, Any]] = []
    for idx, member in enumerate(members):
        ckpt_path = resolve(member["best_checkpoint"])
        cfg_path = resolve(member.get("config", str(base_config_path)))
        model, _cfg, epoch, ema_used = load_model(cfg_path, ckpt_path, input_dim, target_stats, device, bool(args.use_ema))
        print(f"[ensemble-eval] {idx + 1}/{len(members)} {ckpt_path} epoch={epoch} ema={ema_used}", flush=True)
        val_out = infer_model(
            model,
            loaders["val"],
            device,
            use_amp=True,
            tta_crops=int(args.tta_crops),
            tta_samples=int(args.tta_samples),
            crop_size=int(args.crop_size),
        )
        test_out = infer_model(
            model,
            loaders["test"],
            device,
            use_amp=True,
            tta_crops=int(args.tta_crops),
            tta_samples=int(args.tta_samples),
            crop_size=int(args.crop_size),
        )
        split_model_outputs["val"].append(val_out)
        split_model_outputs["test"].append(test_out)
        val_height = denorm(val_out["height_pred_norm"], "height", target_stats)
        test_height = denorm(test_out["height_pred_norm"], "height", target_stats)
        val_m = evaluate_prediction(val_out, val_height, pool="mean")
        test_m = evaluate_prediction(test_out, test_height, pool="mean")
        member_rows.append(
            {
                "member": idx,
                "seed": member.get("seed"),
                "checkpoint": str(ckpt_path),
                "epoch": epoch,
                "ema_used": ema_used,
                "val": val_m,
                "test": test_m,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    val_member_metrics = [row["val"] for row in member_rows]
    soft_weights = model_weights(val_member_metrics, float(args.weight_temp), "softmax")
    rank_weights = model_weights(val_member_metrics, float(args.weight_temp), "rank")
    methods = [
        ("uniform_mean", None, "mean"),
        ("val_softmax_weighted", soft_weights, "mean"),
        ("val_rank_weighted", rank_weights, "mean"),
        ("prediction_median", None, "median"),
        ("trimmed_mean", None, "trimmed_mean"),
        ("inverse_variance", None, "inverse_var"),
    ]
    pools = ("mean", "median", "trimmed")
    method_rows: List[Dict[str, Any]] = []
    best_by_val: Optional[Dict[str, Any]] = None
    best_val_pred_cm: Optional[np.ndarray] = None
    best_val_std_cm: Optional[np.ndarray] = None
    best_test_pred_cm: Optional[np.ndarray] = None
    best_test_std_cm: Optional[np.ndarray] = None
    for method_name, weights, array_method in methods:
        for pool in pools:
            row: Dict[str, Any] = {"method": method_name, "speaker_pool": pool}
            pred_cache: Dict[str, np.ndarray] = {}
            std_cache: Dict[str, np.ndarray] = {}
            for split in ("val", "test"):
                outputs = split_model_outputs[split]
                pred_norm = np.stack([o["height_pred_norm"] for o in outputs], axis=0)
                var = np.stack([o["height_var"] for o in outputs], axis=0)
                ens_norm = ensemble_arrays(pred_norm, var, weights=weights, method=array_method)
                ens_cm = denorm(ens_norm, "height", target_stats)
                all_cm = np.stack([denorm(o["height_pred_norm"], "height", target_stats) for o in outputs], axis=0)
                pred_std_cm = all_cm.std(axis=0)
                pred_cache[split] = ens_cm
                std_cache[split] = pred_std_cm
                row[split] = evaluate_prediction(outputs[0], ens_cm, pool=pool)
            method_rows.append(row)
            score = float(row["val"].get("height_mae_speaker_balanced", row["val"].get("height_mae_speaker", 999.0)))
            if best_by_val is None or score < float(best_by_val["val"].get("height_mae_speaker_balanced", best_by_val["val"].get("height_mae_speaker", 999.0))):
                best_by_val = row
                best_val_pred_cm = pred_cache["val"]
                best_val_std_cm = std_cache["val"]
                best_test_pred_cm = pred_cache["test"]
                best_test_std_cm = std_cache["test"]

    assert best_by_val is not None and best_val_pred_cm is not None and best_val_std_cm is not None and best_test_pred_cm is not None and best_test_std_cm is not None
    report = {
        "members": member_rows,
        "weights": {
            "softmax": soft_weights.tolist(),
            "rank": rank_weights.tolist(),
        },
        "methods": method_rows,
        "selected_by_val": best_by_val,
        "args": vars(args),
    }
    (output_dir / "checkpoint_ensemble_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_prediction_csv(output_dir / "checkpoint_ensemble_predictions_val.csv", split_model_outputs["val"][0], best_val_pred_cm, best_val_std_cm)
    write_prediction_csv(output_dir / "checkpoint_ensemble_predictions_test.csv", split_model_outputs["test"][0], best_test_pred_cm, best_test_std_cm)
    lines = [
        "# Checkpoint Ensemble Evaluation",
        "",
        "## Selected By Validation",
        f"- Method: `{best_by_val['method']}`",
        f"- Speaker pooling: `{best_by_val['speaker_pool']}`",
        f"- Val speaker balanced MAE: `{best_by_val['val'].get('height_mae_speaker_balanced', float('nan')):.3f}cm`",
        f"- Test speaker MAE: `{best_by_val['test'].get('height_mae_speaker', float('nan')):.3f}cm`",
        f"- Test speaker balanced MAE: `{best_by_val['test'].get('height_mae_speaker_balanced', float('nan')):.3f}cm`",
        f"- Test short speaker MAE: `{best_by_val['test'].get('height_short_speaker_mae', float('nan')):.3f}cm`",
        "",
        "## Top Methods",
    ]
    ranked = sorted(method_rows, key=lambda row: float(row["val"].get("height_mae_speaker_balanced", row["val"].get("height_mae_speaker", 999.0))))
    for row in ranked[:18]:
        lines.append(
            f"- `{row['method']}` + `{row['speaker_pool']}`: "
            f"val_bal `{row['val'].get('height_mae_speaker_balanced', float('nan')):.3f}`, "
            f"test_spk `{row['test'].get('height_mae_speaker', float('nan')):.3f}`, "
            f"test_bal `{row['test'].get('height_mae_speaker_balanced', float('nan')):.3f}`, "
            f"short `{row['test'].get('height_short_speaker_mae', float('nan')):.3f}`"
        )
    lines.extend(["", "## Members"])
    for row in member_rows:
        lines.append(
            f"- seed `{row.get('seed')}` epoch `{row['epoch']}`: "
            f"val_bal `{row['val'].get('height_mae_speaker_balanced', float('nan')):.3f}`, "
            f"test_spk `{row['test'].get('height_mae_speaker', float('nan')):.3f}`"
        )
    (output_dir / "CHECKPOINT_ENSEMBLE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ensemble-eval] selected={best_by_val['method']} pool={best_by_val['speaker_pool']}", flush=True)
    print(f"[ensemble-eval] wrote {output_dir / 'CHECKPOINT_ENSEMBLE_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
