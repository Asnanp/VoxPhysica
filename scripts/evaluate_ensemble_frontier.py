#!/usr/bin/env python
"""
VocalMorph Frontier Ensemble Evaluator.

Loads multiple checkpoints, runs inference on the test set,
and ensembles predictions via mean/median/weighted averaging.

Usage:
  python scripts/evaluate_ensemble_frontier.py \\
    --checkpoints outputs/checkpoints_v5_3cm_frontier/seed_*/best.ckpt \\
    --output outputs/ensemble_frontier_results.json

  python scripts/evaluate_ensemble_frontier.py \\
    --checkpoints outputs/checkpoints_v5_3cm_architecture/best.ckpt \\
             outputs/checkpoints_v5_1_direct_3cm/best.ckpt \\
    --output outputs/ensemble_best_of_both.json
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.pibnn import build_model
from src.preprocessing.dataset import VocalMorphDataset, collate_fn
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble multiple VocalMorph checkpoints")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Paths to checkpoint files or glob patterns")
    parser.add_argument("--config", type=str, default=None,
                        help="Config to use for model building (uses first checkpoint's config if not provided)")
    parser.add_argument("--features-dir", type=str, default="data/features_vtl_ssl/test",
                        help="Test features directory")
    parser.add_argument("--output", type=str, default="outputs/ensemble_results.json",
                        help="Path to write results JSON")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-crops", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--method", type=str, default="mean",
                        choices=["mean", "median", "weighted_by_val"],
                        help="Ensemble method")
    parser.add_argument("--val-metrics", type=str, nargs="+", default=None,
                        help="Validation MAEs per checkpoint (for weighted ensemble)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_checkpoints(patterns: List[str]) -> List[str]:
    paths = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if expanded:
            paths.extend(expanded)
        elif os.path.exists(pattern):
            paths.append(pattern)
        else:
            print(f"[WARN] No checkpoints found matching: {pattern}")
    # Deduplicate
    seen = set()
    unique = []
    for p in paths:
        real = os.path.realpath(p)
        if real not in seen:
            seen.add(real)
            unique.append(p)
    return unique


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    print(f"  Loading {checkpoint_path}...")
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def build_model_from_checkpoint(checkpoint: dict, config: dict, device: torch.device) -> torch.nn.Module:
    model = build_model(config)
    if hasattr(model, "set_target_stats"):
        target_stats = config.get("target_stats")
        if target_stats:
            model.set_target_stats(target_stats)
    model = model.to(device)

    # Load state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
        print("  [WARN] Non-strict state dict load (some keys missing/unexpected)")

    # If EMA is available, use EMA weights
    if checkpoint.get("ema_state_dict") is not None:
        ema_state = checkpoint["ema_state_dict"]
        # Try loading EMA weights
        try:
            model.load_state_dict(ema_state, strict=False)
            print("  Using EMA weights for evaluation")
        except Exception:
            pass

    model.eval()
    return model


def build_dataloader(features_dir: str, config: dict, batch_size: int, num_workers: int, crop_size: int, n_crops: int, device: torch.device):
    data_cfg = config.get("data", {})
    stats_path = os.path.join(os.path.dirname(features_dir), "target_stats.json")
    target_stats = None
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            target_stats = json.load(f)

    feat_dir = features_dir if os.path.isabs(features_dir) else os.path.join(ROOT, features_dir)

    if n_crops > 1:
        # Multi-crop: create dataset with center crop mode
        dataset = VocalMorphDataset(
            feat_dir,
            max_len=config.get("training", {}).get("max_feature_frames"),
            target_stats=target_stats,
            crop_mode="center",
            augment=False,
        )
    else:
        dataset = VocalMorphDataset(
            feat_dir,
            max_len=config.get("training", {}).get("max_feature_frames"),
            target_stats=target_stats,
            crop_mode="center",
            augment=False,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    return loader, target_stats


@torch.no_grad()
def predict(model, loader, device, n_crops=1, crop_size=None) -> Dict[str, List[torch.Tensor]]:
    """Run inference and collect predictions."""
    all_heights = []
    all_weights = []
    all_ages = []
    all_gender_preds = []
    all_quality = []
    all_height_vars = []
    all_speaker_ids = []
    all_targets = {
        "height_raw": [],
        "weight_raw": [],
        "age_raw": [],
        "gender": [],
        "source_id": [],
    }

    for batch in loader:
        seq = batch["sequence"].to(device, non_blocking=True)
        padding_mask = batch.get("padding_mask")
        if padding_mask is not None:
            padding_mask = padding_mask.to(device, non_blocking=True)

        kwargs = {
            "padding_mask": padding_mask,
        }
        if getattr(model, "expects_domain", False):
            kwargs["domain"] = batch.get("source_id").to(device) if isinstance(batch.get("source_id"), torch.Tensor) else None

        # Multi-crop: run multiple inferences on different crops
        heights = []
        weights = []
        ages = []
        gender_logits_list = []
        quality_scores = []
        height_vars = []

        if n_crops > 1 and hasattr(model, "predict_with_uncertainty"):
            # Use built-in uncertainty estimation with multiple crops
            result = model.predict_with_uncertainty(
                seq,
                padding_mask=padding_mask,
                domain=kwargs.get("domain"),
                clip_metadata=None,
                deterministic=False,
                n_samples=16,
                crop_size=crop_size,
                n_crops=n_crops,
            )
            heights.append(result["height"]["mean"])
            height_vars.append(result["height"]["var"])
            weights.append(result["weight"]["mean"])
            ages.append(result["age"]["mean"])
            probs = result["gender"]["probs"]
            gender_logits_list.append(probs)
            quality_scores.append(result["utterance"]["quality_score"])
        else:
            # Standard forward pass
            out = model(seq, **kwargs)
            heights.append(out["height"])
            height_vars.append(out.get("height_var", torch.ones_like(out["height"])))
            weights.append(out["weight"])
            ages.append(out["age"])
            gender_logits_list.append(out["gender_logits"])
            quality_scores.append(out.get("quality_score", torch.ones_like(out["height"])))

        all_heights.append(torch.cat(heights, dim=0).cpu())
        all_weights.append(torch.cat(weights, dim=0).cpu())
        all_ages.append(torch.cat(ages, dim=0).cpu())
        all_height_vars.append(torch.cat(height_vars, dim=0).cpu())

        gender_probs = torch.stack(gender_logits_list, dim=0)
        gender_pred = gender_probs.mean(dim=0).argmax(dim=-1)
        all_gender_preds.append(gender_pred.cpu())
        all_quality.append(torch.cat(quality_scores, dim=0).cpu())

        if batch.get("speaker_id"):
            all_speaker_ids.extend(batch["speaker_id"])

        for key in ("height_raw", "weight_raw", "age_raw", "gender", "source_id"):
            val = batch.get(key)
            if isinstance(val, torch.Tensor):
                all_targets[key].append(val.cpu())

    return {
        "height": torch.cat(all_heights),
        "weight": torch.cat(all_weights),
        "age": torch.cat(all_ages),
        "gender_pred": torch.cat(all_gender_preds),
        "height_var": torch.cat(all_height_vars),
        "quality": torch.cat(all_quality),
        "speaker_ids": all_speaker_ids,
        "targets": {k: torch.cat(v) for k, v in all_targets.items() if v},
    }


def compute_mae(pred: np.ndarray, true: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> float:
    if valid_mask is None:
        valid_mask = np.isfinite(pred) & np.isfinite(true)
    if not np.any(valid_mask):
        return float("nan")
    return float(np.mean(np.abs(pred[valid_mask] - true[valid_mask])))


def _denorm(values: np.ndarray, key: str, target_stats: Optional[dict]) -> np.ndarray:
    if target_stats is None:
        return values
    stats = target_stats.get(key, {})
    return values * float(stats.get("std", 1.0)) + float(stats.get("mean", 0.0))


def compute_speaker_mae(preds_all: Dict, targets_all: Dict, target_stats: Optional[dict]) -> Dict[str, float]:
    """Compute clip-level and speaker-level MAE."""
    speaker_ids = preds_all["speaker_ids"]
    height_pred = _denorm(preds_all["height"].numpy(), "height", target_stats)
    height_true = targets_all["height_raw"].numpy()

    # Clip-level
    clip_mae = compute_mae(height_pred, height_true)

    # Speaker-level
    if not speaker_ids:
        return {"clip_mae": clip_mae}

    speaker_preds = {}
    speaker_true = {}
    for i, sid in enumerate(speaker_ids):
        if sid not in speaker_preds:
            speaker_preds[sid] = []
            speaker_true[sid] = height_true[i]
        speaker_preds[sid].append(height_pred[i])

    speaker_maes = []
    for sid, preds in speaker_preds.items():
        speaker_mean = float(np.mean(preds))
        speaker_maes.append(abs(speaker_mean - float(speaker_true[sid])))

    return {
        "clip_mae": clip_mae,
        "speaker_mae": float(np.mean(speaker_maes)) if speaker_maes else float("nan"),
        "speaker_median_ae": float(np.median(speaker_maes)) if speaker_maes else float("nan"),
        "n_speakers": len(speaker_preds),
        "n_clips": len(height_pred),
    }


def ensemble_predictions(all_predictions: List[Dict], method: str = "mean",
                         val_metrics: Optional[List[float]] = None) -> Dict:
    """Ensemble multiple model predictions."""
    n_models = len(all_predictions)
    if n_models == 0:
        raise ValueError("No predictions to ensemble")

    # Stack predictions
    heights = torch.stack([p["height"] for p in all_predictions])
    weights = torch.stack([p["weight"] for p in all_predictions])
    ages = torch.stack([p["age"] for p in all_predictions])
    gender_preds = torch.stack([p["gender_pred"] for p in all_predictions])

    if method == "mean":
        height = heights.mean(dim=0)
        weight = weights.mean(dim=0)
        age = ages.mean(dim=0)
        # Majority vote for gender
        gender = gender_preds.mode(dim=0).values
    elif method == "median":
        height = heights.median(dim=0).values
        weight = weights.median(dim=0).values
        age = ages.median(dim=0).values
        gender = gender_preds.mode(dim=0).values
    elif method == "weighted_by_val":
        if val_metrics is None or len(val_metrics) != n_models:
            print("[WARN] No val_metrics provided, falling back to mean")
            return ensemble_predictions(all_predictions, "mean")
        # Invert MAE for weights (lower MAE = higher weight)
        weights_arr = 1.0 / (np.array(val_metrics) + 1e-6)
        weights_arr = weights_arr / weights_arr.sum()
        weights_pt = torch.tensor(weights_arr, dtype=torch.float32).view(-1, 1, 1)
        height = (heights * weights_pt).sum(dim=0)
        weight = (weights * weights_pt).sum(dim=0)
        age = (ages * weights_pt).sum(dim=0)
        gender = gender_preds.mode(dim=0).values
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    return {
        "height": height,
        "weight": weight,
        "age": age,
        "gender_pred": gender,
        "speaker_ids": all_predictions[0]["speaker_ids"],
        "targets": all_predictions[0]["targets"],
    }


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # Resolve checkpoint paths
    checkpoint_paths = resolve_checkpoints(args.checkpoints)
    if not checkpoint_paths:
        print("[ERROR] No valid checkpoints found!")
        sys.exit(1)
    print(f"Found {len(checkpoint_paths)} checkpoint(s):")
    for cp in checkpoint_paths:
        print(f"  - {cp}")

    # Load config
    if args.config:
        config_path = args.config
    else:
        # Use config from first checkpoint's directory
        ckpt_dir = os.path.dirname(checkpoint_paths[0])
        config_candidates = [
            os.path.join(ckpt_dir, "..", "..", "configs", f"{os.path.basename(ckpt_dir)}.yaml"),
            os.path.join(ROOT, "configs", f"{os.path.basename(ckpt_dir)}.yaml"),
            os.path.join(ROOT, "configs", "pibnn_rtx3060_v5_3cm_frontier.yaml"),
        ]
        config_path = None
        for candidate in config_candidates:
            if os.path.exists(candidate):
                config_path = candidate
                break
        if not config_path:
            config_path = os.path.join(ROOT, "configs", "pibnn_rtx3060_v5_3cm_architecture.yaml")

    print(f"Using config: {config_path}")
    config = load_config(config_path)

    # Build dataloader
    features_dir = args.features_dir if os.path.isabs(args.features_dir) else os.path.join(ROOT, args.features_dir)
    print(f"Loading test data from: {features_dir}")
    loader, target_stats = build_dataloader(
        features_dir, config, args.batch_size, args.num_workers,
        args.crop_size, args.n_crops, device
    )
    config["target_stats"] = target_stats
    print(f"Test samples: {len(loader.dataset)}")

    # Predict with each checkpoint
    all_predictions = []
    individual_results = []

    for ckpt_path in checkpoint_paths:
        print(f"\nProcessing checkpoint: {ckpt_path}")
        checkpoint = load_checkpoint(ckpt_path, device)
        model = build_model_from_checkpoint(checkpoint, config, device)
        print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

        preds = predict(model, loader, device, n_crops=args.n_crops, crop_size=args.crop_size)
        all_predictions.append(preds)

        # Compute individual metrics
        result = compute_speaker_mae(preds, preds["targets"], target_stats)
        individual_results.append({
            "checkpoint": ckpt_path,
            **result,
        })
        print(f"  Clip MAE: {result['clip_mae']:.4f} cm, Speaker MAE: {result['speaker_mae']:.4f} cm")

        del model
        torch.cuda.empty_cache()

    # Ensemble
    print(f"\n{'=' * 60}")
    print(f"Ensembling {len(all_predictions)} models with method: {args.method}")
    print(f"{'=' * 60}")

    val_metrics = [float(m) for m in args.val_metrics] if args.val_metrics else None
    ensemble = ensemble_predictions(all_predictions, method=args.method, val_metrics=val_metrics)
    ensemble_result = compute_speaker_mae(ensemble, ensemble["targets"], target_stats)

    print(f"\n{'=' * 60}")
    print(f"  ENSEMBLE RESULTS ({args.method})")
    print(f"{'=' * 60}")
    print(f"  Clip MAE:         {ensemble_result['clip_mae']:.4f} cm")
    print(f"  Speaker MAE:      {ensemble_result['speaker_mae']:.4f} cm")
    print(f"  Speaker MedianAE: {ensemble_result['speaker_median_ae']:.4f} cm")
    print(f"  Speakers:         {ensemble_result['n_speakers']}")
    print(f"  Clips:            {ensemble_result['n_clips']}")

    # Improvement over best individual
    best_individual = min(individual_results, key=lambda x: x["speaker_mae"])
    improvement = best_individual["speaker_mae"] - ensemble_result["speaker_mae"]
    print(f"\n  Best individual:  {best_individual['speaker_mae']:.4f} cm ({os.path.basename(best_individual['checkpoint'])})")
    print(f"  Improvement:      {improvement:.4f} cm")

    # Save results
    output_path = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_data = {
        "ensemble_method": args.method,
        "n_models": len(all_predictions),
        "ensemble": ensemble_result,
        "individual": individual_results,
        "checkpoints": checkpoint_paths,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults written to: {output_path}")

    # Print useful command for next steps
    print(f"\n{'=' * 60}")
    print(f"  To train more seeds (if needed):")
    print(f"  python scripts/train.py --config configs/pibnn_rtx3060_v5_3cm_frontier.yaml --seed 17")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
