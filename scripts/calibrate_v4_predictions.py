#!/usr/bin/env python
"""Validation-only calibration probe for VocalMorph V4 checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_v4 import EMAWeights, configure_cuda, seed_everything
from src.models.vocalmorph_v4 import build_v4_model
from src.preprocessing.dataset import VocalMorphDataset, collate_fn


SOURCE_NAMES = {0: "TIMIT", 1: "NISP", 2: "EXTERNAL"}


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


def _speaker_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    err = np.abs(pred - target)
    short = target < 160.0
    medium = (target >= 160.0) & (target < 175.0)
    tall = target >= 175.0

    def mean(mask: np.ndarray) -> float:
        return float(err[mask].mean()) if mask.any() else float("nan")

    mae = float(err.mean()) if err.size else float("nan")
    short_mae = mean(short)
    medium_mae = mean(medium)
    tall_mae = mean(tall)
    balanced_terms = [mae] + [v for v in (short_mae, medium_mae, tall_mae) if math.isfinite(v)]
    return {
        "height_mae_speaker": mae,
        "height_rmse_speaker": float(np.sqrt(np.mean((pred - target) ** 2))) if err.size else float("nan"),
        "height_median_ae_speaker": float(np.median(err)) if err.size else float("nan"),
        "height_mae_short_speaker": short_mae,
        "height_mae_medium_speaker": medium_mae,
        "height_mae_tall_speaker": tall_mae,
        "height_mae_speaker_balanced": float(np.mean(balanced_terms)) if balanced_terms else float("nan"),
    }


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@torch.no_grad()
def collect_speaker_predictions(
    *,
    name: str,
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_ema: bool,
) -> Dict[str, Dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tc = config.get("training", {})
    feat_dir = _resolve(config.get("data", {}).get("features_dir", "data/features_v4"))
    stats_path = feat_dir / "target_stats.json"
    target_stats = _load_json(stats_path) if stats_path.exists() else None
    max_len = int(tc.get("max_feature_frames", 640))
    eval_crop_mode = str(tc.get("eval_crop_mode", "center"))

    dataset = VocalMorphDataset(
        str(feat_dir / split),
        max_len=max_len,
        target_stats=target_stats,
        crop_mode=eval_crop_mode,
    )
    config.setdefault("model", {})["input_dim"] = int(dataset.infer_input_dim())

    loader_kwargs = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "collate_fn": collate_fn,
        "pin_memory": device.type == "cuda",
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)

    model = build_v4_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if use_ema and "ema_state" in ckpt:
        ema = EMAWeights(model)
        ema.load_state_dict(ckpt["ema_state"])
        ema.swap_in()

    h_mean = float(target_stats["height"]["mean"]) if target_stats else 0.0
    h_std = float(target_stats["height"]["std"]) if target_stats else 1.0
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    use_meta = int(getattr(model, "meta_dim", 0)) > 0

    model.eval()
    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        metadata = _metadata_features(batch_dev, device) if use_meta else None
        pred_norm = model(
            batch_dev["sequence"],
            padding_mask=batch_dev.get("padding_mask"),
            metadata=metadata,
        )["height"]
        pred_cm = _as_numpy(pred_norm) * h_std + h_mean
        target_cm = _as_numpy(batch["height_raw"]).astype(np.float64)
        gender = _as_numpy(batch["gender"]).astype(np.float64)
        source_id = _as_numpy(batch["source_id"]).astype(np.float64)

        extra_keys = ("f0_mean", "formant_spacing_mean", "vtl_mean", "duration_s")
        extras = {key: _as_numpy(batch[key]).astype(np.float64) for key in extra_keys if key in batch}
        for i, sid in enumerate(batch["speaker_id"]):
            rec = grouped[str(sid)]
            rec[f"pred_{name}"].append(float(pred_cm[i]))
            rec["target"].append(float(target_cm[i]))
            rec["gender"].append(float(gender[i]))
            rec["source_id"].append(float(source_id[i]))
            for key, values in extras.items():
                rec[key].append(float(values[i]))

    out: Dict[str, Dict[str, Any]] = {}
    for sid, rec in grouped.items():
        item: Dict[str, Any] = {}
        for key, values in rec.items():
            item[key] = float(np.mean(values))
        item["source_name"] = SOURCE_NAMES.get(int(round(item.get("source_id", 1.0))), "UNKNOWN")
        out[sid] = item
    return out


def _merge_predictions(per_model: Dict[str, Dict[str, Dict[str, Any]]], model_names: List[str], split: str):
    speaker_ids = sorted(set.intersection(*(set(per_model[name][split]) for name in model_names)))
    rows: List[Dict[str, Any]] = []
    for sid in speaker_ids:
        base: Dict[str, Any] = {"speaker_id": sid}
        for name in model_names:
            rec = per_model[name][split][sid]
            base[f"pred_{name}"] = rec[f"pred_{name}"]
            if "target" not in base:
                for key in ("target", "gender", "source_id", "f0_mean", "formant_spacing_mean", "vtl_mean", "duration_s"):
                    if key in rec:
                        base[key] = rec[key]
        rows.append(base)
    return rows


def _feature_matrix(rows: List[Dict[str, Any]], model_names: List[str]) -> np.ndarray:
    raw = np.asarray([[r[f"pred_{name}"] for name in model_names] for r in rows], dtype=np.float64)
    aux = np.asarray(
        [
            [
                r.get("gender", 0.0),
                r.get("source_id", 1.0),
                r.get("f0_mean", 0.0) / 250.0,
                r.get("formant_spacing_mean", 0.0) / 1200.0,
                r.get("vtl_mean", 0.0) / 20.0,
                math.log1p(max(0.0, r.get("duration_s", 0.0))) / 2.5,
            ]
            for r in rows
        ],
        dtype=np.float64,
    )
    summary = np.column_stack([raw.mean(axis=1), raw.min(axis=1), raw.max(axis=1), raw.std(axis=1)])
    return np.column_stack([raw, summary, aux])


def _targets(rows: List[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([r["target"] for r in rows], dtype=np.float64)


def _raw_pred(rows: List[Dict[str, Any]], name: str) -> np.ndarray:
    return np.asarray([r[f"pred_{name}"] for r in rows], dtype=np.float64)


def _add_affine_candidate(candidates, label, val_pred, test_pred, y_val):
    slope, intercept = np.polyfit(val_pred, y_val, deg=1)
    candidates[label] = (slope * val_pred + intercept, slope * test_pred + intercept)


def _add_piecewise_candidate(candidates, label, val_pred, test_pred, y_val):
    edges = np.quantile(val_pred, [0.0, 0.333333, 0.666667, 1.0])
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    val_out = np.zeros_like(val_pred)
    test_out = np.zeros_like(test_pred)
    global_slope, global_intercept = np.polyfit(val_pred, y_val, deg=1)
    for i in range(3):
        vmask = (val_pred >= edges[i]) & (val_pred < edges[i + 1])
        tmask = (test_pred >= edges[i]) & (test_pred < edges[i + 1])
        if vmask.sum() >= 8 and np.std(val_pred[vmask]) > 1e-6:
            slope, intercept = np.polyfit(val_pred[vmask], y_val[vmask], deg=1)
        else:
            slope, intercept = global_slope, global_intercept
        val_out[vmask] = slope * val_pred[vmask] + intercept
        test_out[tmask] = slope * test_pred[tmask] + intercept
    candidates[label] = (val_out, test_out)


def fit_candidates(val_rows, test_rows, model_names: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    y_val = _targets(val_rows)
    candidates: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for name in model_names:
        val_pred = _raw_pred(val_rows, name)
        test_pred = _raw_pred(test_rows, name)
        candidates[f"{name}:raw"] = (val_pred, test_pred)
        _add_affine_candidate(candidates, f"{name}:affine", val_pred, test_pred, y_val)
        _add_piecewise_candidate(candidates, f"{name}:piecewise_affine", val_pred, test_pred, y_val)

    try:
        from sklearn.isotonic import IsotonicRegression

        for name in model_names:
            val_pred = _raw_pred(val_rows, name)
            test_pred = _raw_pred(test_rows, name)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(val_pred, y_val)
            candidates[f"{name}:isotonic"] = (iso.predict(val_pred), iso.predict(test_pred))
    except Exception as exc:
        print(f"[calibrate-v4] isotonic skipped: {exc}")

    try:
        from sklearn.linear_model import HuberRegressor, RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        x_val = _feature_matrix(val_rows, model_names)
        x_test = _feature_matrix(test_rows, model_names)
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
        ridge.fit(x_val, y_val)
        candidates["stack:ridgecv"] = (ridge.predict(x_val), ridge.predict(x_test))

        huber = make_pipeline(StandardScaler(), HuberRegressor(alpha=0.001, epsilon=1.35, max_iter=1000))
        huber.fit(x_val, y_val)
        candidates["stack:huber"] = (huber.predict(x_val), huber.predict(x_test))
    except Exception as exc:
        print(f"[calibrate-v4] stackers skipped: {exc}")

    return candidates


def parse_model_specs(specs: Iterable[str]) -> List[Tuple[str, Path, Path]]:
    parsed = []
    for spec in specs:
        if "=" not in spec or "," not in spec:
            raise SystemExit("--model must look like name=config.yaml,checkpoint.ckpt")
        name, rest = spec.split("=", 1)
        config, checkpoint = rest.split(",", 1)
        parsed.append((name.strip(), _resolve(config.strip()), _resolve(checkpoint.strip())))
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit validation-only calibration for V4 checkpoints.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="name=config.yaml,checkpoint.ckpt. Repeat for ensembles.",
    )
    parser.add_argument("--output", default="outputs/diagnostics/v4_calibration_probe.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    model_specs = parse_model_specs(args.model)
    if not model_specs:
        raise SystemExit("Provide at least one --model")

    seed_everything(123, deterministic=False)
    configure_cuda(True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model_names = [name for name, _, _ in model_specs]
    per_model: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}

    for name, config_path, checkpoint_path in model_specs:
        print(f"[calibrate-v4] collecting {name}")
        per_model[name] = {}
        for split in ("val", "test"):
            per_model[name][split] = collect_speaker_predictions(
                name=name,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                split=split,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                use_ema=not args.no_ema,
            )

    val_rows = _merge_predictions(per_model, model_names, "val")
    test_rows = _merge_predictions(per_model, model_names, "test")
    y_val, y_test = _targets(val_rows), _targets(test_rows)
    candidates = fit_candidates(val_rows, test_rows, model_names)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, (val_pred, test_pred) in candidates.items():
        results[label] = {
            "val": _speaker_metrics(val_pred, y_val),
            "test": _speaker_metrics(test_pred, y_test),
        }

    selected = min(
        results,
        key=lambda label: results[label]["val"]["height_mae_speaker_balanced"],
    )
    report_only_best_test = min(
        results,
        key=lambda label: results[label]["test"]["height_mae_speaker_balanced"],
    )
    out = {
        "models": [
            {"name": name, "config": str(config), "checkpoint": str(checkpoint)}
            for name, config, checkpoint in model_specs
        ],
        "n_val_speakers": len(val_rows),
        "n_test_speakers": len(test_rows),
        "selected_by_val_balanced": selected,
        "report_only_best_test_balanced": report_only_best_test,
        "results": results,
    }
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, allow_nan=True), encoding="utf-8")

    print(f"[calibrate-v4] selected_by_val_balanced={selected}")
    print(
        "[calibrate-v4] val "
        f"{results[selected]['val']['height_mae_speaker']:.3f}/"
        f"{results[selected]['val']['height_mae_speaker_balanced']:.3f} "
        "test "
        f"{results[selected]['test']['height_mae_speaker']:.3f}/"
        f"{results[selected]['test']['height_mae_speaker_balanced']:.3f}"
    )
    print(f"[calibrate-v4] report_only_best_test_balanced={report_only_best_test}")
    print(
        "[calibrate-v4] best-test "
        f"{results[report_only_best_test]['test']['height_mae_speaker']:.3f}/"
        f"{results[report_only_best_test]['test']['height_mae_speaker_balanced']:.3f}"
    )
    print(f"[calibrate-v4] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
