#!/usr/bin/env python
"""Validation-trained stack over saved speaker-level height prediction sources."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stack saved speaker-level predictions.")
    parser.add_argument("--source", action="append", default=[], help="label:path[:split] CSV source. split is optional.")
    parser.add_argument("--output-dir", default="outputs/diagnostics/prediction_source_stack")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _parse_source(spec: str) -> tuple[str, Path, str | None]:
    parts = spec.split(":", 2)
    if len(parts) < 2:
        raise ValueError(f"Bad --source spec {spec!r}; expected label:path[:split]")
    label, path = parts[0], parts[1]
    split = parts[2] if len(parts) == 3 and parts[2] else None
    return label, _resolve(path), split


def _height_bin(value: float) -> str:
    if value < 160.0:
        return "short"
    if value >= 175.0:
        return "tall"
    return "medium"


def _prediction_columns(row: Mapping[str, str]) -> List[str]:
    return [
        key
        for key in row.keys()
        if key.startswith("pred_") and not key.startswith("pred_std")
    ]


def _load_sources(specs: Sequence[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    by_split: Dict[str, Dict[str, Dict[str, Any]]] = {"val": {}, "test": {}}
    for spec in specs:
        label, path, forced_split = _parse_source(spec)
        rows = _read_csv(path)
        if not rows:
            continue
        pred_cols = _prediction_columns(rows[0])
        for row in rows:
            split = str(forced_split or row.get("split", "")).lower()
            if split not in by_split:
                continue
            speaker_id = str(row.get("speaker_id", "") or "").strip()
            if not speaker_id:
                continue
            entry = by_split[split].setdefault(
                speaker_id,
                {
                    "speaker_id": speaker_id,
                    "height_true": _safe_float(row.get("height_true", row.get("height_cm"))),
                    "gender": row.get("gender_label", row.get("gender", "")),
                    "source": row.get("source", ""),
                },
            )
            if not math.isfinite(float(entry.get("height_true", float("nan")))):
                entry["height_true"] = _safe_float(row.get("height_true", row.get("height_cm")))
            for col in pred_cols:
                value = _safe_float(row.get(col))
                if math.isfinite(value):
                    entry[f"{label}_{col}"] = value
            for col in ("clip_pred_mean", "pred_clip_mean", "pred_clip_median", "pred_embedding_mean"):
                value = _safe_float(row.get(col))
                if math.isfinite(value):
                    entry[f"{label}_{col}"] = value
    return by_split


def _feature_columns(val_rows: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]]) -> List[str]:
    common = None
    for rows in (val_rows, test_rows):
        keys = {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"speaker_id", "height_true", "gender", "source"}
            and math.isfinite(_safe_float(value))
        }
        common = keys if common is None else common & keys
    return sorted(common or [])


def _xy(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[_safe_float(row.get(col)) for col in columns] for row in rows], dtype=np.float32)
    y = np.asarray([_safe_float(row.get("height_true")) for row in rows], dtype=np.float32)
    return x, y


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return {"count": 0.0, "mae": float("nan"), "rmse": float("nan"), "median_ae": float("nan")}
    err = y_true[valid] - y_pred[valid]
    return {
        "count": float(valid.sum()),
        "mae": float(mean_absolute_error(y_true[valid], y_pred[valid])),
        "rmse": float(math.sqrt(mean_squared_error(y_true[valid], y_pred[valid]))),
        "median_ae": float(np.median(np.abs(err))),
        "pred_std": float(np.std(y_pred[valid])),
        "true_std": float(np.std(y_true[valid])),
    }


def _slice_metrics(rows: Sequence[Mapping[str, Any]], pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    y = np.asarray([_safe_float(row["height_true"]) for row in rows], dtype=np.float32)
    out: Dict[str, Dict[str, float]] = {}
    groups = {
        "heightbin": [_height_bin(float(row["height_true"])) for row in rows],
        "gender": [str(row.get("gender", "")).lower() for row in rows],
        "source": [str(row.get("source", "")).upper() for row in rows],
    }
    for group, labels in groups.items():
        for label in sorted(set(labels)):
            mask = np.asarray([item == label for item in labels], dtype=bool)
            out[f"{group}_{label}"] = _metrics(y[mask], pred[mask])
    return out


def _write_predictions(path: Path, rows: Sequence[Mapping[str, Any]], pred_map: Mapping[str, np.ndarray]) -> None:
    fields = ["speaker_id", "height_true", "gender", "source"] + [f"pred_{name}" for name in pred_map]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(rows):
            out = {key: row.get(key, "") for key in ("speaker_id", "height_true", "gender", "source")}
            for name, pred in pred_map.items():
                out[f"pred_{name}"] = f"{float(pred[idx]):.6f}"
            writer.writerow(out)


def main() -> int:
    args = parse_args()
    if not args.source:
        raise RuntimeError("Provide at least one --source label:path[:split]")
    loaded = _load_sources(args.source)
    val_rows = sorted(loaded["val"].values(), key=lambda row: row["speaker_id"])
    test_rows = sorted(loaded["test"].values(), key=lambda row: row["speaker_id"])
    columns = _feature_columns(val_rows, test_rows)
    if not columns:
        raise RuntimeError("No common prediction columns found across val/test.")
    x_val, y_val = _xy(val_rows, columns)
    x_test, y_test = _xy(test_rows, columns)
    print(f"[source-stack] val={len(val_rows)} test={len(test_rows)} features={len(columns)}")

    models = {
        "linear": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", LinearRegression())]),
        "ridge": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-3, 3, 13)))]),
        "huber": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler()), ("model", HuberRegressor(alpha=0.03, epsilon=1.35, max_iter=1000))]),
    }
    pred_map: Dict[str, np.ndarray] = {}
    results: List[Dict[str, Any]] = []
    for idx, col in enumerate(columns):
        pred = x_test[:, idx]
        results.append({"name": col, "fit": "raw", "val": _metrics(y_val, x_val[:, idx]), "test": _metrics(y_test, pred), "test_slices": _slice_metrics(test_rows, pred)})
    for name, model in models.items():
        model.fit(x_val, y_val)
        pred_val = model.predict(x_val).astype(np.float32)
        pred_test = model.predict(x_test).astype(np.float32)
        pred_map[name] = pred_test
        results.append({"name": name, "fit": "validation_only_stack", "val": _metrics(y_val, pred_val), "test": _metrics(y_test, pred_test), "test_slices": _slice_metrics(test_rows, pred_test)})
    results.sort(key=lambda row: (row["val"]["mae"], row["test"]["mae"]))
    best_val = results[0]
    best_test = min(results, key=lambda row: row["test"]["mae"])
    summary = {
        "sources": args.source,
        "val_speakers": len(val_rows),
        "test_speakers": len(test_rows),
        "feature_columns": columns,
        "selected_by_val": best_val,
        "best_test_report_only_do_not_select_on_this": best_test,
        "results": results,
    }
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    _write_predictions(out_dir / "test_predictions.csv", test_rows, pred_map)
    print(
        f"[source-stack] selected_by_val {best_val['name']} "
        f"val={best_val['val']['mae']:.3f} test={best_val['test']['mae']:.3f}"
    )
    print(
        f"[source-stack] best_test_report_only {best_test['name']} "
        f"test={best_test['test']['mae']:.3f} val={best_test['val']['mae']:.3f}"
    )
    print(f"[source-stack] wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
