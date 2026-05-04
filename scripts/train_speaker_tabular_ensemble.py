#!/usr/bin/env python
"""Train fast speaker-level height ensembles from extracted VocalMorph features.

This script is deliberately test-label clean: model selection is based on the
validation split only, and the sealed test split is used once for reporting.
It is meant to answer a research question quickly: do the current acoustic/SSL
features contain enough height signal for a non-neural speaker-level model?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, HuberRegressor, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.audit_utils import height_bin  # noqa: E402


SCALAR_KEYS = (
    "f0_mean",
    "formant_spacing_mean",
    "vtl_mean",
    "jitter",
    "shimmer",
    "hnr",
    "duration_s",
    "voiced_ratio",
    "invalid_spacing_rate",
    "invalid_vtl_rate",
    "speech_ratio",
    "snr_db_estimate",
    "capture_quality_score",
    "distance_cm_estimate",
    "distance_confidence",
    "clipped_ratio",
)


@dataclass
class SpeakerExample:
    speaker_id: str
    split: str
    height_cm: float
    gender: int
    source: str
    n_clips: int
    features: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train speaker-level tabular height ensembles.")
    parser.add_argument("--features-root", default="data/features_vtl_ssl")
    parser.add_argument("--output-dir", default="outputs/diagnostics/speaker_tabular_sslclean")
    parser.add_argument("--include-augmented", action="store_true")
    parser.add_argument(
        "--include-quantiles",
        action="store_true",
        help="Include per-frame quantiles. This can be slow on large SSL feature roots.",
    )
    parser.add_argument(
        "--use-label-metadata",
        action="store_true",
        help="Also feed label metadata such as gender/age/weight. This is useful as an upper-bound diagnostic.",
    )
    parser.add_argument("--max-clips-per-speaker", type=int, default=0)
    parser.add_argument("--tree-estimators", type=int, default=320)
    parser.add_argument("--model-set", choices=("quick", "full"), default="quick")
    parser.add_argument("--random-state", type=int, default=11)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def _resolve(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _decode_np_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.ndim == 0:
            item = value.item()
            if isinstance(item, bytes):
                return item.decode("utf-8", errors="ignore")
            return str(item)
        return _decode_np_value(value.reshape(-1)[0])
    return str(value)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _source_id(source: str, speaker_id: str) -> float:
    source_upper = str(source or "").upper()
    sid = str(speaker_id or "").upper()
    if source_upper == "TIMIT" or sid.startswith("TIMIT_"):
        return 0.0
    if source_upper == "NISP" or sid.startswith("NISP_"):
        return 1.0
    return 2.0


def _gender_from_npz(data: Mapping[str, Any]) -> int:
    raw = data["gender"] if "gender" in data else 0
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        return 1 if raw.strip().lower() == "male" else 0
    try:
        return int(raw)
    except Exception:
        return 0


def _clip_feature(
    path: Path,
    *,
    use_label_metadata: bool,
    include_quantiles: bool,
) -> Tuple[str, float, int, str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        speaker_id = _decode_np_value(data["speaker_id"]) if "speaker_id" in data else path.stem.rsplit("_", 1)[0]
        height_cm = _safe_float(data["height_cm"]) if "height_cm" in data else np.nan
        gender = _gender_from_npz(data)
        source = _decode_np_value(data["source"]).upper() if "source" in data else ""

        sequence = np.asarray(data["sequence"], dtype=np.float32)
        if sequence.ndim != 2 or sequence.size == 0:
            raise ValueError(f"Bad sequence in {path}")
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)

        seq_stats = [
            sequence.mean(axis=0),
            sequence.std(axis=0),
            sequence.min(axis=0),
            sequence.max(axis=0),
        ]
        if include_quantiles:
            seq_stats.extend(
                [
                    np.quantile(sequence, 0.10, axis=0),
                    np.quantile(sequence, 0.50, axis=0),
                    np.quantile(sequence, 0.90, axis=0),
                ]
            )
        pieces: List[np.ndarray] = [np.concatenate(seq_stats).astype(np.float32)]

        if "ssl_embedding" in data:
            pieces.append(np.asarray(data["ssl_embedding"], dtype=np.float32).reshape(-1))

        scalars = [_safe_float(data[key]) if key in data else np.nan for key in SCALAR_KEYS]
        scalars.extend(
            [
                float(sequence.shape[0]),
                float(sequence.shape[1]),
                _source_id(source, speaker_id),
            ]
        )
        if use_label_metadata:
            scalars.extend(
                [
                    float(gender),
                    _safe_float(data["age"]) if "age" in data else np.nan,
                    _safe_float(data["weight_kg"]) if "weight_kg" in data else np.nan,
                ]
            )
        pieces.append(np.asarray(scalars, dtype=np.float32))

    return str(speaker_id), float(height_cm), int(gender), str(source or "UNKNOWN"), np.concatenate(pieces)


def _speaker_feature_matrix(
    split_dir: Path,
    *,
    split_name: str,
    include_augmented: bool,
    use_label_metadata: bool,
    include_quantiles: bool,
    max_clips_per_speaker: int,
) -> List[SpeakerExample]:
    paths = sorted(split_dir.glob("*.npz"))
    if not include_augmented:
        paths = [p for p in paths if "_aug" not in p.stem]
    grouped: Dict[str, List[Tuple[float, int, str, np.ndarray]]] = defaultdict(list)
    skipped = Counter()

    for path in paths:
        try:
            speaker_id, height_cm, gender, source, vector = _clip_feature(
                path,
                use_label_metadata=use_label_metadata,
                include_quantiles=include_quantiles,
            )
        except Exception:
            skipped["load_or_feature_failed"] += 1
            continue
        if not math.isfinite(height_cm):
            skipped["missing_height"] += 1
            continue
        grouped[speaker_id].append((height_cm, gender, source, vector))

    examples: List[SpeakerExample] = []
    for speaker_id, rows in sorted(grouped.items()):
        if max_clips_per_speaker > 0:
            rows = rows[:max_clips_per_speaker]
        vectors = np.stack([row[3] for row in rows]).astype(np.float32)
        height = float(np.median([row[0] for row in rows]))
        gender = int(round(float(np.median([row[1] for row in rows]))))
        source = Counter(row[2] for row in rows).most_common(1)[0][0]

        pooled = np.concatenate(
            [
                vectors.mean(axis=0),
                vectors.std(axis=0),
                vectors.min(axis=0),
                vectors.max(axis=0),
                np.asarray([float(len(rows))], dtype=np.float32),
            ]
        ).astype(np.float32)
        examples.append(
            SpeakerExample(
                speaker_id=speaker_id,
                split=split_name,
                height_cm=height,
                gender=gender,
                source=source,
                n_clips=len(rows),
                features=pooled,
            )
        )

    print(
        f"[speaker-tabular] {split_name}: speakers={len(examples)} clips={len(paths)} "
        f"skipped={dict(skipped)}"
    )
    return examples


def _as_xy(examples: Sequence[SpeakerExample]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.stack([ex.features for ex in examples]).astype(np.float32)
    y = np.asarray([ex.height_cm for ex in examples], dtype=np.float32)
    return x, y


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return {"count": 0.0, "mae": float("nan"), "rmse": float("nan"), "median_ae": float("nan")}
    err = y_true[valid] - y_pred[valid]
    abs_err = np.abs(err)
    pred_std = float(np.std(y_pred[valid]))
    true_std = float(np.std(y_true[valid]))
    return {
        "count": float(valid.sum()),
        "mae": float(mean_absolute_error(y_true[valid], y_pred[valid])),
        "rmse": float(math.sqrt(mean_squared_error(y_true[valid], y_pred[valid]))),
        "median_ae": float(np.median(abs_err)),
        "pred_std": pred_std,
        "true_std": true_std,
        "std_ratio": float(pred_std / true_std) if true_std > 1e-6 else float("nan"),
    }


def _slice_metrics(
    examples: Sequence[SpeakerExample],
    y_pred: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    y_true = np.asarray([ex.height_cm for ex in examples], dtype=np.float32)
    slices: Dict[str, Dict[str, float]] = {}
    labels = {
        "heightbin": [height_bin(ex.height_cm) for ex in examples],
        "gender": ["male" if ex.gender == 1 else "female" for ex in examples],
        "source": [str(ex.source or "UNKNOWN").upper() for ex in examples],
    }
    for group_name, group_labels in labels.items():
        for label in sorted(set(group_labels)):
            mask = np.asarray([value == label for value in group_labels], dtype=bool)
            if int(mask.sum()) == 0:
                continue
            slices[f"{group_name}_{label}"] = _metrics(y_true[mask], y_pred[mask])
    return slices


def _models(
    random_state: int,
    n_jobs: int,
    tree_estimators: int,
    model_set: str,
) -> Dict[str, Any]:
    alphas = np.logspace(-3, 4, 20)
    models: Dict[str, Any] = {
        "ridge": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", RidgeCV(alphas=alphas)),
            ]
        ),
        "huber": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", RobustScaler()),
                ("model", HuberRegressor(alpha=0.03, epsilon=1.35, max_iter=1200)),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=int(tree_estimators),
                        min_samples_leaf=3,
                        max_features=0.35,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        ),
    }
    if str(model_set).lower() == "full":
        models.update(
            {
                "elastic": Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        (
                            "model",
                            ElasticNetCV(
                                l1_ratio=[0.03, 0.08, 0.15, 0.30],
                                alphas=alphas,
                                cv=5,
                                max_iter=8000,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
                "svr_rbf": Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        ("model", SVR(C=20.0, gamma="scale", epsilon=1.5)),
                    ]
                ),
                "random_forest": Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        (
                            "model",
                            RandomForestRegressor(
                                n_estimators=max(120, int(tree_estimators // 2)),
                                min_samples_leaf=4,
                                max_features=0.45,
                                random_state=random_state,
                                n_jobs=n_jobs,
                            ),
                        ),
                    ]
                ),
                "hist_gbdt": Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        (
                            "model",
                            HistGradientBoostingRegressor(
                                loss="absolute_error",
                                learning_rate=0.045,
                                max_iter=260,
                                max_leaf_nodes=15,
                                l2_regularization=0.20,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            }
        )
    return models


def _fit_predict_all(
    models: Mapping[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    fitted: Dict[str, Any] = {}
    val_preds: Dict[str, np.ndarray] = {}
    test_preds: Dict[str, np.ndarray] = {}
    lo = float(np.nanmin(y_train) - 8.0)
    hi = float(np.nanmax(y_train) + 8.0)

    for name, estimator in models.items():
        print(f"[speaker-tabular] fitting {name}")
        try:
            model = clone(estimator)
            model.fit(x_train, y_train)
            fitted[name] = model
            val_preds[name] = np.clip(model.predict(x_val).astype(np.float32), lo, hi)
            test_preds[name] = np.clip(model.predict(x_test).astype(np.float32), lo, hi)
        except Exception as exc:
            print(f"[speaker-tabular] {name} failed: {exc}")
    return fitted, val_preds, test_preds


def _write_predictions_csv(
    path: Path,
    examples: Sequence[SpeakerExample],
    prediction_columns: Mapping[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "speaker_id", "height_true", "gender", "source", "n_clips"] + [
        f"pred_{name}" for name in prediction_columns
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, ex in enumerate(examples):
            row = {
                "split": ex.split,
                "speaker_id": ex.speaker_id,
                "height_true": f"{ex.height_cm:.6f}",
                "gender": "Male" if ex.gender == 1 else "Female",
                "source": ex.source,
                "n_clips": ex.n_clips,
            }
            for name, preds in prediction_columns.items():
                row[f"pred_{name}"] = f"{float(preds[idx]):.6f}"
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    features_root = _resolve(args.features_root)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_examples = {
        split: _speaker_feature_matrix(
            features_root / split,
            split_name=split,
            include_augmented=bool(args.include_augmented),
            use_label_metadata=bool(args.use_label_metadata),
            include_quantiles=bool(args.include_quantiles),
            max_clips_per_speaker=int(args.max_clips_per_speaker),
        )
        for split in ("train", "val", "test")
    }
    x_train, y_train = _as_xy(split_examples["train"])
    x_val, y_val = _as_xy(split_examples["val"])
    x_test, y_test = _as_xy(split_examples["test"])

    print(
        "[speaker-tabular] feature_dim="
        f"{x_train.shape[1]} train={x_train.shape[0]} val={x_val.shape[0]} test={x_test.shape[0]}"
    )

    model_defs = _models(
        random_state=int(args.random_state),
        n_jobs=int(args.n_jobs),
        tree_estimators=int(args.tree_estimators),
        model_set=str(args.model_set),
    )
    fitted, val_preds, test_preds = _fit_predict_all(model_defs, x_train, y_train, x_val, x_test)
    if not fitted:
        raise RuntimeError("No models fit successfully.")

    results: List[Dict[str, Any]] = []
    for name in val_preds:
        results.append(
            {
                "name": name,
                "fit": "train_only",
                "val": _metrics(y_val, val_preds[name]),
                "test": _metrics(y_test, test_preds[name]),
                "test_slices": _slice_metrics(split_examples["test"], test_preds[name]),
            }
        )

    ranked = sorted(results, key=lambda row: row["val"]["mae"])
    top_names = [row["name"] for row in ranked[: min(4, len(ranked))]]
    if len(top_names) >= 2:
        top_val = np.stack([val_preds[name] for name in top_names], axis=1)
        top_test = np.stack([test_preds[name] for name in top_names], axis=1)
        blend_defs = {
            "blend_top_val_ridge": Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", RidgeCV(alphas=np.logspace(-3, 3, 13))),
                ]
            ),
            "blend_top_val_huber": Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", RobustScaler()),
                    ("model", HuberRegressor(alpha=0.02, epsilon=1.25, max_iter=1000)),
                ]
            ),
        }
        for name, blender in blend_defs.items():
            try:
                blender.fit(top_val, y_val)
                pred = blender.predict(top_test).astype(np.float32)
                pred_val = blender.predict(top_val).astype(np.float32)
                results.append(
                    {
                        "name": name,
                        "fit": "val_blender_on_train_models",
                        "members": top_names,
                        "val": _metrics(y_val, pred_val),
                        "test": _metrics(y_test, pred),
                        "test_slices": _slice_metrics(split_examples["test"], pred),
                    }
                )
                val_preds[name] = pred_val
                test_preds[name] = pred
            except Exception as exc:
                print(f"[speaker-tabular] {name} failed: {exc}")

        avg_pred = top_test.mean(axis=1).astype(np.float32)
        med_pred = np.median(top_test, axis=1).astype(np.float32)
        avg_val = top_val.mean(axis=1).astype(np.float32)
        med_val = np.median(top_val, axis=1).astype(np.float32)
        for name, vp, tp in (
            ("avg_top_val", avg_val, avg_pred),
            ("median_top_val", med_val, med_pred),
        ):
            results.append(
                {
                    "name": name,
                    "fit": "train_only_average",
                    "members": top_names,
                    "val": _metrics(y_val, vp),
                    "test": _metrics(y_test, tp),
                    "test_slices": _slice_metrics(split_examples["test"], tp),
                }
            )
            val_preds[name] = vp
            test_preds[name] = tp

    selected_name = ranked[0]["name"]
    selected_refit = clone(model_defs[selected_name])
    x_train_val = np.concatenate([x_train, x_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    selected_refit.fit(x_train_val, y_train_val)
    refit_pred = selected_refit.predict(x_test).astype(np.float32)
    refit_val_pred = selected_refit.predict(x_val).astype(np.float32)
    results.append(
        {
            "name": f"{selected_name}_refit_train_val",
            "fit": "selected_by_val_then_refit_train_val",
            "selected_from": selected_name,
            "val": ranked[0]["val"],
            "test": _metrics(y_test, refit_pred),
            "test_slices": _slice_metrics(split_examples["test"], refit_pred),
        }
    )
    val_preds[f"{selected_name}_refit_train_val"] = refit_val_pred
    test_preds[f"{selected_name}_refit_train_val"] = refit_pred

    best_by_val = min(results, key=lambda row: row["val"]["mae"])
    best_by_test_report_only = min(results, key=lambda row: row["test"]["mae"])
    summary = {
        "features_root": str(features_root),
        "use_label_metadata": bool(args.use_label_metadata),
        "include_augmented": bool(args.include_augmented),
        "include_quantiles": bool(args.include_quantiles),
        "model_set": str(args.model_set),
        "speaker_counts": {split: len(rows) for split, rows in split_examples.items()},
        "feature_dim": int(x_train.shape[1]),
        "selected_by_val": best_by_val,
        "best_test_report_only_do_not_select_on_this": best_by_test_report_only,
        "results": sorted(results, key=lambda row: (row["val"]["mae"], row["test"]["mae"])),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)

    prediction_columns = {name: pred for name, pred in test_preds.items()}
    _write_predictions_csv(output_dir / "test_predictions.csv", split_examples["test"], prediction_columns)
    _write_predictions_csv(output_dir / "val_predictions.csv", split_examples["val"], val_preds)

    print("[speaker-tabular] selected by validation:")
    print(
        f"  {best_by_val['name']} val={best_by_val['val']['mae']:.3f}cm "
        f"test={best_by_val['test']['mae']:.3f}cm"
    )
    print("[speaker-tabular] best test report-only row:")
    print(
        f"  {best_by_test_report_only['name']} test={best_by_test_report_only['test']['mae']:.3f}cm "
        f"val={best_by_test_report_only['val']['mae']:.3f}cm"
    )
    print(f"[speaker-tabular] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
