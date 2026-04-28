"""Direct speaker-level height research pipeline.

This module is intentionally separate from the clip-level neural trainers.
It builds one row per speaker from audited feature NPZ files, trains a small
model zoo, tunes an ensemble on validation speakers, and reports the exact
speaker-level slices that matter for real-world height use.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.utils.audit_utils import decode_np_value, height_bin, safe_float

SCALAR_KEYS: Tuple[str, ...] = (
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
    "gender",
    "source_id",
    "quality_ok",
)

SEQUENCE_STATS: Tuple[str, ...] = (
    "mean",
    "std",
    "median",
    "p10",
    "p25",
    "p75",
    "p90",
    "iqr",
    "min",
    "max",
)

AGGREGATE_STATS: Tuple[str, ...] = ("mean", "median", "std", "p10", "p90")
PREDICTION_COLUMNS: Tuple[str, ...] = ("ensemble", "calibrated", "calibrated_edge")


@dataclass(frozen=True)
class SpeakerFeatureTable:
    split: str
    x: np.ndarray
    y: np.ndarray
    speaker_ids: List[str]
    feature_names: List[str]
    metadata: List[Dict[str, Any]]


@dataclass
class HeightResearchModel:
    feature_names: List[str]
    models: Dict[str, Pipeline]
    weights: Dict[str, float]
    calibrator: Optional[LinearRegression]
    edge_offsets: Dict[str, float]
    val_metrics: Dict[str, float]

    def predict_base(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            name: np.asarray(model.predict(x), dtype=np.float32)
            for name, model in self.models.items()
        }

    def predict(self, x: np.ndarray, *, mode: str = "calibrated_edge") -> np.ndarray:
        base_preds = self.predict_base(x)
        ensemble = weighted_predictions(base_preds, self.weights)
        if mode == "ensemble":
            return ensemble
        calibrated = apply_calibrator(ensemble, self.calibrator)
        if mode == "calibrated":
            return calibrated
        if mode != "calibrated_edge":
            raise ValueError(f"Unknown prediction mode: {mode}")
        return apply_edge_offsets(calibrated, self.edge_offsets)


def _resolve(root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


def _feature_files(split_dir: str) -> List[str]:
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Missing feature split directory: {split_dir}. "
            "Restore or rebuild data/features_audited before running this experiment."
        )
    files = sorted(
        os.path.join(split_dir, name)
        for name in os.listdir(split_dir)
        if name.lower().endswith(".npz")
    )
    if not files:
        raise FileNotFoundError(f"No .npz feature files found in {split_dir}")
    return files


def _source_to_id(value: Any) -> float:
    source = decode_np_value(value).strip().upper()
    if source == "NISP":
        return 1.0
    if source == "TIMIT":
        return 0.0
    return float("nan")


def _safe_np_scalar(data: Mapping[str, Any], key: str) -> float:
    if key == "source_id":
        if "source_id" in data:
            return safe_float(data["source_id"])
        if "source" in data:
            return _source_to_id(data["source"])
        return float("nan")
    if key == "quality_ok":
        if key not in data:
            return float("nan")
        raw = data[key]
        try:
            return 1.0 if bool(np.asarray(raw).item()) else 0.0
        except Exception:
            return float("nan")
    if key not in data:
        return float("nan")
    return safe_float(data[key])


def _nan_percentile(sequence: np.ndarray, q: float) -> np.ndarray:
    with np.errstate(all="ignore"):
        values = np.nanpercentile(sequence, q, axis=0)
    return np.asarray(values, dtype=np.float32)


def _sequence_features(sequence: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    if sequence.ndim != 2:
        raise ValueError(f"Expected sequence with shape (frames, features), got {sequence.shape}")
    dim = int(sequence.shape[1])
    with np.errstate(all="ignore"):
        stat_values = {
            "mean": np.nanmean(sequence, axis=0),
            "std": np.nanstd(sequence, axis=0),
            "median": np.nanmedian(sequence, axis=0),
            "p10": _nan_percentile(sequence, 10),
            "p25": _nan_percentile(sequence, 25),
            "p75": _nan_percentile(sequence, 75),
            "p90": _nan_percentile(sequence, 90),
            "min": np.nanmin(sequence, axis=0),
            "max": np.nanmax(sequence, axis=0),
        }
    stat_values["iqr"] = stat_values["p75"] - stat_values["p25"]
    vectors: List[np.ndarray] = []
    names: List[str] = []
    for stat in SEQUENCE_STATS:
        vectors.append(np.asarray(stat_values[stat], dtype=np.float32))
        names.extend(f"seq_{stat}_{idx:03d}" for idx in range(dim))
    return np.concatenate(vectors, axis=0), names


def clip_feature_vector(data: Mapping[str, Any]) -> Tuple[np.ndarray, List[str]]:
    sequence = np.asarray(data["sequence"], dtype=np.float32)
    sequence_vector, names = _sequence_features(sequence)
    scalars = np.asarray([_safe_np_scalar(data, key) for key in SCALAR_KEYS], dtype=np.float32)
    frame_features = np.asarray(
        [
            float(sequence.shape[0]),
            float(np.isfinite(sequence).mean()) if sequence.size else float("nan"),
        ],
        dtype=np.float32,
    )
    feature_vector = np.concatenate([sequence_vector, scalars, frame_features], axis=0)
    feature_names = names + list(SCALAR_KEYS) + ["n_frames", "finite_ratio"]
    return feature_vector.astype(np.float32, copy=False), feature_names


def _clip_quality_weight(data: Mapping[str, Any]) -> float:
    capture = _safe_np_scalar(data, "capture_quality_score")
    speech = _safe_np_scalar(data, "speech_ratio")
    clipped = _safe_np_scalar(data, "clipped_ratio")
    weight = 1.0
    if np.isfinite(capture):
        weight *= float(np.clip(capture, 0.10, 1.25))
    if np.isfinite(speech):
        weight *= float(np.clip(speech, 0.20, 1.10))
    if np.isfinite(clipped):
        weight *= float(np.clip(1.0 - 4.0 * clipped, 0.20, 1.0))
    return float(np.clip(weight, 0.05, 1.50))


def _aggregate_clip_vectors(vectors: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    with np.errstate(all="ignore"):
        weighted_mean = np.average(vectors, axis=0, weights=weights)
        stat_values = {
            "mean": weighted_mean,
            "median": np.nanmedian(vectors, axis=0),
            "std": np.nanstd(vectors, axis=0),
            "p10": np.nanpercentile(vectors, 10, axis=0),
            "p90": np.nanpercentile(vectors, 90, axis=0),
        }
    parts: List[np.ndarray] = []
    names: List[str] = []
    for stat in AGGREGATE_STATS:
        parts.append(np.asarray(stat_values[stat], dtype=np.float32))
        names.extend(f"speaker_{stat}__{{clip_feature}}" for _ in range(vectors.shape[1]))
    return np.concatenate(parts, axis=0), names


def _speaker_feature_names(clip_names: Sequence[str]) -> List[str]:
    names: List[str] = []
    for stat in AGGREGATE_STATS:
        names.extend(f"speaker_{stat}__{name}" for name in clip_names)
    names.extend(
        [
            "speaker_n_clips",
            "speaker_clip_weight_mean",
            "speaker_clip_weight_std",
            "speaker_clip_weight_min",
            "speaker_clip_weight_max",
        ]
    )
    return names


def load_speaker_split(
    features_dir: str,
    split: str,
    *,
    expected_feature_names: Optional[Sequence[str]] = None,
) -> SpeakerFeatureTable:
    split_dir = os.path.join(features_dir, split)
    speakers: Dict[str, Dict[str, Any]] = {}
    clip_names: Optional[List[str]] = None
    for path in _feature_files(split_dir):
        with np.load(path, allow_pickle=True) as data:
            speaker_id = decode_np_value(data["speaker_id"]).strip()
            if not speaker_id:
                continue
            vector, names = clip_feature_vector(data)
            if clip_names is None:
                clip_names = names
            elif names != clip_names:
                raise ValueError(f"Feature dimensionality changed inside {split}: {path}")
            height_cm = _safe_np_scalar(data, "height_cm")
            if not np.isfinite(height_cm):
                continue
            entry = speakers.setdefault(
                speaker_id,
                {
                    "height_cm": height_cm,
                    "vectors": [],
                    "weights": [],
                    "gender": _safe_np_scalar(data, "gender"),
                    "source_id": _safe_np_scalar(data, "source_id"),
                },
            )
            entry["vectors"].append(vector)
            entry["weights"].append(_clip_quality_weight(data))

    if not speakers or clip_names is None:
        raise RuntimeError(f"No valid speaker examples found in {split_dir}")

    feature_names = _speaker_feature_names(clip_names)
    if expected_feature_names is not None and list(expected_feature_names) != feature_names:
        raise ValueError(f"Feature names for split {split} do not match the training split.")

    rows: List[np.ndarray] = []
    targets: List[float] = []
    speaker_ids: List[str] = []
    metadata: List[Dict[str, Any]] = []
    for speaker_id in sorted(speakers):
        entry = speakers[speaker_id]
        vectors = np.stack(entry["vectors"], axis=0).astype(np.float32, copy=False)
        weights = np.asarray(entry["weights"], dtype=np.float32)
        aggregate, _ = _aggregate_clip_vectors(vectors, weights)
        extras = np.asarray(
            [
                float(vectors.shape[0]),
                float(np.nanmean(weights)),
                float(np.nanstd(weights)),
                float(np.nanmin(weights)),
                float(np.nanmax(weights)),
            ],
            dtype=np.float32,
        )
        rows.append(np.concatenate([aggregate, extras], axis=0))
        targets.append(float(entry["height_cm"]))
        speaker_ids.append(speaker_id)
        metadata.append(
            {
                "speaker_id": speaker_id,
                "height_cm": float(entry["height_cm"]),
                "height_bin": height_bin(float(entry["height_cm"])),
                "n_clips": int(vectors.shape[0]),
                "gender": float(entry.get("gender", float("nan"))),
                "source_id": float(entry.get("source_id", float("nan"))),
            }
        )

    return SpeakerFeatureTable(
        split=split,
        x=np.stack(rows, axis=0).astype(np.float32, copy=False),
        y=np.asarray(targets, dtype=np.float32),
        speaker_ids=speaker_ids,
        feature_names=feature_names,
        metadata=metadata,
    )


def load_research_tables(features_dir: str) -> Dict[str, SpeakerFeatureTable]:
    train = load_speaker_split(features_dir, "train")
    val = load_speaker_split(features_dir, "val", expected_feature_names=train.feature_names)
    test = load_speaker_split(features_dir, "test", expected_feature_names=train.feature_names)
    return {"train": train, "val": val, "test": test}


def _feature_selector_k(n_samples: int, n_features: int) -> int:
    if n_features <= 64:
        return n_features
    return int(min(n_features, max(64, min(768, n_samples * 2))))


def _linear_pipeline(estimator: Any, *, k: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(10.0, 90.0))),
            ("select", SelectKBest(score_func=f_regression, k=k)),
            ("model", estimator),
        ]
    )


def _tree_pipeline(estimator: Any, *, k: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("select", SelectKBest(score_func=f_regression, k=k)),
            ("model", estimator),
        ]
    )


def build_model_zoo(seed: int, *, n_samples: int, n_features: int) -> Dict[str, Pipeline]:
    k = _feature_selector_k(n_samples, n_features)
    alphas = np.logspace(-3, 3, 13)
    return {
        "ridge": _linear_pipeline(RidgeCV(alphas=alphas), k=k),
        "huber": _linear_pipeline(HuberRegressor(alpha=0.003, epsilon=1.35, max_iter=1000), k=k),
        "elasticnet": _linear_pipeline(
            ElasticNetCV(
                alphas=np.logspace(-4, 1, 18),
                l1_ratio=(0.05, 0.15, 0.35, 0.65, 0.90),
                cv=5,
                max_iter=8000,
                random_state=seed,
            ),
            k=k,
        ),
        "extra_trees": _tree_pipeline(
            ExtraTreesRegressor(
                n_estimators=500,
                max_features=0.35,
                min_samples_leaf=3,
                bootstrap=True,
                random_state=seed,
                n_jobs=-1,
            ),
            k=k,
        ),
        "random_forest": _tree_pipeline(
            RandomForestRegressor(
                n_estimators=350,
                max_features=0.45,
                min_samples_leaf=4,
                bootstrap=True,
                random_state=seed,
                n_jobs=-1,
            ),
            k=k,
        ),
        "hist_gbr_l1": _tree_pipeline(
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.035,
                max_iter=450,
                max_leaf_nodes=15,
                l2_regularization=0.08,
                random_state=seed,
            ),
            k=k,
        ),
        "grad_boost_l1": _tree_pipeline(
            GradientBoostingRegressor(
                loss="absolute_error",
                n_estimators=350,
                learning_rate=0.035,
                max_depth=2,
                subsample=0.85,
                random_state=seed,
            ),
            k=k,
        ),
    }


def speaker_sample_weights(y: np.ndarray) -> np.ndarray:
    weights = np.ones_like(y, dtype=np.float32)
    weights[y < 160.0] *= 1.80
    weights[y >= 175.0] *= 1.35
    return weights


def _fit_pipeline(model: Pipeline, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> Pipeline:
    fitted = clone(model)
    try:
        fitted.fit(x, y, model__sample_weight=sample_weight)
    except TypeError:
        fitted.fit(x, y)
    return fitted


def weighted_predictions(preds_by_model: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    total = np.zeros_like(next(iter(preds_by_model.values())), dtype=np.float32)
    for name, weight in weights.items():
        total += float(weight) * np.asarray(preds_by_model[name], dtype=np.float32)
    return total


def tune_ensemble_weights(
    preds_by_model: Mapping[str, np.ndarray],
    y_true: np.ndarray,
    *,
    seed: int,
    trials: int = 5000,
) -> Dict[str, float]:
    names = list(preds_by_model)
    matrix = np.stack([np.asarray(preds_by_model[name], dtype=np.float32) for name in names], axis=1)
    maes = np.mean(np.abs(matrix - y_true.reshape(-1, 1)), axis=0)
    scale = max(0.20, float(np.std(maes)))
    base = np.exp(-(maes - float(np.min(maes))) / scale)
    base = base / np.sum(base)
    best_weights = base.astype(np.float64)
    best_mae = float(np.mean(np.abs(matrix @ best_weights - y_true)))
    rng = np.random.default_rng(seed)

    for idx in range(len(names)):
        candidate = np.zeros(len(names), dtype=np.float64)
        candidate[idx] = 1.0
        mae = float(np.mean(np.abs(matrix @ candidate - y_true)))
        if mae < best_mae:
            best_mae = mae
            best_weights = candidate

    concentration = np.clip(base * 30.0 + 0.25, 0.25, None)
    for _ in range(int(trials)):
        candidate = rng.dirichlet(concentration)
        candidate = 0.65 * candidate + 0.35 * base
        candidate = candidate / np.sum(candidate)
        mae = float(np.mean(np.abs(matrix @ candidate - y_true)))
        if mae < best_mae:
            best_mae = mae
            best_weights = candidate

    cleaned = {
        name: float(weight)
        for name, weight in zip(names, best_weights)
        if float(weight) >= 1e-4
    }
    normalizer = sum(cleaned.values())
    return {name: float(weight / normalizer) for name, weight in cleaned.items()}


def apply_calibrator(pred: np.ndarray, calibrator: Optional[LinearRegression]) -> np.ndarray:
    if calibrator is None:
        return np.asarray(pred, dtype=np.float32)
    return np.asarray(calibrator.predict(np.asarray(pred).reshape(-1, 1)), dtype=np.float32)


def _predicted_height_bin(value: float) -> str:
    if value < 160.0:
        return "short"
    if value < 175.0:
        return "medium"
    return "tall"


def fit_edge_offsets(pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    offsets: Dict[str, float] = {}
    residual = y_true - pred
    for label in ("short", "medium", "tall"):
        mask = np.asarray([_predicted_height_bin(float(value)) == label for value in pred], dtype=bool)
        if not np.any(mask):
            offsets[label] = 0.0
            continue
        shrink = float(mask.sum() / (mask.sum() + 25.0))
        offsets[label] = float(np.clip(np.median(residual[mask]) * shrink, -4.0, 4.0))
    return offsets


def apply_edge_offsets(pred: np.ndarray, offsets: Mapping[str, float]) -> np.ndarray:
    adjusted = np.asarray(pred, dtype=np.float32).copy()
    for idx, value in enumerate(adjusted):
        adjusted[idx] += float(offsets.get(_predicted_height_bin(float(value)), 0.0))
    return adjusted


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, metadata: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    err = np.asarray(y_pred, dtype=np.float32) - np.asarray(y_true, dtype=np.float32)
    abs_err = np.abs(err)
    metrics: Dict[str, float] = {
        "mae": float(np.mean(abs_err)),
        "rmse": float(math.sqrt(float(np.mean(err**2)))),
        "median_ae": float(np.median(abs_err)),
        "p90_ae": float(np.percentile(abs_err, 90)),
        "max_ae": float(np.max(abs_err)),
        "bias": float(np.mean(err)),
        "within_2cm": float(np.mean(abs_err <= 2.0)),
        "within_3cm": float(np.mean(abs_err <= 3.0)),
        "within_5cm": float(np.mean(abs_err <= 5.0)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "n_speakers": float(len(y_true)),
    }
    for label in ("short", "medium", "tall"):
        mask = np.asarray([row.get("height_bin") == label for row in metadata], dtype=bool)
        if np.any(mask):
            metrics[f"{label}_mae"] = float(np.mean(abs_err[mask]))
            metrics[f"{label}_n"] = float(mask.sum())
    edge_values = [metrics[key] for key in ("short_mae", "tall_mae") if key in metrics]
    if edge_values:
        metrics["edge_mae"] = float(np.mean(edge_values))
        metrics["edge_worst_mae"] = float(np.max(edge_values))
    return metrics


def prediction_rows(
    table: SpeakerFeatureTable,
    preds_by_column: Mapping[str, np.ndarray],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, speaker_id in enumerate(table.speaker_ids):
        row = dict(table.metadata[idx])
        row["speaker_id"] = speaker_id
        row["height_cm"] = float(table.y[idx])
        for name, preds in preds_by_column.items():
            pred = float(preds[idx])
            row[f"{name}_pred_cm"] = pred
            row[f"{name}_abs_error_cm"] = abs(pred - float(table.y[idx]))
        rows.append(row)
    return rows


def write_prediction_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    final = payload["final_test"]["calibrated_edge"]
    val = payload["final_val"]["calibrated_edge"]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# VoxPhysica 3 cm Research Run\n\n")
        handle.write("## Headline\n")
        handle.write(f"- Validation speaker MAE: `{val['mae']:.3f} cm`\n")
        handle.write(f"- Test speaker MAE: `{final['mae']:.3f} cm`\n")
        handle.write(f"- Test short speaker MAE: `{final.get('short_mae', float('nan')):.3f} cm`\n")
        handle.write(f"- Test tall speaker MAE: `{final.get('tall_mae', float('nan')):.3f} cm`\n")
        handle.write(f"- Test within 3 cm: `{final['within_3cm'] * 100:.1f}%`\n\n")
        handle.write("## Model Weights\n")
        for name, weight in sorted(payload["ensemble_weights"].items(), key=lambda item: item[1], reverse=True):
            handle.write(f"- `{name}`: `{weight:.4f}`\n")
        handle.write("\n## Notes\n")
        handle.write("- This is a direct speaker-level challenger, separate from the clip-level neural trainer.\n")
        handle.write("- Validation speakers tune only the ensemble/calibration layer; test speakers stay held out.\n")
        handle.write("- `target_met` is true only when the calibrated-edge test MAE is <= the configured target.\n")


def run_research_experiment(
    *,
    features_dir: str,
    output_dir: str,
    seed: int = 11,
    target_mae_cm: float = 3.0,
    model_names: Optional[Sequence[str]] = None,
    ensemble_trials: int = 5000,
    save_model: bool = True,
) -> Dict[str, Any]:
    tables = load_research_tables(features_dir)
    train = tables["train"]
    val = tables["val"]
    test = tables["test"]
    zoo = build_model_zoo(seed, n_samples=train.x.shape[0], n_features=train.x.shape[1])
    if model_names:
        selected = set(model_names)
        unknown = selected - set(zoo)
        if unknown:
            raise ValueError(f"Unknown model names: {sorted(unknown)}")
        zoo = {name: model for name, model in zoo.items() if name in selected}

    sample_weight = speaker_sample_weights(train.y)
    fitted_models: Dict[str, Pipeline] = {}
    val_base_preds: Dict[str, np.ndarray] = {}
    test_base_preds: Dict[str, np.ndarray] = {}
    model_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, model in zoo.items():
        fitted = _fit_pipeline(model, train.x, train.y, sample_weight)
        fitted_models[name] = fitted
        val_pred = np.asarray(fitted.predict(val.x), dtype=np.float32)
        test_pred = np.asarray(fitted.predict(test.x), dtype=np.float32)
        val_base_preds[name] = val_pred
        test_base_preds[name] = test_pred
        model_metrics[name] = {
            "val": regression_metrics(val.y, val_pred, val.metadata),
            "test": regression_metrics(test.y, test_pred, test.metadata),
        }

    weights = tune_ensemble_weights(val_base_preds, val.y, seed=seed, trials=ensemble_trials)
    val_ensemble = weighted_predictions(val_base_preds, weights)
    test_ensemble = weighted_predictions(test_base_preds, weights)
    calibrator = LinearRegression().fit(val_ensemble.reshape(-1, 1), val.y)
    val_calibrated = apply_calibrator(val_ensemble, calibrator)
    test_calibrated = apply_calibrator(test_ensemble, calibrator)
    edge_offsets = fit_edge_offsets(val_calibrated, val.y)
    val_edge = apply_edge_offsets(val_calibrated, edge_offsets)
    test_edge = apply_edge_offsets(test_calibrated, edge_offsets)

    final_val = {
        "ensemble": regression_metrics(val.y, val_ensemble, val.metadata),
        "calibrated": regression_metrics(val.y, val_calibrated, val.metadata),
        "calibrated_edge": regression_metrics(val.y, val_edge, val.metadata),
    }
    final_test = {
        "ensemble": regression_metrics(test.y, test_ensemble, test.metadata),
        "calibrated": regression_metrics(test.y, test_calibrated, test.metadata),
        "calibrated_edge": regression_metrics(test.y, test_edge, test.metadata),
    }

    payload: Dict[str, Any] = {
        "experiment": "speaker_height_research_ensemble",
        "seed": int(seed),
        "features_dir": os.path.abspath(features_dir),
        "target_mae_cm": float(target_mae_cm),
        "target_met": bool(final_test["calibrated_edge"]["mae"] <= float(target_mae_cm)),
        "feature_count": int(train.x.shape[1]),
        "speaker_counts": {
            "train": int(train.x.shape[0]),
            "val": int(val.x.shape[0]),
            "test": int(test.x.shape[0]),
        },
        "models": model_metrics,
        "ensemble_weights": weights,
        "edge_offsets": edge_offsets,
        "final_val": final_val,
        "final_test": final_test,
    }

    os.makedirs(output_dir, exist_ok=True)
    val_rows = prediction_rows(
        val,
        {"ensemble": val_ensemble, "calibrated": val_calibrated, "calibrated_edge": val_edge},
    )
    test_rows = prediction_rows(
        test,
        {"ensemble": test_ensemble, "calibrated": test_calibrated, "calibrated_edge": test_edge},
    )
    write_prediction_csv(os.path.join(output_dir, "predictions_val.csv"), val_rows)
    write_prediction_csv(os.path.join(output_dir, "predictions_test.csv"), test_rows)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    write_summary_markdown(os.path.join(output_dir, "summary.md"), payload)
    if save_model:
        model = HeightResearchModel(
            feature_names=train.feature_names,
            models=fitted_models,
            weights=weights,
            calibrator=calibrator,
            edge_offsets=edge_offsets,
            val_metrics=final_val["calibrated_edge"],
        )
        joblib.dump(model, os.path.join(output_dir, "model.joblib"))
    return payload

