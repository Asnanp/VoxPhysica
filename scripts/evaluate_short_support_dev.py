#!/usr/bin/env python
"""Development-only test of real short-speaker training support.

Selection uses target-training out-of-fold predictions. The frozen candidate is
then evaluated on validation. This script deliberately has no historical-test
input and produces no historical-test prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCALAR_FEATURES = [
    "f0_mean",
    "formant_spacing_mean",
    "vtl_mean",
    "jitter",
    "shimmer",
    "hnr",
    "duration_s",
    "voiced_ratio",
    "invalid_spacing_rate",
    "capture_quality_score",
    "distance_confidence",
    "clipped_ratio",
]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    alpha: float
    external_weight: float
    gate_threshold_cm: float | None = None
    gate_scale_cm: float | None = None
    gate_blend: float | None = None

    @property
    def gated(self) -> bool:
        return self.gate_threshold_cm is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate short-speaker support on training OOF and validation only."
    )
    parser.add_argument(
        "--feature-root",
        default="data/features_vtl_external_ssl_fast",
    )
    parser.add_argument("--train-csv", default="data/splits/train_clean.csv")
    parser.add_argument("--val-csv", default="data/splits/val_clean.csv")
    parser.add_argument(
        "--support-csv",
        default="outputs/short_speaker_collection_v1/short_speakers_accepted.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/short_support_dev_v1",
    )
    parser.add_argument("--short-threshold-cm", type=float, default=160.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _item(data: np.lib.npyio.NpzFile, key: str, default: float = math.nan) -> float:
    if key not in data.files:
        return float(default)
    try:
        return float(np.asarray(data[key]).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return float(default)


def aggregate_features(
    directory: Path,
    wanted_ids: set[str],
) -> dict[str, np.ndarray]:
    embeddings: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    scalars: defaultdict[str, list[np.ndarray]] = defaultdict(list)

    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            speaker_id = str(np.asarray(data["speaker_id"]).item())
            if speaker_id not in wanted_ids:
                continue
            if "ssl_embedding" not in data.files:
                raise RuntimeError(f"Missing ssl_embedding in {path}")
            embeddings[speaker_id].append(
                np.asarray(data["ssl_embedding"], dtype=np.float32).reshape(-1)
            )
            scalars[speaker_id].append(
                np.asarray(
                    [_item(data, key) for key in SCALAR_FEATURES],
                    dtype=np.float32,
                )
            )

    output: dict[str, np.ndarray] = {}
    for speaker_id, speaker_embeddings in embeddings.items():
        embed = np.stack(speaker_embeddings)
        scalar = np.stack(scalars[speaker_id])
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            vector = np.concatenate(
                [
                    np.nanmean(embed, axis=0),
                    np.nanstd(embed, axis=0),
                    np.nanmean(scalar, axis=0),
                    np.nanstd(scalar, axis=0),
                    np.asarray([math.log1p(len(embed))], dtype=np.float32),
                ]
            )
        output[speaker_id] = vector.astype(np.float32)
    return output


def clean_gender(value: str) -> int:
    return 1 if str(value).strip().lower().startswith("m") else 0


def arrays(
    ids: Sequence[str],
    metadata: Mapping[str, Mapping[str, str]],
    features: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack([features[speaker_id] for speaker_id in ids])
    y = np.asarray(
        [float(metadata[speaker_id]["height_cm"]) for speaker_id in ids],
        dtype=np.float64,
    )
    gender = np.asarray(
        [clean_gender(metadata[speaker_id].get("gender", "")) for speaker_id in ids],
        dtype=np.int64,
    )
    source = np.asarray(
        [metadata[speaker_id].get("source", "UNKNOWN") for speaker_id in ids],
        dtype=object,
    )
    x = np.column_stack([x, gender.astype(np.float32)])
    return x, y, gender, source


def height_bin(y: np.ndarray, short_cm: float) -> np.ndarray:
    return np.where(y < short_cm, "short", np.where(y < 175.0, "medium", "tall"))


def regression_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    *,
    short_cm: float,
) -> dict[str, float | int]:
    error = np.abs(np.asarray(y) - np.asarray(pred))
    result: dict[str, float | int] = {
        "n": int(len(y)),
        "mae": float(np.mean(error)),
        "median_ae": float(np.median(error)),
        "rmse": float(np.sqrt(np.mean(np.square(np.asarray(y) - np.asarray(pred))))),
        "within_3cm": float(np.mean(error <= 3.0)),
    }
    masks = {
        "short": y < short_cm,
        "medium": (y >= short_cm) & (y < 175.0),
        "tall": y >= 175.0,
    }
    for name, mask in masks.items():
        result[f"{name}_n"] = int(mask.sum())
        result[f"{name}_mae"] = (
            float(np.mean(error[mask])) if mask.any() else math.nan
        )
    return result


def paired_bootstrap_delta(
    y: np.ndarray,
    baseline: np.ndarray,
    selected: np.ndarray,
    *,
    seed: int,
    samples: int = 10_000,
) -> dict[str, float | int]:
    per_speaker = np.abs(y - selected) - np.abs(y - baseline)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(y), size=(samples, len(y)))
    means = np.mean(per_speaker[indices], axis=1)
    return {
        "samples": int(samples),
        "mean_delta_mae_cm": float(np.mean(per_speaker)),
        "ci95_lower_cm": float(np.quantile(means, 0.025)),
        "ci95_upper_cm": float(np.quantile(means, 0.975)),
        "bootstrap_probability_improvement": float(np.mean(means < 0.0)),
    }


def make_folds(
    y: np.ndarray,
    gender: np.ndarray,
    source: np.ndarray,
    *,
    short_cm: float,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(
        [
            f"{src}|{sex}|{bin_name}"
            for src, sex, bin_name in zip(source, gender, height_bin(y, short_cm))
        ]
    )
    counts = Counter(labels.tolist())
    if min(counts.values()) < folds:
        labels = np.asarray(
            [
                f"{sex}|{bin_name}"
                for sex, bin_name in zip(gender, height_bin(y, short_cm))
            ]
        )
    counts = Counter(labels.tolist())
    if min(counts.values()) < folds:
        labels = height_bin(y, short_cm)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), labels))


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    x_query: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=alpha),
    )
    model.fit(x_train, y_train, ridge__sample_weight=sample_weight)
    return np.asarray(model.predict(x_query), dtype=np.float64)


def base_oof(
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    alpha: float,
) -> np.ndarray:
    pred = np.zeros(len(y), dtype=np.float64)
    for train_index, holdout_index in folds:
        pred[holdout_index] = ridge_predict(
            x[train_index],
            y[train_index],
            np.ones(len(train_index), dtype=np.float64),
            x[holdout_index],
            alpha=alpha,
        )
    return pred


def external_oof(
    x: np.ndarray,
    y: np.ndarray,
    x_external: np.ndarray,
    y_external: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    alpha: float,
    external_weight: float,
) -> np.ndarray:
    pred = np.zeros(len(y), dtype=np.float64)
    for train_index, holdout_index in folds:
        x_fit = np.vstack([x[train_index], x_external])
        y_fit = np.concatenate([y[train_index], y_external])
        weights = np.concatenate(
            [
                np.ones(len(train_index), dtype=np.float64),
                np.full(len(y_external), external_weight, dtype=np.float64),
            ]
        )
        pred[holdout_index] = ridge_predict(
            x_fit,
            y_fit,
            weights,
            x[holdout_index],
            alpha=alpha,
        )
    return pred


def gated_prediction(
    base: np.ndarray,
    external: np.ndarray,
    spec: CandidateSpec,
) -> np.ndarray:
    assert spec.gate_threshold_cm is not None
    assert spec.gate_scale_cm is not None
    assert spec.gate_blend is not None
    gate = 1.0 / (
        1.0 + np.exp((base - spec.gate_threshold_cm) / spec.gate_scale_cm)
    )
    return base + spec.gate_blend * gate * (external - base)


def fit_validation_prediction(
    spec: CandidateSpec,
    *,
    x: np.ndarray,
    y: np.ndarray,
    x_external: np.ndarray,
    y_external: np.ndarray,
    x_val: np.ndarray,
) -> np.ndarray:
    base = ridge_predict(
        x,
        y,
        np.ones(len(y), dtype=np.float64),
        x_val,
        alpha=spec.alpha,
    )
    if spec.external_weight <= 0.0:
        return base

    x_fit = np.vstack([x, x_external])
    y_fit = np.concatenate([y, y_external])
    weights = np.concatenate(
        [
            np.ones(len(y), dtype=np.float64),
            np.full(len(y_external), spec.external_weight, dtype=np.float64),
        ]
    )
    external = ridge_predict(
        x_fit,
        y_fit,
        weights,
        x_val,
        alpha=spec.alpha,
    )
    if spec.gated:
        return gated_prediction(base, external, spec)
    return external


def write_predictions(
    path: Path,
    ids: Sequence[str],
    y: np.ndarray,
    baseline: np.ndarray,
    selected: np.ndarray,
    metadata: Mapping[str, Mapping[str, str]],
    *,
    short_cm: float,
) -> None:
    fields = [
        "speaker_id",
        "source",
        "gender",
        "height_group",
        "height_cm",
        "baseline_prediction_cm",
        "selected_prediction_cm",
        "baseline_abs_error_cm",
        "selected_abs_error_cm",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, speaker_id in enumerate(ids):
            truth = float(y[index])
            group = "short" if truth < short_cm else ("medium" if truth < 175.0 else "tall")
            writer.writerow(
                {
                    "speaker_id": speaker_id,
                    "source": metadata[speaker_id].get("source", ""),
                    "gender": metadata[speaker_id].get("gender", ""),
                    "height_group": group,
                    "height_cm": f"{truth:.6f}",
                    "baseline_prediction_cm": f"{baseline[index]:.6f}",
                    "selected_prediction_cm": f"{selected[index]:.6f}",
                    "baseline_abs_error_cm": f"{abs(truth - baseline[index]):.6f}",
                    "selected_abs_error_cm": f"{abs(truth - selected[index]):.6f}",
                }
            )


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    baseline = payload["validation"]["target_only_baseline"]
    selected = payload["validation"]["selected"]
    delta = payload["validation"]["delta_selected_minus_baseline"]
    bootstrap = payload["validation"]["paired_bootstrap_delta"]
    lines = [
        "# Short-Speaker Support Development Result",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "## Frozen outcome",
        "",
        (
            f"Target-only validation MAE: **{baseline['mae']:.3f} cm**. "
            f"Selected public-support validation MAE: **{selected['mae']:.3f} cm**."
        ),
        (
            f"Difference (support minus baseline): **{delta['mae']:+.3f} cm**. "
            f"Short-slice difference: **{delta['short_mae']:+.3f} cm**."
        ),
        (
            f"Paired 95% bootstrap interval for the overall MAE difference: "
            f"**{bootstrap['ci95_lower_cm']:+.3f} to "
            f"{bootstrap['ci95_upper_cm']:+.3f} cm**."
        ),
        "",
        "This is a development validation result, not a new test result.",
        "",
        "## Data",
        "",
        f"- Target training speakers: {payload['counts']['target_train_speakers']}",
        f"- Public short support speakers: {payload['counts']['public_short_support_speakers']}",
        f"- Support gender counts: {payload['counts']['support_gender_counts']}",
        f"- Validation speakers: {payload['counts']['validation_speakers']}",
        "",
        "## Selection integrity",
        "",
        "- Candidate selection used target-training out-of-fold predictions only.",
        "- HeightCeleb labels were given reduced candidate weights and used only for training support.",
        "- The selected gate and regularization were frozen before validation scoring.",
        "- No historical-test file or label was loaded.",
        "",
        "## Interpretation",
        "",
        (
            "The observed gain is negligible and must not be presented as evidence "
            "that the 3 cm target was reached. The support set is almost entirely "
            "female and its heights are public estimates, so prospective measured "
            "and demographically balanced collection remains necessary."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    feature_root = resolve(args.feature_root)
    train_path = resolve(args.train_csv)
    val_path = resolve(args.val_csv)
    support_path = resolve(args.support_csv)
    output_dir = resolve(args.output_dir)
    for required in (feature_root, train_path, val_path, support_path):
        if not required.exists():
            raise FileNotFoundError(required)

    train_rows = read_rows(train_path)
    val_rows = read_rows(val_path)
    support_rows = [
        row
        for row in read_rows(support_path)
        if float(row["height_cm"]) < float(args.short_threshold_cm)
        and row.get("collection_role") == "train_support_only"
    ]
    train_meta = {row["speaker_id"]: row for row in train_rows}
    val_meta = {row["speaker_id"]: row for row in val_rows}
    support_meta = {row["speaker_id"]: row for row in support_rows}

    train_features = aggregate_features(
        feature_root / "train",
        set(train_meta) | set(support_meta),
    )
    val_features = aggregate_features(feature_root / "val", set(val_meta))
    train_ids = [speaker_id for speaker_id in train_meta if speaker_id in train_features]
    val_ids = [speaker_id for speaker_id in val_meta if speaker_id in val_features]
    support_ids = [
        speaker_id for speaker_id in support_meta if speaker_id in train_features
    ]
    if len(train_ids) != len(train_meta):
        raise RuntimeError(
            f"Missing target-train speaker features: {len(train_meta) - len(train_ids)}"
        )
    if len(val_ids) != len(val_meta):
        raise RuntimeError(
            f"Missing validation speaker features: {len(val_meta) - len(val_ids)}"
        )
    if len(support_ids) != len(support_meta):
        raise RuntimeError(
            f"Missing support speaker features: {len(support_meta) - len(support_ids)}"
        )

    x, y, gender, source = arrays(train_ids, train_meta, train_features)
    x_val, y_val, _, _ = arrays(val_ids, val_meta, val_features)
    x_external, y_external, external_gender, _ = arrays(
        support_ids,
        support_meta,
        train_features,
    )
    observed = np.any(
        np.isfinite(np.vstack([x, x_external, x_val])),
        axis=0,
    )
    x = x[:, observed]
    x_external = x_external[:, observed]
    x_val = x_val[:, observed]
    folds = make_folds(
        y,
        gender,
        source,
        short_cm=float(args.short_threshold_cm),
        folds=int(args.folds),
        seed=int(args.seed),
    )

    alphas = [10.0, 100.0, 1000.0, 10000.0] if args.quick else [
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
        300.0,
        1000.0,
        3000.0,
        10000.0,
    ]
    external_weights = [0.1, 0.4, 0.8] if args.quick else [
        0.02,
        0.05,
        0.1,
        0.2,
        0.4,
        0.8,
    ]
    thresholds = [165.0] if args.quick else [160.0, 165.0, 170.0]
    scales = [5.0] if args.quick else [2.0, 5.0, 10.0]
    blends = [0.5, 1.0] if args.quick else [0.25, 0.5, 0.75, 1.0]

    base_predictions: dict[float, np.ndarray] = {}
    external_predictions: dict[tuple[float, float], np.ndarray] = {}
    candidates: list[tuple[CandidateSpec, dict[str, float | int], np.ndarray]] = []

    for alpha in alphas:
        base = base_oof(x, y, folds, alpha=alpha)
        base_predictions[alpha] = base
        spec = CandidateSpec(f"ridge_a{alpha:g}_target_only", alpha, 0.0)
        candidates.append(
            (
                spec,
                regression_metrics(y, base, short_cm=float(args.short_threshold_cm)),
                base,
            )
        )
        for external_weight in external_weights:
            external = external_oof(
                x,
                y,
                x_external,
                y_external,
                folds,
                alpha=alpha,
                external_weight=external_weight,
            )
            external_predictions[(alpha, external_weight)] = external
            direct_spec = CandidateSpec(
                f"ridge_a{alpha:g}_external_w{external_weight:g}",
                alpha,
                external_weight,
            )
            candidates.append(
                (
                    direct_spec,
                    regression_metrics(
                        y,
                        external,
                        short_cm=float(args.short_threshold_cm),
                    ),
                    external,
                )
            )
            for threshold in thresholds:
                for scale in scales:
                    for blend in blends:
                        spec = CandidateSpec(
                            (
                                f"gate_a{alpha:g}_w{external_weight:g}_"
                                f"t{threshold:g}_s{scale:g}_b{blend:g}"
                            ),
                            alpha,
                            external_weight,
                            threshold,
                            scale,
                            blend,
                        )
                        pred = gated_prediction(base, external, spec)
                        candidates.append(
                            (
                                spec,
                                regression_metrics(
                                    y,
                                    pred,
                                    short_cm=float(args.short_threshold_cm),
                                ),
                                pred,
                            )
                        )

    target_only = [row for row in candidates if row[0].external_weight == 0.0]
    baseline_spec, baseline_oof_metrics, baseline_oof = min(
        target_only,
        key=lambda row: (float(row[1]["mae"]), float(row[1]["short_mae"])),
    )
    eligible = [
        row
        for row in candidates
        if float(row[1]["short_mae"])
        <= float(baseline_oof_metrics["short_mae"]) + 1e-12
    ]
    selected_spec, selected_oof_metrics, selected_oof = min(
        eligible or candidates,
        key=lambda row: (float(row[1]["mae"]), float(row[1]["short_mae"])),
    )

    baseline_val = fit_validation_prediction(
        baseline_spec,
        x=x,
        y=y,
        x_external=x_external,
        y_external=y_external,
        x_val=x_val,
    )
    selected_val = fit_validation_prediction(
        selected_spec,
        x=x,
        y=y,
        x_external=x_external,
        y_external=y_external,
        x_val=x_val,
    )
    baseline_val_metrics = regression_metrics(
        y_val,
        baseline_val,
        short_cm=float(args.short_threshold_cm),
    )
    selected_val_metrics = regression_metrics(
        y_val,
        selected_val,
        short_cm=float(args.short_threshold_cm),
    )
    delta = {
        key: float(selected_val_metrics[key]) - float(baseline_val_metrics[key])
        for key in ("mae", "short_mae", "medium_mae", "tall_mae")
    }
    delta_bootstrap = paired_bootstrap_delta(
        y_val,
        baseline_val,
        selected_val,
        seed=int(args.seed) + 101,
    )

    ranked = sorted(
        candidates,
        key=lambda row: (float(row[1]["mae"]), float(row[1]["short_mae"])),
    )
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "development_validation_only_no_historical_test_loaded",
        "selection_rule": (
            "minimum target-train OOF MAE among candidates whose short OOF MAE "
            "does not exceed the best target-only candidate"
        ),
        "short_threshold_cm_exclusive": float(args.short_threshold_cm),
        "counts": {
            "target_train_speakers": len(train_ids),
            "public_short_support_speakers": len(support_ids),
            "validation_speakers": len(val_ids),
            "support_gender_counts": dict(
                Counter(
                    "Male" if value == 1 else "Female"
                    for value in external_gender.tolist()
                )
            ),
        },
        "baseline": {
            "spec": asdict(baseline_spec),
            "train_oof": baseline_oof_metrics,
        },
        "selected": {
            "spec": asdict(selected_spec),
            "train_oof": selected_oof_metrics,
        },
        "validation": {
            "target_only_baseline": baseline_val_metrics,
            "selected": selected_val_metrics,
            "delta_selected_minus_baseline": delta,
            "paired_bootstrap_delta": delta_bootstrap,
        },
        "candidate_count": len(candidates),
        "top_train_oof_candidates": [
            {"spec": asdict(spec), "metrics": metrics}
            for spec, metrics, _ in ranked[:100]
        ],
        "warnings": [
            "HeightCeleb height labels are internet-derived estimates.",
            "Public short support is highly gender imbalanced.",
            "The historical test set was not evaluated in this experiment.",
            "A tiny validation delta is not evidence of a population-level gain.",
        ],
        "args": vars(args),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "short_support_dev_results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    write_predictions(
        output_dir / "validation_predictions.csv",
        val_ids,
        y_val,
        baseline_val,
        selected_val,
        val_meta,
        short_cm=float(args.short_threshold_cm),
    )
    write_report(output_dir / "SHORT_SUPPORT_DEV_REPORT.md", payload)

    print(
        f"[short-support] selected={selected_spec.name} "
        f"OOF={selected_oof_metrics['mae']:.4f} "
        f"validation={selected_val_metrics['mae']:.4f}",
        flush=True,
    )
    print(
        f"[short-support] target-only validation={baseline_val_metrics['mae']:.4f} "
        f"delta={delta['mae']:+.4f}",
        flush=True,
    )
    print("[short-support] historical test not loaded", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
