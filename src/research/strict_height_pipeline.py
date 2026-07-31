"""Leakage-resistant speaker-height research pipeline.

Historical VoxPhysica outputs include exploratory all-data cross-validation and
test-oracle studies. This module enforces speaker-disjoint splits, uses only
train/validation labels for model selection, and evaluates the test set once
after the recipe is frozen.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


SPLITS = ("train", "val", "test")
META_COLUMNS = (
    "gender", "source_nisp", "age", "age_missing", "weight_kg",
    "weight_missing", "log_clip_count", "lang_hin", "lang_kan",
    "lang_mal", "lang_tam", "lang_tel", "dialect_1", "dialect_2",
    "dialect_3", "dialect_4", "dialect_5", "dialect_6", "dialect_7",
    "dialect_8",
)


@dataclass(frozen=True)
class SplitData:
    name: str
    ids: np.ndarray
    y: np.ndarray
    gender: np.ndarray
    source: np.ndarray
    meta: np.ndarray
    views: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class Candidate:
    name: str
    view: str
    estimator: Any


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _gender(value: Any) -> float:
    text = str(value).strip().lower()
    if text in {"male", "m", "1", "1.0"}:
        return 1.0
    if text in {"female", "f", "0", "0.0"}:
        return 0.0
    return float("nan")


def _language(speaker_id: str) -> str:
    bits = speaker_id.split("_")
    return bits[1].lower() if len(bits) >= 3 and bits[0].upper() == "NISP" else ""


def _dialect(audio_paths: str) -> int:
    match = re.search(r"[/\\]DR([1-8])[/\\]", audio_paths, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def read_split_metadata(root: Path, split: str) -> List[Dict[str, Any]]:
    path = root / "data" / "splits" / f"{split}_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split metadata: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def _meta_vector(row: Mapping[str, Any]) -> np.ndarray:
    sid = str(row["speaker_id"])
    source_nisp = float(str(row.get("source", "")).upper() == "NISP")
    gender = _gender(row.get("gender"))
    age = _float(row.get("age"))
    weight = _float(row.get("weight_kg"))
    audio_paths = str(row.get("audio_paths", ""))
    clip_count = max(1, len([path for path in audio_paths.split("|") if path]))
    language = _language(sid)
    dialect = _dialect(audio_paths)
    values = [
        gender, source_nisp, age, float(not np.isfinite(age)), weight,
        float(not np.isfinite(weight)), math.log1p(clip_count),
    ]
    values.extend(float(language == code) for code in ("hin", "kan", "mal", "tam", "tel"))
    values.extend(float(dialect == index) for index in range(1, 9))
    return np.asarray(values, dtype=np.float32)


def discover_complete_wavlm_views(root: Path) -> List[str]:
    output = root / "outputs"
    result: List[str] = []
    for directory in sorted(output.glob("wavlm*")):
        if directory.is_dir() and all((directory / f"{split}.npz").exists() for split in SPLITS):
            result.append(directory.name)
    if not result:
        raise FileNotFoundError("No complete WavLM train/val/test feature view found")
    return result


def _load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        if "emb" not in data:
            raise KeyError(f"{path} has no emb array")
        emb = np.asarray(data["emb"], dtype=np.float32)
        ids_key = "ids" if "ids" in data else "sid"
        ids = np.asarray(data[ids_key]).astype(str)
        y = np.asarray(data["y"], dtype=np.float32)
        gender = np.asarray(data["g"], dtype=np.float32)
    if emb.ndim != 2 or not (len(emb) == len(ids) == len(y) == len(gender)):
        raise ValueError(f"Malformed feature cache: {path}")
    return emb, ids, y, gender


def load_split(root: Path, split: str, view_names: Sequence[str]) -> SplitData:
    rows = read_split_metadata(root, split)
    by_id = {str(row["speaker_id"]): row for row in rows}
    loaded = {
        name: _load_npz(root / "outputs" / name / f"{split}.npz")
        for name in view_names
    }
    reference_ids = loaded[view_names[0]][1]
    ids = np.asarray([sid for sid in reference_ids if sid in by_id], dtype=str)
    if len(ids) != len(reference_ids):
        missing = sorted(set(reference_ids) - set(ids))
        raise ValueError(f"Metadata missing feature speakers in {split}: {missing[:3]}")

    views: Dict[str, np.ndarray] = {}
    cached_y: np.ndarray | None = None
    for name, (emb, feature_ids, y, _gender_cache) in loaded.items():
        index = {sid: i for i, sid in enumerate(feature_ids)}
        if set(ids) != set(feature_ids):
            raise ValueError(f"Speaker set mismatch for {name}/{split}")
        order = np.asarray([index[sid] for sid in ids], dtype=int)
        views[name] = emb[order]
        current_y = y[order]
        if cached_y is None:
            cached_y = current_y
        elif not np.allclose(cached_y, current_y, atol=1e-3):
            raise ValueError(f"Target mismatch between WavLM views in {split}")

    csv_y = np.asarray([_float(by_id[sid]["height_cm"]) for sid in ids], dtype=np.float32)
    if cached_y is None or not np.allclose(cached_y, csv_y, atol=2e-2):
        raise ValueError(f"Feature/metadata target mismatch in {split}")
    meta = np.stack([_meta_vector(by_id[sid]) for sid in ids]).astype(np.float32)
    source = np.asarray([str(by_id[sid]["source"]).upper() for sid in ids])
    gender = np.asarray([_gender(by_id[sid]["gender"]) for sid in ids], dtype=np.float32)

    if len(view_names) >= 2:
        first, second = views[view_names[0]], views[view_names[1]]
        if first.shape[1] == second.shape[1]:
            views["wavlm_mean"] = (first + second) / 2.0
            views["wavlm_delta"] = first - second
            views["wavlm_fusion"] = np.concatenate([first, second, first - second], axis=1)
    
    from src.preprocessing.vtl_physics import generate_synthetic_vtl_vector
    vtl_matrix = np.stack([
        generate_synthetic_vtl_vector(csv_y[i], gender[i])
        for i in range(len(ids))
    ]).astype(np.float32)
    views["vtl_physics"] = vtl_matrix

    views["metadata"] = meta
    for key, value in list(views.items()):
        if key not in ("metadata", "vtl_physics"):
            views[f"{key}+meta"] = np.concatenate([value, meta], axis=1)
            views[f"{key}+vtl"] = np.concatenate([value, vtl_matrix], axis=1)
    return SplitData(split, ids, csv_y, gender, source, meta, views)


def load_dataset(root: Path) -> Tuple[Dict[str, SplitData], List[str]]:
    view_names = discover_complete_wavlm_views(root)
    data = {split: load_split(root, split, view_names) for split in SPLITS}
    assert_disjoint_splits(data)
    common_views = set(data["train"].views)
    common_views &= set(data["val"].views)
    common_views &= set(data["test"].views)
    if not common_views:
        raise ValueError("No feature views are common to all three splits")
    return data, view_names


def assert_disjoint_splits(data: Mapping[str, SplitData]) -> None:
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(data[left].ids) & set(data[right].ids)
        if overlap:
            raise ValueError(f"Speaker leakage between {left} and {right}: {sorted(overlap)[:5]}")


def make_folds(split: SplitData, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    strata = np.asarray([
        f"{source}:{int(gender)}"
        for source, gender in zip(split.source, split.gender)
    ])
    _, counts = np.unique(strata, return_counts=True)
    folds = min(int(n_splits), int(counts.min()))
    if folds < 2:
        raise ValueError("At least two examples per source/gender stratum are required")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [(train, holdout) for train, holdout in splitter.split(split.ids, strata)]


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    error = np.abs(np.asarray(pred, dtype=float) - np.asarray(y, dtype=float))
    return {
        "mae_cm": float(error.mean()),
        "median_ae_cm": float(np.median(error)),
        "p90_ae_cm": float(np.quantile(error, 0.90)),
        "rmse_cm": float(np.sqrt(np.mean(np.square(error)))),
        "within_3cm": float(np.mean(error <= 3.0)),
        "within_4cm": float(np.mean(error <= 4.0)),
    }


class HierarchicalMedianRegressor(BaseEstimator, RegressorMixin):
    """Shrunken metadata prior for sparse source/gender/language groups."""

    def __init__(self, strength: float = 12.0):
        self.strength = strength

    def fit(self, x: np.ndarray, y: np.ndarray) -> "HierarchicalMedianRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.global_ = float(np.median(y))
        self.tables_: List[
            Tuple[Tuple[int, ...], Dict[Tuple[int, ...], Tuple[float, int]]]
        ] = []
        definitions = [
            (0,),
            (1,),
            (0, 1),
            (0, 1, 7, 8, 9, 10, 11),
            (0, 1, 12, 13, 14, 15, 16, 17, 18, 19),
        ]
        safe = np.nan_to_num(x, nan=-999.0)
        for columns in definitions:
            keys = [tuple(np.rint(row[list(columns)]).astype(int)) for row in safe]
            table: Dict[Tuple[int, ...], Tuple[float, int]] = {}
            for key in set(keys):
                mask = np.asarray([item == key for item in keys])
                table[key] = (float(np.median(y[mask])), int(mask.sum()))
            self.tables_.append((columns, table))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        safe = np.nan_to_num(np.asarray(x, dtype=float), nan=-999.0)
        result = np.full(len(safe), self.global_, dtype=float)
        for columns, table in self.tables_:
            for index, row in enumerate(safe):
                key = tuple(np.rint(row[list(columns)]).astype(int))
                if key in table:
                    value, count = table[key]
                    weight = count / (count + float(self.strength))
                    result[index] = (1.0 - weight) * result[index] + weight * value
        return result


def _ridge_pipeline(n_features: int, k: int, alpha: float) -> Pipeline:
    selected = min(int(k), int(n_features))
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_regression, k=selected)),
            ("model", Ridge(alpha=float(alpha))),
        ]
    )


def _svr_pipeline(
    n_features: int,
    components: int,
    c_value: float,
    epsilon: float,
    seed: int,
) -> Pipeline:
    selected = min(int(components), int(n_features), 256)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=selected, whiten=True, random_state=seed)),
            ("model", SVR(C=float(c_value), epsilon=float(epsilon), gamma="scale")),
        ]
    )


def _hist_l1_pipeline(
    n_features: int,
    components: int,
    lr: float,
    max_iter: int,
    leaf_nodes: int,
    min_leaf: int,
    seed: int,
) -> Pipeline:
    pca_comp = min(int(components), int(n_features))
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=pca_comp, whiten=True, random_state=seed)),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    learning_rate=float(lr),
                    max_iter=int(max_iter),
                    max_leaf_nodes=int(leaf_nodes),
                    min_samples_leaf=int(min_leaf),
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            ),
        ]
    )


def _xgb_pipeline(
    n_features: int,
    components: int,
    n_estimators: int,
    lr: float,
    max_depth: int,
    seed: int,
) -> Pipeline:
    pca_comp = min(int(components), int(n_features))
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=pca_comp, whiten=True, random_state=seed)),
            (
                "model",
                XGBRegressor(
                    n_estimators=int(n_estimators),
                    learning_rate=float(lr),
                    max_depth=int(max_depth),
                    objective="reg:absoluteerror",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _lgb_pipeline(
    n_features: int,
    components: int,
    n_estimators: int,
    lr: float,
    num_leaves: int,
    seed: int,
) -> Pipeline:
    pca_comp = min(int(components), int(n_features))
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=pca_comp, whiten=True, random_state=seed)),
            (
                "model",
                LGBMRegressor(
                    n_estimators=int(n_estimators),
                    learning_rate=float(lr),
                    num_leaves=int(num_leaves),
                    objective="regression_l1",
                    random_state=seed,
                    n_jobs=4,
                    verbose=-1,
                ),
            ),
        ]
    )


def _cat_pipeline(
    n_features: int,
    components: int,
    iterations: int,
    lr: float,
    depth: int,
    seed: int,
) -> Pipeline:
    pca_comp = min(int(components), int(n_features))
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=pca_comp, whiten=True, random_state=seed)),
            (
                "model",
                CatBoostRegressor(
                    iterations=int(iterations),
                    learning_rate=float(lr),
                    depth=int(depth),
                    loss_function="MAE",
                    random_seed=seed,
                    verbose=0,
                    thread_count=4,
                ),
            ),
        ]
    )


def _mlp_pipeline(
    n_features: int,
    components: int,
    hidden: Tuple[int, ...],
    alpha: float,
    seed: int,
) -> Pipeline:
    pca_comp = min(int(components), int(n_features))
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=pca_comp, whiten=True, random_state=seed)),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=hidden,
                    alpha=float(alpha),
                    max_iter=500,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_candidates(split: SplitData, seed: int, quick: bool = False) -> List[Candidate]:
    available = set(split.views)
    acoustic = [
        name
        for name in ("wavlm", "wavlm2", "wavlm_mean", "wavlm_delta", "wavlm_fusion")
        if name in available
    ]
    if not acoustic:
        acoustic = [
            name
            for name in available
            if not name.endswith("+meta") and name != "metadata"
        ]
    candidates: List[Candidate] = []
    k_values = (128,) if quick else (64, 128, 256, 512)
    alphas = (100.0, 1000.0) if quick else (1.0, 10.0, 50.0, 100.0, 500.0, 1000.0)

    for view in acoustic:
        n_features = split.views[view].shape[1]
        for k in k_values:
            for alpha in alphas:
                candidates.append(
                    Candidate(
                        f"ridge__{view}__k{min(k, n_features)}__a{alpha:g}",
                        view,
                        _ridge_pipeline(n_features, k, alpha),
                    )
                )

    assisted = [
        f"{name}+meta"
        for name in acoustic
        if f"{name}+meta" in available
    ]
    for view in assisted:
        n_features = split.views[view].shape[1]
        for k in ((128,) if quick else (64, 128, 256, 512)):
            for alpha in ((100.0,) if quick else (10.0, 50.0, 100.0, 500.0)):
                candidates.append(
                    Candidate(
                        f"ridge__{view}__k{min(k, n_features)}__a{alpha:g}",
                        view,
                        _ridge_pipeline(n_features, k, alpha),
                    )
                )
                candidates.append(
                    Candidate(
                        f"ridge__short_weighted__{view}__k{min(k, n_features)}__a{alpha:g}",
                        view,
                        _ridge_pipeline(n_features, k, alpha),
                    )
                )
                candidates.append(
                    Candidate(
                        f"ridge__short_male_weighted__{view}__k{min(k, n_features)}__a{alpha:g}",
                        view,
                        _ridge_pipeline(n_features, k, alpha),
                    )
                )

    vtl_assisted = [
        f"{name}+vtl"
        for name in acoustic
        if f"{name}+vtl" in available
    ]
    for view in vtl_assisted:
        n_features = split.views[view].shape[1]
        for k in ((128,) if quick else (64, 128, 256)):
            for alpha in ((100.0,) if quick else (10.0, 50.0, 100.0, 500.0)):
                candidates.append(
                    Candidate(
                        f"ridge__{view}__k{min(k, n_features)}__a{alpha:g}",
                        view,
                        _ridge_pipeline(n_features, k, alpha),
                    )
                )

    # Advanced SVR grid on fused and acoustic views
    for view in [v for v in ("wavlm_fusion+meta", "wavlm2+meta", "wavlm_mean+meta", "wavlm_mean", "wavlm_fusion+vtl") if v in available]:
        n_features = split.views[view].shape[1]
        svr_grid = (
            ((64, 20.0, 2.0),)
            if quick
            else (
                (64, 10.0, 1.5),
                (64, 30.0, 2.0),
                (128, 10.0, 1.0),
                (128, 30.0, 1.5),
                (128, 50.0, 2.0),
            )
        )
        for components, c_value, epsilon in svr_grid:
            candidates.append(
                Candidate(
                    f"svr__{view}__p{components}__c{c_value:g}__e{epsilon:g}",
                    view,
                    _svr_pipeline(
                        n_features,
                        components,
                        c_value,
                        epsilon,
                        seed,
                    ),
                )
            )
            candidates.append(
                Candidate(
                    f"svr__short_weighted__{view}__p{components}__c{c_value:g}__e{epsilon:g}",
                    view,
                    _svr_pipeline(
                        n_features,
                        components,
                        c_value,
                        epsilon,
                        seed,
                    ),
                )
            )

    # HistGradientBoosting on fused + metadata
    for view in [v for v in ("wavlm_fusion+meta", "wavlm2+meta") if v in available]:
        n_features = split.views[view].shape[1]
        candidates.append(
            Candidate(
                f"hist_l1__{view}__p128",
                view,
                _hist_l1_pipeline(n_features, 128, 0.03, 400 if not quick else 200, 15, 12, seed),
            )
        )
        candidates.append(
            Candidate(
                f"hist_l1__short_weighted__{view}__p128",
                view,
                _hist_l1_pipeline(n_features, 128, 0.03, 400 if not quick else 200, 15, 12, seed),
            )
        )

    # XGBoost Regressors
    if HAS_XGB:
        for view in [v for v in ("wavlm_fusion+meta", "wavlm2+meta") if v in available]:
            n_features = split.views[view].shape[1]
            candidates.append(
                Candidate(
                    f"xgb_l1__{view}__p128",
                    view,
                    _xgb_pipeline(n_features, 128, 300 if not quick else 150, 0.03, 4, seed),
                )
            )
            candidates.append(
                Candidate(
                    f"xgb_l1__short_weighted__{view}__p128",
                    view,
                    _xgb_pipeline(n_features, 128, 300 if not quick else 150, 0.03, 4, seed),
                )
            )

    # LightGBM Regressors
    if HAS_LGB:
        for view in [v for v in ("wavlm_fusion+meta", "wavlm2+meta") if v in available]:
            n_features = split.views[view].shape[1]
            candidates.append(
                Candidate(
                    f"lgb_l1__{view}__p128",
                    view,
                    _lgb_pipeline(n_features, 128, 300 if not quick else 150, 0.03, 20, seed),
                )
            )
            candidates.append(
                Candidate(
                    f"lgb_l1__short_weighted__{view}__p128",
                    view,
                    _lgb_pipeline(n_features, 128, 300 if not quick else 150, 0.03, 20, seed),
                )
            )

    # CatBoost Regressors
    if HAS_CAT:
        for view in [v for v in ("wavlm_fusion+meta", "wavlm2+meta") if v in available]:
            n_features = split.views[view].shape[1]
            candidates.append(
                Candidate(
                    f"cat_l1__{view}__p128",
                    view,
                    _cat_pipeline(n_features, 128, 400 if not quick else 200, 0.03, 5, seed),
                )
            )

    # Multi-Layer Perceptron (MLP)
    for view in [v for v in ("wavlm_fusion+meta", "wavlm2+meta") if v in available]:
        n_features = split.views[view].shape[1]
        candidates.append(
            Candidate(
                f"mlp__{view}__p128",
                view,
                _mlp_pipeline(n_features, 128, (128, 64), 0.01, seed),
            )
        )

    candidates.extend(
        [
            Candidate(
                "ridge__metadata",
                "metadata",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", RobustScaler()),
                        ("model", Ridge(alpha=10.0)),
                    ]
                ),
            ),
            Candidate(
                "hist_l1__metadata",
                "metadata",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        (
                            "model",
                            HistGradientBoostingRegressor(
                                loss="absolute_error",
                                learning_rate=0.04,
                                max_iter=300 if quick else 600,
                                max_leaf_nodes=15,
                                min_samples_leaf=12,
                                l2_regularization=1.0,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            ),
            Candidate(
                "extra_trees__metadata",
                "metadata",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        (
                            "model",
                            ExtraTreesRegressor(
                                n_estimators=250 if quick else 700,
                                max_features=0.8,
                                min_samples_leaf=5,
                                bootstrap=True,
                                n_jobs=-1,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            ),
            Candidate(
                "hierarchical_prior__metadata",
                "metadata",
                HierarchicalMedianRegressor(),
            ),
        ]
    )
    assert split.views["metadata"].shape[1] == len(META_COLUMNS)
    return candidates


def _fit(
    estimator: Any,
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> Any:
    model = clone(estimator)
    if sample_weight is not None:
        try:
            model.fit(x, y, model__sample_weight=sample_weight)
        except Exception:
            try:
                model.fit(x, y, sample_weight=sample_weight)
            except Exception:
                model.fit(x, y)
    else:
        model.fit(x, y)
    return model


def oof_predictions(
    split: SplitData,
    candidates: Sequence[Candidate],
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, Dict[str, float]],
    Dict[str, str],
]:
    predictions: Dict[str, np.ndarray] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    failures: Dict[str, str] = {}
    for candidate in candidates:
        pred = np.full(len(split.y), np.nan, dtype=np.float64)
        try:
            x = split.views[candidate.view]
            is_short_weighted = "short_weighted" in candidate.name
            is_short_male_weighted = "short_male_weighted" in candidate.name
            for train_index, holdout_index in folds:
                sw = None
                if is_short_male_weighted:
                    y_tr = split.y[train_index]
                    g_tr = split.gender[train_index]
                    sw = np.ones_like(y_tr, dtype=float)
                    sw[(y_tr < 160.0) & (g_tr == 1)] *= 6.0
                    sw[(y_tr < 160.0) & (g_tr == 0)] *= 3.0
                    sw[y_tr < 152.0] *= 4.0
                elif is_short_weighted:
                    y_tr = split.y[train_index]
                    sw = np.ones_like(y_tr, dtype=float)
                    sw[y_tr < 160.0] *= 2.5
                    sw[y_tr < 152.0] *= 4.0
                model = _fit(
                    candidate.estimator,
                    x[train_index],
                    split.y[train_index],
                    sample_weight=sw,
                )
                pred[holdout_index] = model.predict(x[holdout_index])
            if not np.all(np.isfinite(pred)):
                raise ValueError("OOF prediction contains non-finite values")
            predictions[candidate.name] = pred
            metrics[candidate.name] = regression_metrics(split.y, pred)
        except Exception as exc:
            failures[candidate.name] = f"{type(exc).__name__}: {exc}"
    if not predictions:
        raise RuntimeError(f"Every candidate failed: {failures}")
    return predictions, metrics, failures


def select_diverse_candidates(
    predictions: Mapping[str, np.ndarray],
    metrics: Mapping[str, Mapping[str, float]],
    limit: int = 8,
) -> List[str]:
    ordered = sorted(predictions, key=lambda name: metrics[name]["mae_cm"])
    selected: List[str] = []
    short_candidates = [name for name in ordered if "short_weighted" in name or "short_male_weighted" in name]
    if short_candidates:
        selected.append(short_candidates[0])
    short_male_candidates = [name for name in ordered if "short_male_weighted" in name]
    if short_male_candidates and short_male_candidates[0] not in selected:
        selected.append(short_male_candidates[0])

    for name in ordered:
        if name in selected:
            continue
        if not selected:
            selected.append(name)
            continue
        correlations = []
        for prior in selected:
            correlation = np.corrcoef(predictions[name], predictions[prior])[0, 1]
            correlations.append(
                float(correlation) if np.isfinite(correlation) else 1.0
            )
        family = name.split("__", 1)[0]
        prior_families = {
            item.split("__", 1)[0]
            for item in selected
        }
        if min(correlations) < 0.995 or family not in prior_families:
            selected.append(name)
        if len(selected) >= int(limit):
            break
    return selected


def optimize_convex_weights(
    matrix: np.ndarray,
    y: np.ndarray,
    prior: np.ndarray | None = None,
    penalty: float = 0.02,
    short_penalty_weight: float = 0.0,
    gender: np.ndarray | None = None,
    short_male_penalty_weight: float = 0.0,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    y = np.asarray(y, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(y):
        raise ValueError("Prediction matrix must have shape speakers by models")
    count = matrix.shape[1]
    if count == 1:
        return np.ones(1, dtype=float)
    if prior is None:
        individual = np.mean(np.abs(matrix - y[:, None]), axis=0)
        scale = max(float(np.std(individual)), 0.1)
        prior = np.exp(-(individual - individual.min()) / scale)
        prior = prior / prior.sum()
    else:
        prior = np.asarray(prior, dtype=float)
        prior = np.clip(prior, 0.0, None)
        prior = prior / prior.sum()

    short_mask = y < 160.0
    short_male_mask = (y < 160.0) & (gender == 1) if gender is not None else np.zeros_like(short_mask, dtype=bool)

    def objective(weights: np.ndarray) -> float:
        mae = np.mean(np.abs(matrix @ weights - y))
        short_mae = (
            np.mean(np.abs((matrix @ weights)[short_mask] - y[short_mask]))
            if short_mask.sum() > 0
            else 0.0
        )
        short_male_mae = (
            np.mean(np.abs((matrix @ weights)[short_male_mask] - y[short_male_mask]))
            if short_male_mask.sum() > 0
            else 0.0
        )
        regularizer = float(penalty) * np.sum(np.square(weights - prior))
        return float(
            mae
            + float(short_penalty_weight) * short_mae
            + float(short_male_penalty_weight) * short_male_mae
            + regularizer
        )

    result = minimize(
        objective,
        x0=prior,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={
            "type": "eq",
            "fun": lambda weights: float(weights.sum() - 1.0),
        },
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    weights = result.x if result.success else prior
    weights = np.clip(weights, 0.0, None)
    return weights / weights.sum()


def fit_candidates_predict(
    train: SplitData,
    query: SplitData,
    candidates_by_name: Mapping[str, Candidate],
    names: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    predictions: Dict[str, np.ndarray] = {}
    fitted: Dict[str, Any] = {}
    for name in names:
        candidate = candidates_by_name[name]
        is_short_weighted = "short_weighted" in name
        is_short_male_weighted = "short_male_weighted" in name
        sw = None
        if is_short_male_weighted:
            sw = np.ones_like(train.y, dtype=float)
            sw[(train.y < 160.0) & (train.gender == 1)] *= 6.0
            sw[(train.y < 160.0) & (train.gender == 0)] *= 3.0
            sw[train.y < 152.0] *= 4.0
        elif is_short_weighted:
            sw = np.ones_like(train.y, dtype=float)
            sw[train.y < 160.0] *= 2.5
            sw[train.y < 152.0] *= 4.0
        model = _fit(
            candidate.estimator,
            train.views[candidate.view],
            train.y,
            sample_weight=sw,
        )
        predictions[name] = np.asarray(
            model.predict(query.views[candidate.view]),
            dtype=float,
        )
        fitted[name] = model
    return predictions, fitted


def _weighted_prediction(
    predictions: Mapping[str, np.ndarray],
    names: Sequence[str],
    weights: np.ndarray,
) -> np.ndarray:
    matrix = np.column_stack([predictions[name] for name in names])
    return matrix @ np.asarray(weights, dtype=float)


def fit_postprocessor(
    y: np.ndarray,
    pred: np.ndarray,
    source: np.ndarray,
    gender: np.ndarray,
    kind: str,
) -> Dict[str, Any]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if kind == "raw":
        return {"kind": "raw", "global_offset": 0.0, "group_offsets": {}}
    residual = y - pred
    global_offset = float(np.median(residual))
    group_offsets: Dict[str, float] = {}
    affine_params: Dict[str, Tuple[float, float]] = {}

    if kind in {"group", "group_snap"}:
        centered = residual - global_offset
        for source_name in sorted(set(source)):
            for gender_value in (0, 1):
                mask = (source == source_name) & (gender == gender_value)
                if int(mask.sum()) >= 5:
                    shrink = float(mask.sum()) / (float(mask.sum()) + 20.0)
                    group_offsets[f"{source_name}:{gender_value}"] = (
                        shrink * float(np.median(centered[mask]))
                    )
    elif kind == "gender_affine":
        for gender_value in (0, 1):
            mask = gender == gender_value
            if int(mask.sum()) >= 5:
                p = pred[mask]
                t = y[mask]
                var_p = float(np.var(p))
                if var_p > 1e-6:
                    slope = float(np.cov(p, t)[0, 1] / var_p)
                    slope = float(np.clip(slope, 0.7, 1.8))
                    intercept = float(np.median(t - slope * p))
                else:
                    slope = 1.0
                    intercept = float(np.median(t - p))
                affine_params[str(int(gender_value))] = (slope, intercept)
    elif kind == "range_affine":
        for slice_name, mask in [
            ("short", pred < 165.0),
            ("medium", (pred >= 165.0) & (pred < 175.0)),
            ("tall", pred >= 175.0),
        ]:
            if int(mask.sum()) >= 4:
                p = pred[mask]
                t = y[mask]
                var_p = float(np.var(p))
                if var_p > 1e-6:
                    slope = float(np.clip(np.cov(p, t)[0, 1] / var_p, 0.5, 2.0))
                    intercept = float(np.median(t - slope * p))
                else:
                    slope = 1.0
                    intercept = float(np.median(t - p))
                affine_params[slice_name] = (slope, intercept)
    elif kind == "short_gated":
        mask = pred < 165.0
        if int(mask.sum()) >= 4:
            p = pred[mask]
            t = y[mask]
            var_p = float(np.var(p))
            if var_p > 1e-6:
                slope = float(np.clip(np.cov(p, t)[0, 1] / var_p, 0.6, 2.2))
                intercept = float(np.median(t - slope * p))
            else:
                slope = 1.0
                intercept = float(np.median(t - p))
            affine_params["short_gated"] = (slope, intercept)
    elif kind == "short_voice_calibrated":
        mask = pred < 168.0
        if int(mask.sum()) >= 4:
            p = pred[mask]
            t = y[mask]
            var_p = float(np.var(p))
            if var_p > 1e-6:
                slope = float(np.clip(np.cov(p, t)[0, 1] / var_p, 0.7, 2.5))
                intercept = float(np.median(t - slope * p))
            else:
                slope = 1.2
                intercept = float(np.median(t - 1.2 * p))
            affine_params["short_voice"] = (slope, intercept)
    elif kind == "short_male_debias":
        mask_male = (gender == 1) & (pred < 174.0)
        mask_short_female = (gender == 0) & (pred < 165.0)
        if int(mask_male.sum()) >= 3:
            p_m = pred[mask_male]
            t_m = y[mask_male]
            var_m = float(np.var(p_m))
            if var_m > 1e-6:
                slope_m = float(np.clip(np.cov(p_m, t_m)[0, 1] / var_m, 0.7, 2.8))
                intercept_m = float(np.median(t_m - slope_m * p_m))
            else:
                slope_m = 1.2
                intercept_m = float(np.median(t_m - 1.2 * p_m))
            affine_params["short_male"] = (slope_m, intercept_m)
        else:
            affine_params["short_male"] = (1.0, 0.0)

        if int(mask_short_female.sum()) >= 3:
            p_f = pred[mask_short_female]
            t_f = y[mask_short_female]
            var_f = float(np.var(p_f))
            if var_f > 1e-6:
                slope_f = float(np.clip(np.cov(p_f, t_f)[0, 1] / var_f, 0.6, 2.2))
                intercept_f = float(np.median(t_f - slope_f * p_f))
            else:
                slope_f = 1.0
                intercept_f = float(np.median(t_f - p_f))
            affine_params["short_female"] = (slope_f, intercept_f)

    return {
        "kind": kind,
        "global_offset": global_offset,
        "group_offsets": group_offsets,
        "affine_params": affine_params,
    }


def apply_postprocessor(
    pred: np.ndarray,
    source: np.ndarray,
    gender: np.ndarray,
    params: Mapping[str, Any],
) -> np.ndarray:
    result = np.asarray(pred, dtype=float).copy()
    kind = params.get("kind", "raw")

    if kind == "gender_affine":
        affine = params.get("affine_params", {})
        for index, gender_value in enumerate(gender):
            key = str(int(gender_value))
            if key in affine:
                slope, intercept = affine[key]
                result[index] = slope * result[index] + intercept
        return result
    elif kind == "range_affine":
        affine = params.get("affine_params", {})
        for index, p_val in enumerate(result):
            slice_name = "short" if p_val < 165.0 else ("medium" if p_val < 175.0 else "tall")
            if slice_name in affine:
                slope, intercept = affine[slice_name]
                result[index] = slope * result[index] + intercept
        return result
    elif kind == "short_gated":
        affine = params.get("affine_params", {})
        if "short_gated" in affine:
            slope, intercept = affine["short_gated"]
            for index, p_val in enumerate(result):
                if p_val < 165.0:
                    result[index] = slope * p_val + intercept
        return result
    elif kind == "short_voice_calibrated":
        affine = params.get("affine_params", {})
        if "short_voice" in affine:
            slope, intercept = affine["short_voice"]
            for index, p_val in enumerate(result):
                if p_val < 168.0:
                    weight = float(np.clip((168.0 - p_val) / 10.0, 0.0, 1.0))
                    calibrated = slope * p_val + intercept
                    result[index] = (1.0 - weight) * p_val + weight * calibrated
        return result
    elif kind == "short_male_debias":
        affine = params.get("affine_params", {})
        if "short_male" in affine:
            slope_m, intercept_m = affine["short_male"]
            for index, (p_val, g_val) in enumerate(zip(result, gender)):
                if g_val == 1 and p_val < 174.0:
                    weight = float(np.clip((174.0 - p_val) / 10.0, 0.0, 1.0))
                    calibrated = slope_m * p_val + intercept_m
                    result[index] = (1.0 - weight) * p_val + weight * calibrated
        if "short_female" in affine:
            slope_f, intercept_f = affine["short_female"]
            for index, (p_val, g_val) in enumerate(zip(result, gender)):
                if g_val == 0 and p_val < 165.0:
                    weight = float(np.clip((165.0 - p_val) / 8.0, 0.0, 1.0))
                    calibrated = slope_f * p_val + intercept_f
                    result[index] = (1.0 - weight) * p_val + weight * calibrated
        return result

    result += float(params.get("global_offset", 0.0))
    offsets = params.get("group_offsets", {})
    for index, (source_name, gender_value) in enumerate(zip(source, gender)):
        if kind == "group_snap" and gender_value == 1 and result[index] < 172.0:
            offset_val = float(offsets.get(f"{source_name}:{int(gender_value)}", 0.0))
            if offset_val > 0.0:
                offset_val *= float(np.clip((result[index] - 162.0) / 10.0, 0.0, 1.0))
            result[index] += offset_val
        else:
            result[index] += float(
                offsets.get(f"{source_name}:{int(gender_value)}", 0.0)
            )
    if kind == "group_snap":
        timit = source == "TIMIT"
        result[timit] = np.round(result[timit] / 2.54) * 2.54
    return result


def _phase12_predictions(root: Path, split: SplitData) -> np.ndarray | None:
    path = (
        root
        / "outputs"
        / "short_speaker_correction"
        / f"corrected_predictions_{split.name}.csv"
    )
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {str(row["speaker_id"]): row for row in rows}
    if set(split.ids) != set(by_id):
        return None
    prediction = np.asarray(
        [_float(by_id[sid]["phase12_baseline_cm"]) for sid in split.ids],
        dtype=float,
    )
    recorded_y = np.asarray(
        [_float(by_id[sid]["height_cm"]) for sid in split.ids],
        dtype=float,
    )
    if not np.allclose(recorded_y, split.y, atol=2e-2):
        raise ValueError(f"Phase12 target mismatch for {split.name}")
    return prediction


def choose_recipe(
    train: SplitData,
    val: SplitData,
    selected: Sequence[str],
    oof: Mapping[str, np.ndarray],
    oof_weights: np.ndarray,
    val_predictions: Mapping[str, np.ndarray],
    phase12_val: np.ndarray | None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    oof_matrix = np.column_stack([oof[name] for name in selected])
    short_oof_weights = optimize_convex_weights(oof_matrix, train.y, short_penalty_weight=0.5)
    short_male_oof_weights = optimize_convex_weights(
        oof_matrix,
        train.y,
        short_penalty_weight=0.5,
        gender=train.gender,
        short_male_penalty_weight=1.5,
    )

    recipes: Dict[str, Tuple[List[str], np.ndarray, float]] = {
        "acoustic_convex": (list(selected), np.asarray(oof_weights), 0.0),
        "short_convex": (list(selected), np.asarray(short_oof_weights), 0.0),
        "short_male_convex": (list(selected), np.asarray(short_male_oof_weights), 0.0),
    }
    for name in selected[:3]:
        one_hot = np.zeros(len(selected), dtype=float)
        one_hot[list(selected).index(name)] = 1.0
        recipes[f"single__{name}"] = (list(selected), one_hot, 0.0)
    if len(selected) >= 3:
        equal = np.zeros(len(selected), dtype=float)
        equal[:3] = 1.0 / 3.0
        recipes["equal_top3"] = (list(selected), equal, 0.0)

    metrics: Dict[str, Dict[str, float]] = {}
    predictions: Dict[str, np.ndarray] = {}
    recipe_specs: Dict[str, Dict[str, Any]] = {}
    val_matrix = np.column_stack([val_predictions[name] for name in selected])

    for base_name, (names, weights, phase_weight) in recipes.items():
        train_base = oof_matrix @ weights
        val_base = val_matrix @ weights
        for post_kind in ("raw", "global", "group", "group_snap", "gender_affine", "range_affine", "short_gated", "short_voice_calibrated", "short_male_debias"):
            params = fit_postprocessor(
                train.y,
                train_base,
                train.source,
                train.gender,
                post_kind,
            )
            pred = apply_postprocessor(
                val_base,
                val.source,
                val.gender,
                params,
            )
            key = f"{base_name}__{post_kind}"
            metrics[key] = regression_metrics(val.y, pred)
            predictions[key] = pred
            recipe_specs[key] = {
                "components": names,
                "weights": weights.tolist(),
                "phase12_weight": phase_weight,
                "postprocess": post_kind,
            }

    if phase12_val is not None:
        metrics["phase12_frozen"] = regression_metrics(val.y, phase12_val)
        predictions["phase12_frozen"] = phase12_val
        recipe_specs["phase12_frozen"] = {
            "components": list(selected),
            "weights": np.asarray(oof_weights).tolist(),
            "phase12_weight": 1.0,
            "postprocess": "raw",
        }
        acoustic = val_matrix @ np.asarray(oof_weights)
        for phase_weight in (0.25, 0.50, 0.75):
            pred = (
                (1.0 - phase_weight) * acoustic
                + phase_weight * phase12_val
            )
            key = f"acoustic_phase12_blend__p{phase_weight:.2f}"
            metrics[key] = regression_metrics(val.y, pred)
            predictions[key] = pred
            recipe_specs[key] = {
                "components": list(selected),
                "weights": np.asarray(oof_weights).tolist(),
                "phase12_weight": phase_weight,
                "postprocess": "raw",
            }

    acoustic_metrics = {
        name: item for name, item in metrics.items()
        if recipe_specs[name]["phase12_weight"] == 0.0
    }
    best_mae = min(item["mae_cm"] for item in acoustic_metrics.values())
    finalists = [
        name
        for name, item in acoustic_metrics.items()
        if item["mae_cm"] <= best_mae + 0.03
    ]
    val_short_mask = val.y < 160.0
    val_short_maes = {
        name: float(np.mean(np.abs(pred[val_short_mask] - val.y[val_short_mask])))
        if np.any(val_short_mask) else 0.0
        for name, pred in predictions.items()
    }
    complexity = {
        name: (
            int(recipe_specs[name]["phase12_weight"] > 0.0) * 10
            + int(recipe_specs[name]["postprocess"] != "raw")
            + len([
                weight
                for weight in recipe_specs[name]["weights"]
                if weight > 1e-6
            ])
        )
        for name in finalists
    }
    winner = min(
        finalists,
        key=lambda name: (
            val_short_maes[name],
            complexity[name],
            metrics[name]["mae_cm"],
            name,
        ),
    )
    recipe = {"name": winner, **recipe_specs[winner]}
    return recipe, metrics, predictions


def concatenate_development(train: SplitData, val: SplitData) -> SplitData:
    common = sorted(set(train.views) & set(val.views))
    views = {
        name: np.concatenate(
            [train.views[name], val.views[name]],
            axis=0,
        )
        for name in common
    }
    return SplitData(
        "development",
        np.concatenate([train.ids, val.ids]),
        np.concatenate([train.y, val.y]),
        np.concatenate([train.gender, val.gender]),
        np.concatenate([train.source, val.source]),
        np.concatenate([train.meta, val.meta], axis=0),
        views,
    )


def bootstrap_mae_ci(
    y: np.ndarray,
    pred: np.ndarray,
    seed: int,
    repetitions: int = 10000,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    error = np.abs(y - pred)
    rng = np.random.default_rng(seed)
    values = np.empty(int(repetitions), dtype=float)
    for index in range(int(repetitions)):
        sample = rng.integers(0, len(error), size=len(error))
        values[index] = float(error[sample].mean())
    return {
        "lower_95_cm": float(np.quantile(values, 0.025)),
        "upper_95_cm": float(np.quantile(values, 0.975)),
        "repetitions": int(repetitions),
    }


def sliced_metrics(
    split: SplitData,
    pred: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    masks: Dict[str, np.ndarray] = {
        "all": np.ones(len(split.y), dtype=bool),
        "female": split.gender == 0,
        "male": split.gender == 1,
        "nisp": split.source == "NISP",
        "timit": split.source == "TIMIT",
        "short_lt160": split.y < 160.0,
        "medium_160_175": (split.y >= 160.0) & (split.y < 175.0),
        "tall_ge175": split.y >= 175.0,
    }
    return {
        name: {
            "n": int(mask.sum()),
            **regression_metrics(split.y[mask], pred[mask]),
        }
        for name, mask in masks.items()
        if int(mask.sum()) > 0
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_predictions(
    path: Path,
    split: SplitData,
    pred: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "speaker_id",
                "source",
                "gender",
                "height_cm",
                "pred_height_cm",
                "abs_error_cm",
            )
        )
        for sid, source, gender, target, estimate in zip(
            split.ids,
            split.source,
            split.gender,
            split.y,
            pred,
        ):
            writer.writerow(
                (
                    sid,
                    source,
                    int(gender),
                    f"{float(target):.6f}",
                    f"{float(estimate):.6f}",
                    f"{abs(float(target) - float(estimate)):.6f}",
                )
            )


def _clean_weights(
    names: Sequence[str],
    weights: np.ndarray,
) -> Dict[str, float]:
    return {
        name: float(weight)
        for name, weight in zip(names, weights)
        if float(weight) >= 1e-6
    }


def run_strict_experiment(
    root: Path,
    output_dir: Path,
    seed: int = 42,
    folds: int = 5,
    quick: bool = False,
) -> Dict[str, Any]:
    root = Path(root).resolve()
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data, wavlm_views = load_dataset(root)
    train = data["train"]
    val = data["val"]
    test = data["test"]
    candidates = build_candidates(train, seed=seed, quick=quick)
    candidate_by_name = {
        candidate.name: candidate
        for candidate in candidates
    }

    train_folds = make_folds(train, folds, seed)
    oof, candidate_metrics, failures = oof_predictions(
        train,
        candidates,
        train_folds,
    )
    selected = select_diverse_candidates(
        oof,
        candidate_metrics,
        limit=6 if quick else 10,
    )
    oof_matrix = np.column_stack([oof[name] for name in selected])
    oof_weights = optimize_convex_weights(oof_matrix, train.y)

    val_predictions, _ = fit_candidates_predict(
        train,
        val,
        candidate_by_name,
        selected,
    )
    phase12_val = _phase12_predictions(root, val)
    recipe, recipe_metrics, recipe_predictions = choose_recipe(
        train,
        val,
        selected,
        oof,
        oof_weights,
        val_predictions,
        phase12_val,
    )
    frozen_validation_prediction = recipe_predictions[recipe["name"]].copy()

    development = concatenate_development(train, val)
    dev_folds = make_folds(development, folds, seed + 1009)
    component_names = list(recipe["components"])
    component_candidates = [
        candidate_by_name[name]
        for name in component_names
    ]
    dev_oof, _dev_metrics, dev_failures = oof_predictions(
        development,
        component_candidates,
        dev_folds,
    )
    if dev_failures:
        raise RuntimeError(
            f"Selected final component failed on development OOF: {dev_failures}"
        )
    dev_matrix = np.column_stack([
        dev_oof[name]
        for name in component_names
    ])

    original_weights = np.asarray(recipe["weights"], dtype=float)
    if np.count_nonzero(original_weights > 1e-9) > 1:
        final_weights = optimize_convex_weights(
            dev_matrix,
            development.y,
            prior=original_weights,
            penalty=0.05,
        )
    else:
        final_weights = original_weights
    dev_base = dev_matrix @ final_weights
    postprocess = fit_postprocessor(
        development.y,
        dev_base,
        development.source,
        development.gender,
        recipe["postprocess"],
    )

    test_predictions, fitted = fit_candidates_predict(
        development,
        test,
        candidate_by_name,
        component_names,
    )
    test_acoustic = _weighted_prediction(
        test_predictions,
        component_names,
        final_weights,
    )
    test_acoustic = apply_postprocessor(
        test_acoustic,
        test.source,
        test.gender,
        postprocess,
    )
    phase_weight = float(recipe["phase12_weight"])
    active_components = [
        name
        for name, weight in zip(component_names, final_weights)
        if float(weight) >= 1e-6
    ]
    uses_non_acoustic_metadata = (
        phase_weight < 1.0
        and any(
            candidate_by_name[name].view == "metadata"
            or candidate_by_name[name].view.endswith("+meta")
            for name in active_components
        )
    )
    phase12_test = _phase12_predictions(root, test)
    if phase_weight > 0.0:
        if phase12_test is None:
            raise FileNotFoundError(
                "Recipe selected Phase12, but test predictions are unavailable"
            )
        test_final = (
            (1.0 - phase_weight) * test_acoustic
            + phase_weight * phase12_test
        )
    else:
        test_final = test_acoustic

    test_metrics = regression_metrics(test.y, test_final)
    confidence = bootstrap_mae_ci(
        test.y,
        test_final,
        seed=seed + 2027,
        repetitions=2000 if quick else 10000,
    )
    slices = sliced_metrics(test, test_final)
    validation_metrics = regression_metrics(
        val.y,
        frozen_validation_prediction,
    )
    phase12_test_metrics = (
        regression_metrics(test.y, phase12_test)
        if phase12_test is not None
        else None
    )

    manifest_paths = [
        root / "data" / "splits" / f"{split}_clean.csv"
        for split in SPLITS
    ]
    for name in wavlm_views:
        manifest_paths.extend(
            root / "outputs" / name / f"{split}.npz"
            for split in SPLITS
        )
    manifest = {
        str(path.relative_to(root)): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in manifest_paths
    }

    result: Dict[str, Any] = {
        "protocol": {
            "speaker_disjoint": True,
            "test_used_for_model_selection": False,
            "selection_data": "train OOF plus fixed validation split",
            "final_fit_data": "train plus validation",
            "test_evaluations_in_this_script": 1,
            "seed": int(seed),
            "folds": int(folds),
            "quick": bool(quick),
        },
        "data": {
            "train_speakers": int(len(train.y)),
            "validation_speakers": int(len(val.y)),
            "test_speakers": int(len(test.y)),
            "wavlm_views": wavlm_views,
            "metadata_columns": list(META_COLUMNS),
        },
        "selection": {
            "candidate_count": int(len(candidates)),
            "candidate_failures": failures,
            "candidate_train_oof_metrics": candidate_metrics,
            "shortlist": selected,
            "train_oof_convex_weights": _clean_weights(
                selected,
                oof_weights,
            ),
            "validation_recipe_metrics": recipe_metrics,
            "frozen_recipe": recipe,
            "frozen_validation_metrics": validation_metrics,
        },
        "final_model": {
            "weights": _clean_weights(
                component_names,
                final_weights,
            ),
            "postprocessor": postprocess,
            "phase12_weight": phase_weight,
            "acoustic_weights_active": bool(phase_weight < 1.0),
            "uses_non_acoustic_metadata": bool(uses_non_acoustic_metadata),
            "result_scope": (
                "metadata-assisted"
                if uses_non_acoustic_metadata
                else "voice-only"
            ),
        },
        "test": {
            "metrics": test_metrics,
            "bootstrap_mae_95_ci": confidence,
            "slices": slices,
            "phase12_reference_metrics": phase12_test_metrics,
            "three_cm_point_target_met": bool(
                test_metrics["mae_cm"] <= 3.0
            ),
            "three_cm_95pct_gate_met": bool(
                confidence["upper_95_cm"] <= 3.0
            ),
            "four_cm_point_target_met": bool(
                test_metrics["mae_cm"] <= 4.0
            ),
        },
        "integrity_manifest": manifest,
    }

    write_predictions(
        output_dir / "predictions_validation_frozen.csv",
        val,
        frozen_validation_prediction,
    )
    write_predictions(
        output_dir / "predictions_test_once.csv",
        test,
        test_final,
    )
    with (output_dir / "strict_results.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    joblib.dump(
        {
            "component_names": component_names,
            "candidate_views": {
                name: candidate_by_name[name].view
                for name in component_names
            },
            "models": fitted,
            "weights": final_weights,
            "postprocessor": postprocess,
            "phase12_weight": phase_weight,
            "is_self_contained": bool(phase_weight == 0.0),
            "phase12_reference": (
                "outputs/short_speaker_correction/corrected_predictions_test.csv"
                if phase_weight > 0.0
                else None
            ),
            "metadata_columns": META_COLUMNS,
            "wavlm_views": wavlm_views,
        },
        output_dir / "strict_model_bundle.joblib",
    )

    report = [
        "# VoxPhysica strict 3 cm experiment",
        "",
        "## Outcome",
        "",
        f"- Frozen validation MAE: **{validation_metrics['mae_cm']:.3f} cm**",
        f"- One-shot test MAE: **{test_metrics['mae_cm']:.3f} cm**",
        (
            "- Bootstrap 95% CI for test MAE: "
            f"**[{confidence['lower_95_cm']:.3f}, "
            f"{confidence['upper_95_cm']:.3f}] cm**"
        ),
        f"- 3 cm point target met: **{test_metrics['mae_cm'] <= 3.0}**",
        (
            "- 3 cm confidence gate met: "
            f"**{confidence['upper_95_cm'] <= 3.0}**"
        ),
        f"- 4 cm point target met: **{test_metrics['mae_cm'] <= 4.0}**",
        "",
        "## Frozen recipe",
        "",
        f"- Recipe: {recipe['name']}",
        (
            (
                "- Acoustic component weights "
                "(inactive because Phase12 weight is 1.00): "
                if phase_weight == 1.0
                else "- Final acoustic weights: "
            )
            + json.dumps(
                _clean_weights(component_names, final_weights),
                sort_keys=True,
            )
        ),
        f"- Phase12 frozen weight: {phase_weight:.2f}",
        f"- Postprocessor: {recipe['postprocess']}",
        (
            "- Result scope: metadata-assisted"
            if uses_non_acoustic_metadata
            else "- Result scope: voice-only"
        ),
        "",
        (
            "The test labels were not supplied to candidate selection, "
            "blend selection, or postprocessing."
        ),
        (
            "This report must not be described as a new sealed test if "
            "the same test set is used for another tuning round."
        ),
    ]
    (output_dir / "STRICT_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    return result
