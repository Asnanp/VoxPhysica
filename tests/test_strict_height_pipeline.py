import numpy as np
import pytest

from src.research.strict_height_pipeline import (
    SplitData,
    apply_postprocessor,
    assert_disjoint_splits,
    make_folds,
    optimize_convex_weights,
)


def _split(name: str, ids: list[str]) -> SplitData:
    count = len(ids)
    gender = np.asarray([index % 2 for index in range(count)], dtype=np.float32)
    source = np.asarray([
        "NISP" if (index // 2) % 2 == 0 else "TIMIT"
        for index in range(count)
    ])
    meta = np.zeros((count, 20), dtype=np.float32)
    meta[:, 0] = gender
    meta[:, 1] = source == "NISP"
    y = 158.0 + 12.0 * gender + np.linspace(-2.0, 2.0, count)
    return SplitData(
        name=name,
        ids=np.asarray(ids),
        y=np.asarray(y, dtype=np.float32),
        gender=gender,
        source=source,
        meta=meta,
        views={"metadata": meta},
    )


def test_assert_disjoint_splits_rejects_speaker_overlap():
    data = {
        "train": _split("train", ["a", "b", "c", "d"]),
        "val": _split("val", ["e", "f", "g", "h"]),
        "test": _split("test", ["a", "i", "j", "k"]),
    }

    with pytest.raises(ValueError, match="Speaker leakage"):
        assert_disjoint_splits(data)


def test_make_folds_is_deterministic_and_covers_each_speaker_once():
    split = _split("train", [f"speaker_{index:02d}" for index in range(40)])

    first = make_folds(split, n_splits=5, seed=17)
    second = make_folds(split, n_splits=5, seed=17)

    held_out = np.concatenate([holdout for _, holdout in first])
    assert sorted(held_out.tolist()) == list(range(40))
    assert [
        (train.tolist(), holdout.tolist())
        for train, holdout in first
    ] == [
        (train.tolist(), holdout.tolist())
        for train, holdout in second
    ]
    for train, holdout in first:
        assert not set(train) & set(holdout)


def test_convex_weight_optimizer_is_bounded_and_improves_complements():
    target = np.asarray([150.0, 160.0, 170.0, 180.0])
    matrix = np.column_stack([target - 2.0, target + 2.0])

    weights = optimize_convex_weights(matrix, target, penalty=0.0)
    blended_mae = np.mean(np.abs(matrix @ weights - target))

    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(1.0)
    assert blended_mae < 1e-5


def test_legacy_final_entrypoint_routes_to_strict_pipeline():
    import scripts.final_ensemble as entrypoint

    assert entrypoint.main.__module__ == "scripts.run_strict_3cm_research"


def test_timit_grid_snap_does_not_change_nisp_prediction():
    pred = np.asarray([170.10, 170.10])
    source = np.asarray(["TIMIT", "NISP"])
    gender = np.asarray([1.0, 1.0])
    params = {
        "kind": "group_snap",
        "global_offset": 0.0,
        "group_offsets": {},
    }

    corrected = apply_postprocessor(pred, source, gender, params)

    assert corrected[0] == pytest.approx(170.18)
    assert corrected[1] == pytest.approx(170.10)
