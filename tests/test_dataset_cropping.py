import numpy as np
from torch.utils.data import WeightedRandomSampler

from src.preprocessing.dataset import (
    VocalMorphDataset,
    _sample_weight_from_npz,
    build_dataloaders_from_dirs,
)


def _write_feature(
    path,
    time_steps: int = 10,
    input_dim: int = 136,
    height_cm: float = 172.0,
    gender: int = 1,
    speaker_id: str = "spk_a",
    source: str = "NISP",
):
    seq = np.stack(
        [np.full((input_dim,), float(t), dtype=np.float32) for t in range(time_steps)],
        axis=0,
    )
    np.savez(
        path,
        sequence=seq,
        height_cm=np.float32(height_cm),
        weight_kg=np.float32(70.0),
        age=np.float32(30.0),
        gender=np.int64(gender),
        speaker_id=np.array(speaker_id, dtype=object),
        source=np.array(source, dtype=object),
        f0_mean=np.float32(140.0),
        formant_spacing_mean=np.float32(700.0),
        vtl_mean=np.float32(16.5),
    )


def test_dataset_crop_modes_apply_expected_window(tmp_path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    _write_feature(feature_dir / "sample.npz")

    head_ds = VocalMorphDataset(str(feature_dir), max_len=4, crop_mode="head")
    center_ds = VocalMorphDataset(str(feature_dir), max_len=4, crop_mode="center")

    np.random.seed(0)
    random_ds = VocalMorphDataset(str(feature_dir), max_len=4, crop_mode="random")

    head_seq = head_ds[0]["sequence"].numpy()
    center_seq = center_ds[0]["sequence"].numpy()
    random_seq = random_ds[0]["sequence"].numpy()

    assert head_seq.shape[0] == 4
    assert center_seq.shape[0] == 4
    assert random_seq.shape[0] == 4

    assert np.allclose(head_seq[:, 0], np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(center_seq[:, 0], np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32))
    assert np.allclose(random_seq[:, 0], np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))


def test_sample_weighting_prioritizes_rare_short_male_examples(tmp_path):
    short_path = tmp_path / "short_male.npz"
    medium_path = tmp_path / "medium_female.npz"
    _write_feature(
        short_path,
        height_cm=151.0,
        gender=1,
        speaker_id="NISP_Hin_short",
        source="NISP",
    )
    _write_feature(
        medium_path,
        height_cm=168.0,
        gender=0,
        speaker_id="NISP_Hin_medium",
        source="NISP",
    )
    config = {
        "height_bin_weights": {"short": 3.0, "medium": 1.0},
        "gender_height_weights": {"male_short": 8.0},
        "source_height_weights": {"nisp_short": 1.5},
        "extreme_short_cm": 152.0,
        "extreme_short_weight": 8.0,
        "max_weight": 30.0,
    }

    assert _sample_weight_from_npz(str(short_path), config) == 30.0
    assert _sample_weight_from_npz(str(medium_path), config) == 1.0


def test_build_dataloaders_uses_weighted_sampler_when_enabled(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    test_dir = tmp_path / "test"
    for feature_dir in (train_dir, val_dir, test_dir):
        feature_dir.mkdir()
        _write_feature(feature_dir / "sample_a.npz", height_cm=151.0, speaker_id="a")
        _write_feature(feature_dir / "sample_b.npz", height_cm=170.0, speaker_id="b")

    train_loader, _, _ = build_dataloaders_from_dirs(
        str(train_dir),
        str(val_dir),
        str(test_dir),
        batch_size=2,
        num_workers=0,
        sample_weighting={
            "enabled": True,
            "height_bin_weights": {"short": 3.0, "medium": 1.0},
        },
        pin_memory=False,
    )

    assert isinstance(train_loader.sampler, WeightedRandomSampler)
