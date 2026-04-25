import numpy as np

from src.preprocessing.dataset import VocalMorphDataset


def _write_feature(path, time_steps: int = 10, input_dim: int = 136):
    seq = np.stack([np.full((input_dim,), float(t), dtype=np.float32) for t in range(time_steps)], axis=0)
    np.savez(
        path,
        sequence=seq,
        height_cm=np.float32(172.0),
        weight_kg=np.float32(70.0),
        age=np.float32(30.0),
        gender=np.int64(1),
        speaker_id=np.array("spk_a", dtype=object),
        source=np.array("NISP", dtype=object),
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
