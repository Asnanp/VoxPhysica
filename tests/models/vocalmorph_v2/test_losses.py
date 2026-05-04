import torch

from src.models.vocalmorph_v2 import SpeakerAlignmentConfig, VocalTractSimulatorLossV2


def test_height_focal_weighting_emphasizes_hard_residuals():
    preds = {
        "height_mu": torch.tensor([0.0, 0.0]),
        "height_logvar": torch.zeros(2),
    }
    targets = {
        "height": torch.tensor([0.1, 2.0]),
        "epoch": 1,
    }

    plain = VocalTractSimulatorLossV2(focal_after_epoch=99, focal_gamma=2.0)
    focal = VocalTractSimulatorLossV2(focal_after_epoch=1, focal_gamma=2.0)
    plain_loss, _ = plain._height_loss(preds, targets, torch.device("cpu"))
    focal_loss, _ = focal._height_loss(preds, targets, torch.device("cpu"))

    assert focal_loss > plain_loss


def test_height_distribution_loss_penalizes_compressed_predictions():
    targets = {
        "height": torch.tensor([-2.0, -1.2, -0.4, 0.4, 1.2, 2.0]),
        "epoch": 1,
    }
    compressed_preds = {
        "height_mu": torch.tensor([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]),
        "height_logvar": torch.zeros(6),
    }
    matched_preds = {
        "height_mu": targets["height"].clone(),
        "height_logvar": torch.zeros(6),
    }
    alignment = SpeakerAlignmentConfig(
        height_distribution_weight_max=1.0,
        height_distribution_min_items=4,
    )
    loss_fn = VocalTractSimulatorLossV2(speaker_alignment=alignment)

    compressed = loss_fn._height_distribution_loss(
        compressed_preds, targets, torch.device("cpu")
    )
    matched = loss_fn._height_distribution_loss(
        matched_preds, targets, torch.device("cpu")
    )

    assert compressed > matched
    assert compressed > 0


def test_height_loss_gender_tail_weights_emphasize_rare_short_male_errors():
    preds = {
        "height_mu": torch.tensor([0.0, 0.0, 0.0]),
        "height_logvar": torch.zeros(3),
    }
    targets = {
        "height": torch.tensor([3.0, 0.1, 0.1]),
        "height_raw": torch.tensor([151.0, 168.0, 182.0]),
        "gender": torch.tensor([1, 0, 1]),
        "epoch": 1,
    }

    plain = VocalTractSimulatorLossV2(
        focal_after_epoch=99,
        speaker_alignment=SpeakerAlignmentConfig(height_bin_loss_start_epoch=99),
    )
    weighted = VocalTractSimulatorLossV2(
        focal_after_epoch=99,
        speaker_alignment=SpeakerAlignmentConfig(
            height_bin_loss_start_epoch=1,
            height_bin_loss_weight_short=2.0,
            height_bin_loss_weight_male_short=3.0,
            height_extreme_short_cm=152.0,
            height_extreme_loss_weight_short=2.0,
        ),
    )

    plain_loss, _ = plain._height_loss(preds, targets, torch.device("cpu"))
    weighted_loss, _ = weighted._height_loss(preds, targets, torch.device("cpu"))

    assert weighted_loss > plain_loss


def test_height_bin_loss_uses_raw_height_bins():
    preds = {
        "height_mu": torch.zeros(3),
        "height_logvar": torch.zeros(3),
        "height_bin_logits": torch.tensor(
            [
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ),
    }
    targets = {
        "height": torch.zeros(3),
        "height_raw": torch.tensor([151.0, 168.0, 182.0]),
        "epoch": 1,
    }
    loss_fn = VocalTractSimulatorLossV2()

    good = loss_fn._height_bin_loss(preds, targets, torch.device("cpu"))
    bad_preds = dict(preds)
    bad_preds["height_bin_logits"] = torch.flip(preds["height_bin_logits"], dims=[0])
    bad = loss_fn._height_bin_loss(bad_preds, targets, torch.device("cpu"))

    assert good < bad
