"""
tests/test_model_shapes.py
===========================
Automated test suite verifying tensor dimensionality, gradient flow,
and forward pass contracts across the 3D vision pipeline.
"""

import pytest
import torch
from src.models.swin_unetr_3d import SwinUNETR3D
from src.models.mae_3d import MaskedAutoencoder3D
from src.models.losses import VolumetricSegmentationLoss
from src.utils.metrics import VolumetricSegmentationMetrics


def test_swin_unetr_forward_shape():
    """Verifies that 3D Swin-UNETR preserves 3D spatial resolution."""
    model = SwinUNETR3D(
        in_channels=4,
        out_channels=4,
        feature_size=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        use_checkpoint=False,
    )
    x = torch.randn(1, 4, 64, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 4, 64, 64, 64), f"Expected (1, 4, 64, 64, 64), got {out.shape}"


def test_mae_3d_masking_ratio():
    """Verifies that 3D-MAE randomly masks the expected proportion of 3D tokens."""
    mae = MaskedAutoencoder3D(
        in_channels=4,
        img_size=(32, 32, 32),
        patch_size=(8, 8, 8),
        embed_dim=64,
        encoder_depth=2,
        num_heads=2,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=2,
        mask_ratio=0.75,
    )
    x = torch.randn(2, 4, 32, 32, 32)
    loss, pred, mask = mae(x)
    assert mask.shape == (2, mae.num_patches)
    assert torch.isclose(mask.float().mean(), torch.tensor(0.75), atol=0.05)
    assert loss.item() > 0


def test_compound_loss_backward():
    """Verifies gradient flow through compound Dice-Cross-Entropy loss."""
    pred = torch.randn(2, 4, 16, 16, 16, requires_grad=True)
    target = torch.randint(0, 4, (2, 1, 16, 16, 16), dtype=torch.int64)

    loss_fn = VolumetricSegmentationLoss(include_background=False, to_onehot_y=True, softmax=True)
    loss = loss_fn(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()


def test_segmentation_dice_metric():
    """Verifies that perfect predictions achieve Dice = 1.0."""
    metrics = VolumetricSegmentationMetrics(include_background=False, compute_hd95=False)
    # Perfect match: identical one-hot tensors
    pred = torch.zeros(1, 4, 16, 16, 16)
    pred[:, 1, 4:12, 4:12, 4:12] = 1.0
    target = pred.clone()

    metrics.update(pred, target)
    score = metrics.compute()
    assert pytest.approx(score["val_mean_dice"], 0.01) == 1.0
