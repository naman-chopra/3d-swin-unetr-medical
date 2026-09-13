"""
losses.py - Compound Loss Functions for Volumetric Segmentation
==============================================================
Implements robust loss formulations to solve extreme class imbalance in 3D
medical scans: Compound Dice-Cross-Entropy Loss and Focal Tversky Loss.
"""

import torch
import torch.nn as nn
from monai.losses import DiceCELoss, DiceFocalLoss, GeneralizedDiceFocalLoss


class VolumetricSegmentationLoss(nn.Module):
    """
    Compound loss function combining Soft Dice Loss with Cross-Entropy / Focal Loss.
    Ensures stable gradient dynamics for thin boundaries and tiny lesion volumes.
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = True,
        sigmoid: bool = False,
        softmax: bool = True,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        focal_gamma: float = 2.0,
        loss_type: str = "dice_ce",  # "dice_ce" or "dice_focal"
    ):
        super().__init__()
        self.loss_type = loss_type

        if loss_type == "dice_ce":
            self.loss_fn = DiceCELoss(
                include_background=include_background,
                to_onehot_y=to_onehot_y,
                sigmoid=sigmoid,
                softmax=softmax,
                lambda_dice=lambda_dice,
                lambda_ce=lambda_ce,
                batch=True,
            )
        elif loss_type == "dice_focal":
            self.loss_fn = DiceFocalLoss(
                include_background=include_background,
                to_onehot_y=to_onehot_y,
                sigmoid=sigmoid,
                softmax=softmax,
                lambda_dice=lambda_dice,
                lambda_focal=lambda_ce,
                gamma=focal_gamma,
                batch=True,
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes compound volumetric loss.
        pred:   (B, Classes, H, W, D) logits
        target: (B, 1, H, W, D) integer ground truth masks
        """
        return self.loss_fn(pred, target)
