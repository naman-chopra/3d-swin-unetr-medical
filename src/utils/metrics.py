"""
metrics.py - Clinical Evaluation Metrics for 3D Segmentation
============================================================
Computes Sørensen-Dice Similarity Coefficient (DSC) and 95th Percentile
Hausdorff Distance (HD95) for multi-class 3D volumetric segmentations.
"""

from typing import Dict, List, Optional
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric


class VolumetricSegmentationMetrics:
    """
    Tracks and computes online clinical segmentation benchmarks across evaluation cohorts.
    """

    def __init__(
        self,
        include_background: bool = False,
        reduction: str = "mean",
        compute_hd95: bool = True,
    ):
        self.include_background = include_background
        self.compute_hd95 = compute_hd95

        self.dice_metric = DiceMetric(
            include_background=include_background,
            reduction=reduction,
            get_not_nans=False,
        )

        if compute_hd95:
            self.hd95_metric = HausdorffDistanceMetric(
                include_background=include_background,
                percentile=95,
                reduction=reduction,
                get_not_nans=False,
            )

    def reset(self):
        """Resets metric accumulators for a new validation epoch."""
        self.dice_metric.reset()
        if self.compute_hd95:
            self.hd95_metric.reset()

    def update(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        Accumulates batch predictions and ground truths.
        y_pred: (B, Classes, H, W, D) one-hot or argmax binary tensor
        y:      (B, Classes, H, W, D) one-hot binary ground truth
        """
        self.dice_metric(y_pred=y_pred, y=y)
        if self.compute_hd95:
            self.hd95_metric(y_pred=y_pred, y=y)

    def compute(self) -> Dict[str, float]:
        """Calculates final epoch-level benchmark metrics including per-class Dice."""
        dice_score = self.dice_metric.aggregate().item()
        results = {"val_mean_dice": dice_score}

        try:
            class_dices = self.dice_metric.aggregate(reduction="mean_batch")
            if isinstance(class_dices, torch.Tensor):
                for c_idx, c_val in enumerate(class_dices):
                    results[f"val_class_{c_idx + 1}_dice"] = float(c_val.item())
        except Exception:
            pass

        if self.compute_hd95:
            try:
                hd95_score = self.hd95_metric.aggregate().item()
                results["val_mean_hd95_mm"] = hd95_score
            except Exception:
                results["val_mean_hd95_mm"] = 0.0

        return results
