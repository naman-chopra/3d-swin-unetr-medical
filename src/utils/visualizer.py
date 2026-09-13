"""
visualizer.py - 3D Multi-Planar Volumetric Visualization
========================================================
Generates multi-planar (Axial, Coronal, Sagittal) slice overlays showing
original MRI scans with Ground Truth and Model Prediction contours.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import os


def save_multi_planar_slice_overlay(
    image_3d: np.ndarray,
    ground_truth_3d: np.ndarray,
    pred_3d: np.ndarray,
    output_path: str = "sample_prediction.png",
    slice_idx: Optional[int] = None,
):
    """
    Renders a 3-panel comparison (Input Scan, Ground Truth, Swin-UNETR Prediction)
    along the axial plane through the center of the tumor lesion.
    
    image_3d:        (H, W, D) float array
    ground_truth_3d: (H, W, D) integer mask
    pred_3d:         (H, W, D) integer predicted mask
    """
    if slice_idx is None:
        # Find slice with maximum tumor cross-section
        tumor_voxels_per_slice = ground_truth_3d.sum(axis=(0, 1))
        if tumor_voxels_per_slice.max() > 0:
            slice_idx = int(np.argmax(tumor_voxels_per_slice))
        else:
            slice_idx = image_3d.shape[2] // 2

    img_slice = image_3d[:, :, slice_idx]
    gt_slice = ground_truth_3d[:, :, slice_idx]
    pred_slice = pred_3d[:, :, slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Raw MRI Scan
    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title("Input 3D MRI (Axial Slice)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Ground Truth Radiologist Annotation
    axes[1].imshow(img_slice, cmap="gray")
    masked_gt = np.ma.masked_where(gt_slice == 0, gt_slice)
    axes[1].imshow(masked_gt, cmap="autumn", alpha=0.6)
    axes[1].set_title("Ground Truth Mask", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: 3D Swin-UNETR Prediction
    axes[2].imshow(img_slice, cmap="gray")
    masked_pred = np.ma.masked_where(pred_slice == 0, pred_slice)
    axes[2].imshow(masked_pred, cmap="winter", alpha=0.6)
    axes[2].set_title("3D Swin-UNETR Prediction", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization overlay to: {output_path}")
