"""
transforms.py - Medical Imaging 3D Transforms Pipeline
======================================================
Robust 3D data augmentation and preprocessing pipeline utilizing MONAI.
Handles isotropic resampling, orientation standardization, non-zero intensity
z-scoring, and foreground-balanced 3D patch cropping for volumetric MRI/CT.
"""

from typing import Tuple, List, Optional
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    ScaleIntensityRangePercentilesd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    EnsureTyped,
    CastToTyped,
)
import torch


def get_pretrain_ssl_transforms(
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    num_samples_per_volume: int = 4,
) -> Compose:
    """
    Transforms for Stage 1: Self-Supervised 3D Masked Autoencoder Pre-training.
    Unlabeled volumes undergo spatial resampling, orientation alignment,
    non-zero z-score intensity normalization, and random 3D sub-volume extraction.
    """
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear",
            ),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=roi_size,
                num_samples=num_samples_per_volume,
                random_size=False,
            ),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image"], prob=0.5, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
    )


def get_finetune_train_transforms(
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    num_samples: int = 1,
    pos_neg_ratio: float = 1.0,
) -> Compose:
    """
    Transforms for Stage 2: Downstream 3D Volumetric Segmentation Training.
    Extracts foreground-balanced 3D patches centered around tumor/organ lesions
    using RandCropByPosNegLabeld to counter class imbalance.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=pos_neg_ratio,
                neg=1.0,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
                prob=0.2,
            ),
            RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.3),
            CastToTyped(keys=["label"], dtype=torch.uint8),
            EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.uint8)),
        ]
    )


def get_val_transforms() -> Compose:
    """
    Validation / Inference Transforms: Deterministic preprocessing with no cropping
    or random augmentations. Full 3D sliding-window inference will be executed on this.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            CastToTyped(keys=["label"], dtype=torch.uint8),
            EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.uint8)),
        ]
    )
