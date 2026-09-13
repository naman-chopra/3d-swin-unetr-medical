"""
datamodule.py - PyTorch Lightning Volumetric Medical DataModule
==============================================================
Production data loader handling 3D NIfTI medical scans (BraTS / MSD Decathlon),
multi-threaded MONAI CacheDataset integration, and reproducible cohort splitting.
"""

import os
from typing import Optional, List, Dict, Tuple
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from monai.data import CacheDataset, Dataset, list_data_collate
import numpy as np

from src.data.transforms import (
    get_pretrain_ssl_transforms,
    get_finetune_train_transforms,
    get_val_transforms,
)


class VolumetricMedicalDataModule(pl.LightningDataModule):
    """
    LightningDataModule for 3D Volumetric Medical Imaging.
    Supports both Self-Supervised Pretraining (unlabeled mode) and
    Supervised Fine-Tuning (labeled image-mask pairs).
    """

    def __init__(
        self,
        data_dir: str,
        task_name: str = "Task01_BrainTumour",
        stage: str = "finetune",  # "ssl" or "finetune"
        roi_size: Tuple[int, int, int] = (96, 96, 96),
        batch_size: int = 2,
        num_workers: int = 4,
        cache_rate: float = 0.5,
        val_split: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.task_name = task_name
        self.stage = stage
        self.roi_size = roi_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.val_split = val_split
        self.seed = seed

        self.train_files: List[Dict[str, str]] = []
        self.val_files: List[Dict[str, str]] = []
        self.test_files: List[Dict[str, str]] = []

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def _discover_files(self) -> List[Dict[str, str]]:
        """
        Discovers 3D NIfTI volumes (.nii / .nii.gz) in data_dir.
        If structured data folder is not found, creates deterministic
        synthetic benchmark registry for scaffolding and test pipelines.
        """
        image_dir = os.path.join(self.data_dir, self.task_name, "imagesTr")
        label_dir = os.path.join(self.data_dir, self.task_name, "labelsTr")

        files = []
        if os.path.exists(image_dir) and os.path.exists(label_dir):
            image_names = sorted(
                [f for f in os.listdir(image_dir) if f.endswith((".nii", ".nii.gz"))]
            )
            for img_name in image_names:
                img_path = os.path.join(image_dir, img_name)
                lbl_path = os.path.join(label_dir, img_name)
                if os.path.exists(lbl_path):
                    files.append({"image": img_path, "label": lbl_path})
                elif self.stage == "ssl":
                    files.append({"image": img_path})
        return files

    def setup(self, stage: Optional[str] = None):
        """Prepares train, validation, and test datasets with caching."""
        all_files = self._discover_files()
        
        # If running in a mock/dry-run test environment without pre-downloaded 100GB scans:
        if len(all_files) == 0:
            # Fallback to local data registry if empty
            all_files = [{"image": f"data/synthetic_sample_{i:03d}.nii.gz", "label": f"data/synthetic_label_{i:03d}.nii.gz"} for i in range(10)]

        np.random.seed(self.seed)
        np.random.shuffle(all_files)

        num_total = len(all_files)
        num_val = max(1, int(num_total * self.val_split))
        num_train = num_total - num_val

        self.train_files = all_files[:num_train]
        self.val_files = all_files[num_train:]
        self.test_files = self.val_files

        if self.stage == "ssl":
            train_transform = get_pretrain_ssl_transforms(roi_size=self.roi_size)
            val_transform = get_pretrain_ssl_transforms(roi_size=self.roi_size)
        else:
            train_transform = get_finetune_train_transforms(roi_size=self.roi_size)
            val_transform = get_val_transforms()

        # Wrap in MONAI CacheDataset for high-throughput GPU saturation
        if self.cache_rate > 0.0:
            self.train_dataset = CacheDataset(
                data=self.train_files,
                transform=train_transform,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
            )
            self.val_dataset = CacheDataset(
                data=self.val_files,
                transform=val_transform,
                cache_rate=self.cache_rate,
                num_workers=max(1, self.num_workers // 2),
            )
        else:
            self.train_dataset = Dataset(data=self.train_files, transform=train_transform)
            self.val_dataset = Dataset(data=self.val_files, transform=val_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=1,  # Full volume sliding-window inference uses batch size 1
            shuffle=False,
            num_workers=max(1, self.num_workers // 2),
            collate_fn=list_data_collate,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()
