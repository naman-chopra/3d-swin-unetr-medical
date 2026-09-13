"""
tests/test_data_pipeline.py
===========================
Tests data augmentation transforms, batch generation, and dataset splits.
"""

import pytest
import torch
from src.data.transforms import get_pretrain_ssl_transforms, get_finetune_train_transforms, get_val_transforms
from src.data.datamodule import VolumetricMedicalDataModule


def test_transforms_instantiation():
    """Verifies that all transform pipelines compile cleanly."""
    ssl_tf = get_pretrain_ssl_transforms(roi_size=(64, 64, 64))
    train_tf = get_finetune_train_transforms(roi_size=(64, 64, 64), num_samples=1)
    val_tf = get_val_transforms()

    assert ssl_tf is not None
    assert train_tf is not None
    assert val_tf is not None


def test_datamodule_discovery():
    """Verifies that data module properly discovers scans or uses fallback."""
    dm = VolumetricMedicalDataModule(
        data_dir="data",
        task_name="Task01_BrainTumour",
        stage="finetune",
        roi_size=(64, 64, 64),
        batch_size=1,
        num_workers=0,
        cache_rate=0.0,
    )
    dm.setup()
    assert len(dm.train_files) > 0
    assert len(dm.val_files) > 0
    assert len(dm.train_files) + len(dm.val_files) in (10, 484)
