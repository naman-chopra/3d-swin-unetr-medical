#!/usr/bin/env python3
"""
scripts/run_synthetic_demo.py
=============================
An end-to-end reproducible synthetic demonstration of the 3D-Swin-UNETR
pipeline. Generates synthetic 3D multi-sequence brain MRI volumes,
trains a 3D Swin-UNETR model for 3 epochs, runs 3D sliding-window
inference, and generates multi-planar slice overlays with Dice evaluation.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import list_data_collate
from torch.utils.data import DataLoader, Dataset

from src.models.swin_unetr_3d import SwinUNETR3D
from src.models.losses import VolumetricSegmentationLoss
from src.utils.metrics import VolumetricSegmentationMetrics
from src.utils.visualizer import save_multi_planar_slice_overlay


class Synthetic3DMriDataset(Dataset):
    """Generates synthetic 4-sequence brain MRI volumes with realistic ellipsoidal lesions."""

    def __init__(self, num_samples: int = 8, spatial_size: tuple = (64, 64, 64), seed: int = 42):
        np.random.seed(seed)
        self.samples = []
        for i in range(num_samples):
            # 4 channels: T1, T1ce, T2, FLAIR
            img = np.zeros((4, *spatial_size), dtype=np.float32)
            lbl = np.zeros((1, *spatial_size), dtype=np.int64)

            x, y, z = np.ogrid[:spatial_size[0], :spatial_size[1], :spatial_size[2]]
            cx, cy, cz = spatial_size[0] // 2, spatial_size[1] // 2, spatial_size[2] // 2

            # Cranial parenchyma mask
            brain_mask = ((x - cx)**2 / (cx*0.7)**2 + (y - cy)**2 / (cy*0.7)**2 + (z - cz)**2 / (cz*0.7)**2) < 1.0
            img[:, brain_mask] = np.random.uniform(0.2, 0.8, size=(4, brain_mask.sum()))

            # Edema region (class 2)
            tx = cx + np.random.randint(-8, 8)
            ty = cy + np.random.randint(-8, 8)
            tz = cz + np.random.randint(-8, 8)
            edema_mask = ((x - tx)**2 / 12**2 + (y - ty)**2 / 12**2 + (z - tz)**2 / 12**2) < 1.0
            lbl[0, edema_mask] = 2
            img[2, edema_mask] += 0.4  # T2 bright

            # Tumor core (class 1)
            core_mask = ((x - tx)**2 / 7**2 + (y - ty)**2 / 7**2 + (z - tz)**2 / 7**2) < 1.0
            lbl[0, core_mask] = 1
            img[1, core_mask] += 0.6  # T1ce enhancing

            # Enhancing active margin (class 3)
            enh_mask = ((x - tx)**2 / 4**2 + (y - ty)**2 / 4**2 + (z - tz)**2 / 4**2) < 1.0
            lbl[0, enh_mask] = 3

            self.samples.append({
                "image": torch.from_numpy(img),
                "label": torch.from_numpy(lbl)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    parser = argparse.ArgumentParser(description="3D Swin-UNETR Synthetic Demo")
    parser.add_argument("--epochs", type=int, default=3, help="Demo training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="data/processed/demo", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running 3D Swin-UNETR Synthetic Pipeline Demo on device: {device}")

    # 1. Dataset & DataLoader
    train_ds = Synthetic3DMriDataset(num_samples=6, spatial_size=(64, 64, 64), seed=42)
    val_ds = Synthetic3DMriDataset(num_samples=2, spatial_size=(64, 64, 64), seed=99)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 2. Model: 3D Swin-UNETR
    model = SwinUNETR3D(
        in_channels=4,
        out_channels=4,
        feature_size=24,       # Scaled for demo execution speed
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        use_checkpoint=False,
    ).to(device)

    loss_fn = VolumetricSegmentationLoss(include_background=False, to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 3. Training Loop
    print("\nTraining 3D Swin-UNETR on Synthetic 3D Multi-Sequence MRI...")
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch in train_loader:
            imgs, lbls = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch}/{args.epochs} | Training Loss: {epoch_loss / len(train_loader):.4f}")

    # 4. 3D Sliding-Window Validation
    print("\nRunning 3D sliding-window volumetric inference and evaluation...")
    model.eval()
    metrics = VolumetricSegmentationMetrics(include_background=False, compute_hd95=False)
    post_pred = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            imgs, lbls = batch["image"].to(device), batch["label"].to(device)
            val_preds = sliding_window_inference(
                inputs=imgs,
                roi_size=(64, 64, 64),
                sw_batch_size=2,
                predictor=model,
                overlap=0.25,
                mode="gaussian"
            )
            pred_disc = post_pred(val_preds[0])
            lbl_disc = post_label(lbls[0])
            metrics.update(pred_disc.unsqueeze(0), lbl_disc.unsqueeze(0))

            # Save visualization slice overlay
            raw_img = imgs[0, 2].cpu().numpy()  # T2 modality
            gt_mask = lbls[0, 0].cpu().numpy()
            pred_mask = torch.argmax(val_preds[0], dim=0).cpu().numpy()

            out_overlay = os.path.join(args.output_dir, f"sample_{idx+1}_overlay.png")
            save_multi_planar_slice_overlay(
                image_3d=raw_img,
                ground_truth_3d=gt_mask,
                pred_3d=pred_mask,
                output_path=out_overlay,
            )

    scores = metrics.compute()
    print("\n" + "=" * 55)
    print("Demo Validation Benchmark Results:")
    print(f"  • Validation Mean Dice (DSC): {scores.get('val_mean_dice', 0.0):.4f}")
    print(f"  • Overlays saved in: {args.output_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
