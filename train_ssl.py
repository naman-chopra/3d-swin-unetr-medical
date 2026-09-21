"""
train_ssl.py - Stage 1: Self-Supervised 3D Masked Autoencoder Pretraining
========================================================================
Executes self-supervised pre-training on unlabeled 3D medical volumes.
Saves pre-trained encoder weights for transfer learning to downstream tasks.
"""

import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from src.data.datamodule import VolumetricMedicalDataModule
from src.models.mae_3d import MaskedAutoencoder3D


class SSLPretrainLightningModule(pl.LightningModule):
    """Lightning Module for 3D Masked Autoencoder Pre-training."""

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.model = MaskedAutoencoder3D(
            in_channels=config["model"]["in_channels"],
            img_size=tuple(config["model"]["img_size"]),
            patch_size=tuple(config["model"]["patch_size"]),
            embed_dim=config["model"]["embed_dim"],
            encoder_depth=config["model"]["encoder_depth"],
            num_heads=config["model"]["num_heads"],
            decoder_embed_dim=config["model"]["decoder_embed_dim"],
            decoder_depth=config["model"]["decoder_depth"],
            mask_ratio=config["model"]["mask_ratio"],
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        loss, _, _ = self.model(images)
        self.log("train_ssl_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        loss, _, _ = self.model(images)
        self.log("val_ssl_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config["training"]["lr"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
            betas=(0.9, 0.95),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    parser = argparse.ArgumentParser(description="Stage 1: 3D-MAE Self-Supervised Pretraining")
    parser.add_argument("--config", type=str, default="configs/pretrain_mae.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config["system"]["seed"], workers=True)

    # DataModule
    datamodule = VolumetricMedicalDataModule(
        data_dir=config["data"]["data_dir"],
        task_name=config["data"]["task_name"],
        stage="ssl",
        roi_size=tuple(config["data"]["roi_size"]),
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        cache_rate=config["data"]["cache_rate"],
    )

    # Model
    model = SSLPretrainLightningModule(config)

    # Callbacks & Logger
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/ssl",
        filename="mae_3d_best_{epoch:02d}_{val_ssl_loss:.4f}",
        monitor="val_ssl_loss",
        mode="min",
        save_top_k=2,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = CSVLogger(save_dir="logs", name="ssl_pretrain")

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="auto",
        devices=1,
        precision="16-mixed" if config["system"]["amp"] else 32,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        gradient_clip_val=1.0,
    )

    print("Starting Stage 1: 3D Masked Autoencoder Pretraining...")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
