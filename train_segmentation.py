"""
train_segmentation.py - Stage 2: 3D Swin-UNETR Volumetric Fine-Tuning
====================================================================
Fine-tunes the 3D Swin-UNETR backbone on labeled volumetric scans (BraTS / Decathlon).
Supports transfer learning from SSL checkpoints and evaluates online Dice / HD95 metrics
using MONAI sliding-window volumetric inference.
"""

import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

from src.data.datamodule import VolumetricMedicalDataModule
from src.models.swin_unetr_3d import SwinUNETR3D
from src.models.losses import VolumetricSegmentationLoss
from src.utils.metrics import VolumetricSegmentationMetrics


class SwinUNETRSegmentationLightningModule(pl.LightningModule):
    """Lightning Module for 3D Swin-UNETR Downstream Segmentation."""

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Backbone Model
        self.model = SwinUNETR3D(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            img_size=tuple(config["model"]["img_size"]),
            feature_size=config["model"]["feature_size"],
            depths=tuple(config["model"]["depths"]),
            num_heads=tuple(config["model"]["num_heads"]),
            use_checkpoint=config["model"]["use_checkpoint"],
        )

        # Load pre-trained checkpoint if provided
        pretrained_ckpt = config["model"].get("pretrained_checkpoint") or config["model"].get("ssl_pretrained_checkpoint")
        if pretrained_ckpt and os.path.exists(pretrained_ckpt):
            self.model.load_pretrained_checkpoint(pretrained_ckpt)

        # Loss & Metrics
        self.loss_fn = VolumetricSegmentationLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            loss_type=config["training"]["loss_type"],
        )
        self.metrics = VolumetricSegmentationMetrics(include_background=False, compute_hd95=False)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=config["model"]["out_channels"])
        self.post_label = AsDiscrete(to_onehot=config["model"]["out_channels"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]
        preds = self.model(images)
        loss = self.loss_fn(preds, labels)
        self.log("train_seg_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        images, labels = batch["image"], batch["label"]
        roi_size = tuple(self.config["data"]["roi_size"])

        # Full 3D sliding-window inference with Gaussian blending (memory-optimized for 6GB VRAM)
        val_outputs = sliding_window_inference(
            inputs=images,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=self.model,
            overlap=0.25,
            mode="gaussian",
            device=images.device,
        )

        val_loss = self.loss_fn(val_outputs, labels)
        self.log("val_seg_loss", val_loss, prog_bar=True, on_epoch=True)

        # Compute online Dice score
        val_pred_onehot = [self.post_pred(i) for i in val_outputs]
        val_label_onehot = [self.post_label(i) for i in labels]
        self.metrics.update(
            torch.stack(val_pred_onehot), torch.stack(val_label_onehot)
        )
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        scores = self.metrics.compute()
        self.log("val_mean_dice", scores.get("val_mean_dice", 0.0), prog_bar=True)
        for k, v in scores.items():
            if k != "val_mean_dice":
                self.log(k, v)
        self.metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config["training"]["lr"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    parser = argparse.ArgumentParser(description="Stage 2: 3D Swin-UNETR Volumetric Fine-Tuning")
    parser.add_argument("--config", type=str, default="configs/finetune_seg.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config["system"]["seed"], workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # DataModule
    datamodule = VolumetricMedicalDataModule(
        data_dir=config["data"]["data_dir"],
        task_name=config["data"]["task_name"],
        stage="finetune",
        roi_size=tuple(config["data"]["roi_size"]),
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        cache_rate=config["data"]["cache_rate"],
    )

    # Model
    model = SwinUNETRSegmentationLightningModule(config)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/segmentation",
        filename="swin_unetr_best_{epoch:02d}_{val_mean_dice:.4f}",
        monitor="val_mean_dice",
        mode="max",
        save_top_k=2,
    )
    early_stop = EarlyStopping(monitor="val_mean_dice", patience=25, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = CSVLogger(save_dir="logs", name="swin_unetr_finetune")

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="auto",
        devices=1,
        precision="16-mixed" if config["system"]["amp"] else 32,
        callbacks=[checkpoint_callback, early_stop, lr_monitor],
        logger=logger,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=2,
        limit_val_batches=25,
    )

    print("Starting Stage 2: 3D Swin-UNETR Volumetric Segmentation Fine-Tuning...")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
