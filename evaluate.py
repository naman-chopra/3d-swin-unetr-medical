"""
evaluate.py - Benchmark Evaluation & Multi-Planar Visualization Pipeline
========================================================================
Runs full-resolution 3D sliding-window inference on test cohorts, outputs
per-class Dice Similarity Coefficients & HD95, and renders visual slice overlays.
"""

import os
import argparse
import yaml
import torch
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

from src.data.datamodule import VolumetricMedicalDataModule
from src.models.swin_unetr_3d import SwinUNETR3D
from src.utils.metrics import VolumetricSegmentationMetrics
from src.utils.visualizer import save_multi_planar_slice_overlay


def evaluate(
    config_path: str,
    checkpoint_path: str,
    output_dir: str = "eval_results",
    max_samples: int = 30,
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = SwinUNETR3D(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        img_size=tuple(config["model"]["img_size"]),
        feature_size=config["model"]["feature_size"],
    ).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip only leading lightning wrapper prefix 'model.'
        clean_state_dict = {
            k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
        print(f"Loaded weights from: {checkpoint_path} (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

    model.eval()

    # Load Test DataModule
    datamodule = VolumetricMedicalDataModule(
        data_dir=config["data"]["data_dir"],
        task_name=config["data"]["task_name"],
        stage="finetune",
        roi_size=tuple(config["data"]["roi_size"]),
        batch_size=1,
        num_workers=2,
        cache_rate=0.0,
    )
    datamodule.setup()
    test_loader = datamodule.val_dataloader()

    metrics = VolumetricSegmentationMetrics(include_background=False)
    post_pred = AsDiscrete(argmax=True, to_onehot=config["model"]["out_channels"])
    post_label = AsDiscrete(to_onehot=config["model"]["out_channels"])

    print("Evaluating 3D sliding-window volumetric inference...")
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if max_samples is not None and idx >= max_samples:
                print(f"Reached evaluation sample limit: {max_samples} patients.")
                break
            images, labels = batch["image"].to(device), batch["label"].to(device)
            val_outputs = sliding_window_inference(
                inputs=images,
                roi_size=tuple(config["data"]["roi_size"]),
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
                mode="gaussian",
            )

            pred_discrete = post_pred(val_outputs[0])
            label_discrete = post_label(labels[0])
            metrics.update(pred_discrete.unsqueeze(0), label_discrete.unsqueeze(0))

            # Save visual slice overlay for first 5 samples
            if idx < 5:
                raw_img = images[0, 0].cpu().numpy()
                gt_mask = labels[0, 0].cpu().numpy()
                pred_mask = torch.argmax(val_outputs[0], dim=0).cpu().numpy()
                save_multi_planar_slice_overlay(
                    image_3d=raw_img,
                    ground_truth_3d=gt_mask,
                    pred_3d=pred_mask,
                    output_path=os.path.join(output_dir, f"patient_case_{idx:03d}_overlay.png"),
                )

    results = metrics.compute()
    print("\n" + "=" * 65)
    print("3D Swin-UNETR Evaluation Results:")
    print("=" * 65)
    print(f"  • Overall Mean Dice (DSC):               {results.get('val_mean_dice', 0.0):.4f}")
    if "val_class_1_dice" in results:
        print(f"  • Class 1 (Peritumoral Edema) Dice:      {results['val_class_1_dice']:.4f}")
    if "val_class_2_dice" in results:
        print(f"  • Class 2 (Necrotic Core / Non-Enh) Dice:{results['val_class_2_dice']:.4f}")
    if "val_class_3_dice" in results:
        print(f"  • Class 3 (Enhancing Tumor) Dice:        {results['val_class_3_dice']:.4f}")
    if "val_mean_hd95_mm" in results:
        print(f"  • 95th Percentile Hausdorff (HD95):      {results.get('val_mean_hd95_mm', 0.0):.2f} mm")
    print("=" * 65)

    # Save summary to JSON
    import json
    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation summary saved to: {summary_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D Swin-UNETR Model")
    parser.add_argument("--config", type=str, default="configs/finetune_seg.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/segmentation/swin_unetr_best_epoch=29_val_mean_dice=0.6957.ckpt",
    )
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--max_samples", type=int, default=30)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.output_dir, max_samples=args.max_samples)
