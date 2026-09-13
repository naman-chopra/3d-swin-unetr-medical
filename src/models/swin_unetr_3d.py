"""
swin_unetr_3d.py - 3D Swin Transformer U-Net for Volumetric Segmentation
========================================================================
Combines hierarchical 3D Swin Transformer encoder with 3D residual CNN
decoder and multi-scale skip connections for state-of-the-art volumetric
brain tumor and multi-organ segmentation.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETR3D(nn.Module):
    """
    3D Swin-UNETR Segmentation Backbone.
    
    Args:
        in_channels (int): Number of input modalities (e.g., 4 for BraTS: T1, T1ce, T2, FLAIR; 1 for CT).
        out_channels (int): Number of segmentation target classes (e.g., 3 for BraTS tumor subregions).
        img_size (Tuple[int, int, int]): 3D spatial patch crop size (default: 96x96x96).
        feature_size (int): Embedding dimension multiplier (default: 48).
        depths (Tuple[int, ...]): Number of Swin Transformer blocks per stage.
        num_heads (Tuple[int, ...]): Number of self-attention heads per stage.
        use_checkpoint (bool): Enable gradient checkpointing for VRAM memory optimization.
        spatial_dims (int): Spatial dimensions (3 for volumetric 3D).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 48,
        depths: Tuple[int, ...] = (2, 2, 2, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        use_checkpoint: bool = True,
        spatial_dims: int = 3,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.feature_size = feature_size

        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            norm_name="instance",
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=0.0,
            normalize=True,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 3D volumetric input tensor.
        Input shape:  (Batch, Channels, H, W, D)
        Output shape: (Batch, Classes, H, W, D)
        """
        return self.model(x)

    def load_pretrained_checkpoint(self, checkpoint_path: str, strict: bool = False):
        """
        Loads pre-trained Swin-UNETR backbone weights from checkpoint.

        Architectural Note:
            Stage 1 pre-training explores isotropic 3D-MAE (ViT), while Stage 2 downstream
            segmentation utilizes a hierarchical shifted-window Swin Transformer (SwinUNETR).
            Because the parameter topologies differ, this method loads native Swin-UNETR
            checkpoints directly into the hierarchical encoder and decoder.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Filter model weights
        clean_state_dict = {
            (k[6:] if k.startswith("model.") else k): v
            for k, v in state_dict.items()
        }
        missing, unexpected = self.load_state_dict(clean_state_dict, strict=strict)
        print(f"Loaded weights from: {checkpoint_path} (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
