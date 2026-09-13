"""
mae_3d.py - 3D Masked Autoencoder (MAE) for Volumetric Self-Supervision
======================================================================
Implements 3D Masked Image Modeling (MIM) on volumetric medical scans.
Randomly masks 75% of 3D cubical voxel patches and trains the 3D vision
encoder to reconstruct the missing anatomical structures.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from monai.networks.nets import ViT


class MaskedAutoencoder3D(nn.Module):
    """
    3D Masked Autoencoder (3D-MAE) for Volumetric Pre-training.
    
    Args:
        in_channels (int): Input MRI/CT modalities (e.g., 4 for BraTS, 1 for T1/CT).
        img_size (Tuple[int, int, int]): 3D spatial patch crop size (96, 96, 96).
        patch_size (Tuple[int, int, int]): 3D patch cuboid size (e.g., 16, 16, 16).
        embed_dim (int): Transformer latent embedding dimension (768).
        encoder_depth (int): Number of encoder self-attention layers (12).
        num_heads (int): Number of multi-head attention heads (12).
        decoder_embed_dim (int): Lightweight decoder embedding dimension (384).
        decoder_depth (int): Number of decoder layers (4).
        mask_ratio (float): Fraction of 3D patches to mask out (default: 0.75).
    """

    def __init__(
        self,
        in_channels: int = 4,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        embed_dim: int = 768,
        encoder_depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 384,
        decoder_depth: int = 4,
        decoder_num_heads: int = 6,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        # Calculate total number of 3D patches
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.patch_voxels = patch_size[0] * patch_size[1] * patch_size[2] * in_channels

        # 3D Patch Projection Encoder
        self.patch_embed = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        # 3D ViT Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        self.enc_norm = nn.LayerNorm(embed_dim)

        # 3D Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim)
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_num_heads,
            dim_feedforward=decoder_embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.dec_norm = nn.LayerNorm(decoder_embed_dim)

        # Predict voxel values for each 3D patch
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_voxels, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        """Initializes positional embeddings and projection weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def patchify_3d(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Splits 3D volumes into flattened patch tokens.
        imgs: (B, C, H, W, D) -> patches: (B, num_patches, patch_voxels)
        """
        p = self.patch_size[0]
        c = self.in_channels
        gh, gw, gd = self.grid_size
        
        # Reshape to (B, C, gh, p, gw, p, gd, p)
        x = imgs.reshape(imgs.shape[0], c, gh, p, gw, p, gd, p)
        # Permute to (B, gh, gw, gd, c, p, p, p)
        x = torch.einsum("nchpwqdr->nhwdcpqr", x)
        patches = x.reshape(imgs.shape[0], self.num_patches, self.patch_voxels)
        return patches

    def random_masking_3d(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs per-sample 3D random masking by shuffling patch indices.
        """
        N, L, D = x.shape  # Batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float):
        """Projects 3D patches, adds pos embed, and applies masking."""
        # Conv3D patch projection: (B, embed_dim, gh, gw, gd)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        x = x + self.pos_embed
        x, mask, ids_restore = self.random_masking_3d(x, mask_ratio)
        x = self.encoder(x)
        x = self.enc_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        """Inserts MASK tokens and reconstructs missing voxel values."""
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        x = x + self.decoder_pos_embed
        x = self.decoder(x)
        x = self.dec_norm(x)
        pred = self.decoder_pred(x)
        return pred

    def forward(self, imgs: torch.Tensor, mask_ratio: Optional[float] = None):
        """
        Forward pass computing 3D reconstruction loss on masked patches.
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        target = self.patchify_3d(imgs)

        # MSE Loss computed only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # Mean over masked patches

        return loss, pred, mask
