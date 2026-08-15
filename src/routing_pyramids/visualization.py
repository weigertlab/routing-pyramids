"""Visualization utilities for routing-pyramid analyses."""

import torch
from torch import Tensor


def pca_colorize(features: Tensor) -> Tensor:
    """
    Map high-dimensional feature maps to RGB via PCA.

    Parameters
    ----------
    features : Tensor
        Feature tensor (B, D, H, W).

    Returns
    -------
    Tensor
        RGB tensor (B, 3, H, W) with values in [0, 1].
    """
    B, _D, H, W = features.shape
    features = features.float()
    flat = features.flatten(2).permute(0, 2, 1)  # (B, H*W, D)

    mean = flat.mean(dim=1, keepdim=True)
    centered = flat - mean

    with torch.autocast(device_type=features.device.type, enabled=False):
        _, _, V = torch.pca_lowrank(centered, q=3)
    projected = centered @ V  # (B, H*W, 3)

    proj_min = projected.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    proj_max = projected.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    rgb = (projected - proj_min) / (proj_max - proj_min + 1e-8)

    return rgb.permute(0, 2, 1).view(B, 3, H, W)
