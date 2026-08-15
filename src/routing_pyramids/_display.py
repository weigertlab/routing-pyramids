"""Private tensor display helpers used by training diagnostics."""

from functools import cache

import torch
from cmap import Colormap
from torch import Tensor


def _normalize_per_sample_for_display(image: Tensor) -> Tensor:
    if image.ndim != 4:
        raise ValueError(
            f"image must have shape (B, C, H, W), got {tuple(image.shape)}"
        )
    mins = image.amin(dim=(1, 2, 3), keepdim=True)
    maxs = image.amax(dim=(1, 2, 3), keepdim=True)
    return (image - mins) / (maxs - mins + 1e-8)


def _as_rgb_display(image: Tensor) -> Tensor:
    image = _normalize_per_sample_for_display(image)
    if image.shape[1] == 1:
        image = image.expand(-1, 3, -1, -1)
    elif image.shape[1] == 2:
        gfp, dna = image.unbind(dim=1)
        image = torch.stack((dna, gfp, dna), dim=1)
    elif image.shape[1] > 3:
        image = image[:, :3]
    return image.clamp(0.0, 1.0)


@cache
def _colormap_lut(name: str, size: int) -> Tensor:
    if size < 2:
        raise ValueError(f"size must be at least 2, got {size}")
    lut = Colormap(name).lut(N=size)[:, :3]
    return torch.as_tensor(lut, dtype=torch.float32)


def _as_colormap_display(
    image: Tensor, *, colormap: str, vmin: float, vmax: float, size: int = 256
) -> Tensor:
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError(
            "image must have shape (B, 1, H, W) for scalar colormap display, "
            f"got {tuple(image.shape)}"
        )
    if vmax <= vmin:
        raise ValueError(f"vmax must be greater than vmin, got {vmin=} and {vmax=}")
    values = image.float().sub(float(vmin)).div(float(vmax) - float(vmin))
    values = values.clamp(0.0, 1.0).squeeze(1)
    lut = _colormap_lut(colormap, size).to(device=values.device, dtype=values.dtype)
    scaled = values * float(size - 1)
    lower_idx = scaled.floor().long()
    upper_idx = scaled.ceil().long()
    mix = (scaled - lower_idx.to(dtype=scaled.dtype)).unsqueeze(1)
    lower_rgb = lut[lower_idx].permute(0, 3, 1, 2)
    upper_rgb = lut[upper_idx].permute(0, 3, 1, 2)
    return (1.0 - mix) * lower_rgb + mix * upper_rgb


def _as_magma_display(image: Tensor) -> Tensor:
    return _as_colormap_display(image, colormap="magma", vmin=0.0, vmax=1.0)


def _as_berlin_zero_display(image: Tensor) -> Tensor:
    return _as_colormap_display(image, colormap="berlin", vmin=-1.0, vmax=1.0)
