"""Coarse-to-fine pyramid flow autoencoder prototype."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TypedDict, cast

import numpy as np
import torch
import torchvision.utils as vutils
from einops import rearrange
from skimage.measure import label
from torch import Tensor, nn
from torch.nn import functional as F

from routing_pyramids._display import (
    _as_berlin_zero_display,
    _as_magma_display,
    _as_rgb_display,
)
from routing_pyramids.data.shared_frame_bank import validate_scale_factor
from routing_pyramids.typed_lightning import TypedLightningModule
from routing_pyramids.types import SegmentationPredictionPayload, TemporalBatch

Norm2d = str | tuple[str, dict[str, Any]]


class LossWeightSchedule(NamedTuple):
    """Linear schedule for a weighted loss term."""

    start_weight: float
    end_weight: float
    warmup_epochs: int


@dataclass(frozen=True)
class FineToCoarseLookup:
    """Static local 3x3 lookup from fine-grid sites to coarse-grid sites."""

    target_indices: Tensor
    valid_edges: Tensor
    offsets_y: Tensor
    offsets_x: Tensor
    fine_hw: tuple[int, int]
    coarse_hw: tuple[int, int]


class PyramidTransport(NamedTuple):
    """One fine-to-coarse local attention transport layer."""

    probs: Tensor
    logits: Tensor
    expected_offset: Tensor
    lookup: FineToCoarseLookup
    pixel_stride: int


class PyramidFlowEncoderOutput(NamedTuple):
    """Outputs produced by :class:`PyramidFlowEncoder2d.forward`."""

    features: Tensor


class PyramidFlowSegmentationEncoderOutput(NamedTuple):
    """Center probabilities and layer transports for segmentation-only inference."""

    center_logits: Tensor
    layer_transports: tuple[PyramidTransport, ...]


class PyramidFlowOutput(TypedDict):
    """Outputs produced by :class:`PyramidFlowSystem.forward`."""

    recon: Tensor
    foreground_logits: Tensor
    foreground_presence: Tensor
    p_fg: Tensor
    foreground_sparsity: Tensor
    foreground_kl: Tensor
    background_kl: Tensor
    flow_l2: Tensor
    foreground_kl_total: Tensor
    background_kl_total: Tensor
    flow_l2_total: Tensor
    expected_flow: Tensor
    foreground_latents: Tensor
    foreground_latent_mu: Tensor
    foreground_latent_logvar: Tensor
    background_latents: Tensor
    background_latent_mu: Tensor
    background_latent_logvar: Tensor
    foreground_features: Tensor
    background_features: Tensor
    layer_transports: tuple[PyramidTransport, ...]
    dense_assignment: Tensor | None


class PyramidFlowSegmentationInferenceOutput(TypedDict):
    """System outputs needed by flow-induced segmentation inference."""

    center_logits: Tensor
    center_presence: Tensor
    p_fg: Tensor
    expected_flow: Tensor
    layer_transports: tuple[PyramidTransport, ...]


@dataclass(frozen=True)
class PyramidFlowSegmentationTestConfig:
    """
    Flow-induced segmentation settings for prediction.

    Parameters
    ----------
    center_threshold
        Threshold on bottleneck center probabilities used to define seed regions.
    pixel_mass_threshold
        Minimum propagated object mass required to assign a pixel to an instance.
    component_chunk_size
        Number of center components propagated together on the accelerator. Larger
        chunks reduce launch overhead at the cost of additional peak memory.
    min_object_area
        Optional minimum predicted object area, measured as the number of pixels
        in the model-input-resolution label grid after pixel assignment. This
        filtering occurs before ``prediction_output_scale_factor`` is applied.
    max_object_area
        Optional maximum predicted object area in the same model-input-resolution
        pixel units, applied before ``prediction_output_scale_factor``.
    prediction_output_scale_factor
        Spatial output/input ratio used to restore predicted label images to the
        source resolution for prediction payloads.
    """

    center_threshold: float = 0.5
    pixel_mass_threshold: float = 0.5
    component_chunk_size: int = 16
    min_object_area: int | None = 50
    max_object_area: int | None = None
    prediction_output_scale_factor: float = 1.0


class PyramidFlowSegmentationOutput(NamedTuple):
    """Flow-induced instance labels and diagnostic maps."""

    center_components: Tensor
    pred_labels_unfiltered: Tensor
    pred_labels: Tensor
    winning_mass: Tensor


def _norm_name_and_kwargs(norm: Norm2d) -> tuple[str, dict[str, Any]]:
    if isinstance(norm, str):
        return norm.upper(), {}
    name, kwargs = norm
    return name.upper(), dict(kwargs)


def _scale_label_images(labels: Tensor, *, scale_factor: float) -> Tensor:
    """Nearest-neighbor scale label images without converting their dtype."""
    scale_factor = validate_scale_factor(scale_factor)
    if scale_factor > 1:
        integer_factor = round(scale_factor)
        return labels.repeat_interleave(integer_factor, dim=-2).repeat_interleave(
            integer_factor, dim=-1
        )
    if scale_factor == 1:
        return labels

    reduction_factor = round(1.0 / scale_factor)
    height, width = labels.shape[-2:]
    if height % reduction_factor != 0 or width % reduction_factor != 0:
        raise ValueError(
            f"label image shape {(height, width)} must be divisible by reduction "
            f"factor {reduction_factor} for scale_factor={scale_factor}"
        )
    return labels[..., ::reduction_factor, ::reduction_factor]


def _filter_label_tensor_by_area(
    labels: Tensor,
    *,
    min_object_area: int | None,
    max_object_area: int | None,
) -> Tensor:
    """Filter a 2D label tensor by area and compact the retained IDs."""
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape (H, W), got {tuple(labels.shape)}")
    if min_object_area is None and max_object_area is None:
        return labels.to(dtype=torch.int32)

    labels_long = labels.to(dtype=torch.long)
    counts = torch.bincount(labels_long.reshape(-1))
    object_ids = torch.arange(counts.numel(), device=labels.device)
    keep = object_ids != 0
    if min_object_area is not None:
        keep &= counts >= int(min_object_area)
    if max_object_area is not None:
        keep &= counts <= int(max_object_area)

    mapping = torch.zeros(counts.numel(), dtype=torch.int32, device=labels.device)
    kept_ids = object_ids[keep]
    mapping[kept_ids] = torch.arange(
        1,
        kept_ids.numel() + 1,
        dtype=torch.int32,
        device=labels.device,
    )
    return mapping[labels_long]


def _make_norm(norm: Norm2d, num_channels: int) -> nn.Module:
    name, kwargs = _norm_name_and_kwargs(norm)
    if name == "INSTANCE":
        return nn.InstanceNorm2d(
            num_channels, affine=bool(kwargs.pop("affine", True)), **kwargs
        )
    if name == "BATCH":
        return nn.BatchNorm2d(num_channels, **kwargs)
    if name == "GROUP":
        num_groups = int(kwargs.pop("num_groups", min(8, num_channels)))
        if num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {num_groups}")
        while num_channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, **kwargs)
    raise ValueError(f"Unsupported 2D normalization {name!r}")


def _positive_int(name: str, value: int) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _nonnegative_int(name: str, value: int) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _unit_interval_float(name: str, value: float) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {result}")
    return result


def _positive_int_sequence(
    name: str,
    values: Sequence[int],
    *,
    expected_len: int | None = None,
) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if expected_len is not None and len(result) != expected_len:
        raise ValueError(f"{name} length must be {expected_len}, got {result}")
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _validate_spatial_tensor(name: str, tensor: Tensor, *, channels: int) -> None:
    if tensor.ndim != 4:
        raise ValueError(
            f"{name} must have shape (B, C, H, W), got {tuple(tensor.shape)}"
        )
    if int(tensor.shape[1]) != channels:
        raise ValueError(
            f"{name} must have {channels} channels, got {tuple(tensor.shape)}"
        )


def _validate_matching_batch_and_spatial(
    named_tensors: Sequence[tuple[str, Tensor]],
) -> None:
    reference_name, reference = named_tensors[0]
    reference_batch = int(reference.shape[0])
    reference_hw = tuple(reference.shape[-2:])
    for name, tensor in named_tensors[1:]:
        if int(tensor.shape[0]) != reference_batch:
            raise ValueError(
                f"{reference_name} and {name} batch sizes must match, got "
                f"{reference_batch} and {tensor.shape[0]}"
            )
        if tuple(tensor.shape[-2:]) != reference_hw:
            raise ValueError(
                f"{reference_name} and {name} spatial shapes must match, got "
                f"{reference_hw} and {tuple(tensor.shape[-2:])}"
            )


class PointwiseMLP2d(nn.Sequential):
    """
    Two-layer pointwise MLP for spatial feature maps.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of channels in the hidden representation.
    out_channels : int
        Number of output channels.
    hidden_norm : nn.Module or None
        Optional normalization applied after the input projection and before
        GELU. If None, no hidden normalization is applied.
    bias : bool
        If True, include a bias in both pointwise projections.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        hidden_norm: nn.Module | None = None,
        bias: bool = True,
    ) -> None:
        in_channels = _positive_int("in_channels", in_channels)
        hidden_channels = _positive_int("hidden_channels", hidden_channels)
        out_channels = _positive_int("out_channels", out_channels)
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=bias)
        ]
        if hidden_norm is not None:
            layers.append(hidden_norm)
        layers.extend(
            [
                nn.GELU(),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=bias),
            ]
        )
        super().__init__(*layers)


class ChannelLayerNorm2d(nn.Module):
    """
    Layer normalization over channels at each spatial location.

    Parameters
    ----------
    num_channels : int
        Number of feature channels.
    affine : bool
        If True, learn a per-channel scale.
    bias : bool
        If True and ``affine`` is enabled, learn a per-channel offset.
    eps : float
        Positive variance regularizer.
    """

    def __init__(
        self,
        num_channels: int,
        *,
        affine: bool = True,
        bias: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        num_channels = _positive_int("num_channels", num_channels)
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.num_channels = num_channels
        self.affine = bool(affine)
        self.use_bias = self.affine and bool(bias)
        self.eps = float(eps)
        self.weight: nn.Parameter | None = None
        self.bias: nn.Parameter | None = None
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            if self.use_bias:
                self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize each pixel across channels."""
        _validate_spatial_tensor("x", x, channels=self.num_channels)
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).square().mean(dim=1, keepdim=True)
        y = (x - mean) * torch.rsqrt(var + self.eps)
        if self.affine:
            weight = cast(Tensor, self.weight).view(1, self.num_channels, 1, 1)
            y = y * weight
            if self.use_bias:
                bias = cast(Tensor, self.bias).view(1, self.num_channels, 1, 1)
                y = y + bias
        return y


class ResidualConvBlock2d(nn.Module):
    """Two-convolution residual block for deterministic feature transforms."""

    def __init__(self, *, in_channels: int, out_channels: int, norm: Norm2d) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = _make_norm(norm, out_channels)
        self.act1 = nn.PReLU(num_parameters=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = _make_norm(norm, out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.out_act = nn.PReLU(num_parameters=out_channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.out_act(out + residual)


def _validate_block_counts(
    name: str, block_counts: Sequence[int] | None, expected_len: int
) -> tuple[int, ...]:
    if block_counts is None:
        return (1,) * expected_len
    return _positive_int_sequence(name, block_counts, expected_len=expected_len)


def _make_residual_stack(
    *,
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    norm: Norm2d,
) -> nn.Sequential:
    blocks: list[nn.Module] = [
        ResidualConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            norm=norm,
        )
    ]
    blocks.extend(
        ResidualConvBlock2d(
            in_channels=out_channels,
            out_channels=out_channels,
            norm=norm,
        )
        for _ in range(num_blocks - 1)
    )
    return nn.Sequential(*blocks)


class DownStage2d(nn.Module):
    """Average-pooling downsampling stage followed by residual blocks."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int,
        num_blocks: int,
        norm: Norm2d,
    ) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)
        self.blocks = _make_residual_stack(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            norm=norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Downsample and refine a feature map."""
        return self.blocks(self.pool(x))


def fine_to_coarse_lookup(
    *,
    fine_h: int,
    fine_w: int,
    coarse_h: int,
    coarse_w: int,
    device: torch.device,
) -> FineToCoarseLookup:
    """Build a 3x3 fine-to-coarse neighborhood lookup."""
    fine_h = int(fine_h)
    fine_w = int(fine_w)
    coarse_h = int(coarse_h)
    coarse_w = int(coarse_w)
    if fine_h <= 0 or fine_w <= 0 or coarse_h <= 0 or coarse_w <= 0:
        raise ValueError(
            "fine and coarse dimensions must be positive, got "
            f"fine=({fine_h}, {fine_w}) and coarse=({coarse_h}, {coarse_w})"
        )
    if fine_h % coarse_h != 0 or fine_w % coarse_w != 0:
        raise ValueError(
            "fine dimensions must be divisible by coarse dimensions, got "
            f"fine=({fine_h}, {fine_w}) and coarse=({coarse_h}, {coarse_w})"
        )

    scale_h = fine_h // coarse_h
    scale_w = fine_w // coarse_w
    offsets = torch.tensor([-1, 0, 1], device=device, dtype=torch.long)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    dy = dy.reshape(-1)
    dx = dx.reshape(-1)

    rows = torch.arange(fine_h * fine_w, device=device)
    row_y = torch.div(rows, fine_w, rounding_mode="floor")
    row_x = torch.remainder(rows, fine_w)
    anchor_y = torch.div(row_y, scale_h, rounding_mode="floor")
    anchor_x = torch.div(row_x, scale_w, rounding_mode="floor")
    target_y = anchor_y.unsqueeze(1) + dy.unsqueeze(0)
    target_x = anchor_x.unsqueeze(1) + dx.unsqueeze(0)
    valid_edges = (
        (target_y >= 0)
        & (target_y < coarse_h)
        & (target_x >= 0)
        & (target_x < coarse_w)
    )
    target_indices = target_y.clamp(0, coarse_h - 1) * coarse_w + target_x.clamp(
        0, coarse_w - 1
    )
    return FineToCoarseLookup(
        target_indices=target_indices.to(dtype=torch.long),
        valid_edges=valid_edges,
        offsets_y=dy,
        offsets_x=dx,
        fine_hw=(fine_h, fine_w),
        coarse_hw=(coarse_h, coarse_w),
    )


def _expected_local_offset(probs: Tensor, *, lookup: FineToCoarseLookup) -> Tensor:
    offsets_y = lookup.offsets_y.to(device=probs.device, dtype=probs.dtype)
    offsets_x = lookup.offsets_x.to(device=probs.device, dtype=probs.dtype)
    offsets = torch.stack((offsets_y, offsets_x), dim=0)
    return torch.einsum("bkhw,ck->bchw", probs, offsets)


def _transport_from_logits(
    logits: Tensor,
    *,
    lookup: FineToCoarseLookup,
    pixel_stride: int,
) -> PyramidTransport:
    fine_h, fine_w = lookup.fine_hw
    logits_float = logits.float()
    valid = lookup.valid_edges.T.reshape(1, 9, fine_h, fine_w).to(device=logits.device)
    masked_logits = logits_float.masked_fill(~valid, -torch.finfo(torch.float32).max)
    probs = torch.softmax(masked_logits, dim=1) * valid.to(dtype=torch.float32)
    return PyramidTransport(
        probs=probs.to(dtype=logits.dtype),
        logits=masked_logits.to(dtype=logits.dtype),
        expected_offset=_expected_local_offset(probs, lookup=lookup).to(
            dtype=logits.dtype
        ),
        lookup=lookup,
        pixel_stride=int(pixel_stride),
    )


class PyramidFlowEncoder2d(nn.Module):
    """
    Downsampling encoder that returns deterministic bottleneck features.

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    channels : Sequence[int]
        Feature widths from full-resolution stem to bottleneck.
    strides : Sequence[int]
        Downsampling strides between adjacent channel stages.
    down_blocks : Sequence[int] or None
        Number of residual blocks in the stem and each downsampling stage.
    norm : str | tuple[str, dict[str, Any]]
        Normalization spec for encoder residual blocks.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        down_blocks: Sequence[int] | None = None,
        norm: Norm2d = "INSTANCE",
    ) -> None:
        super().__init__()
        in_channels = _positive_int("in_channels", in_channels)
        channels_tuple = _positive_int_sequence("channels", channels)
        strides_tuple = _positive_int_sequence("strides", strides)
        if len(channels_tuple) != len(strides_tuple) + 1:
            raise ValueError(
                "channels length must equal len(strides) + 1, got "
                f"{channels_tuple} and strides={strides_tuple}"
            )
        down_blocks_tuple = _validate_block_counts(
            "down_blocks", down_blocks, len(channels_tuple)
        )

        self.in_channels = in_channels
        self.out_channels = channels_tuple[-1]
        self.channels = channels_tuple
        self.strides = strides_tuple
        self.down_blocks = down_blocks_tuple
        self.bottleneck_channels = channels_tuple[-1]
        self.dim = channels_tuple[-1]
        self.feature_stride = math.prod(strides_tuple)

        self.stem = _make_residual_stack(
            in_channels=in_channels,
            out_channels=channels_tuple[0],
            num_blocks=down_blocks_tuple[0],
            norm=norm,
        )
        self.down_stages = nn.ModuleList(
            DownStage2d(
                in_channels=in_stage_channels,
                out_channels=out_stage_channels,
                stride=stride,
                num_blocks=num_blocks,
                norm=norm,
            )
            for in_stage_channels, out_stage_channels, stride, num_blocks in zip(
                channels_tuple[:-1],
                channels_tuple[1:],
                strides_tuple,
                down_blocks_tuple[1:],
                strict=True,
            )
        )

    def forward(self, x: Tensor) -> PyramidFlowEncoderOutput:
        """Return deterministic bottleneck features."""
        _validate_spatial_tensor("x", x, channels=self.in_channels)
        height, width = int(x.shape[-2]), int(x.shape[-1])
        if height % self.feature_stride != 0 or width % self.feature_stride != 0:
            raise ValueError(
                "x spatial shape must be divisible by prod(strides), got "
                f"{tuple(x.shape[-2:])} and strides={self.strides}"
            )

        features = self.stem(x)
        for stage in self.down_stages:
            features = stage(features)
        return PyramidFlowEncoderOutput(features=features)


class TransportResidualBlock2d(nn.Module):
    """Residual block whose value path is routed by a local transport plan."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        ffn_expansion: int,
        value_modulation: bool = False,
    ) -> None:
        super().__init__()
        in_channels = _positive_int("in_channels", in_channels)
        out_channels = _positive_int("out_channels", out_channels)
        ffn_expansion = _positive_int("ffn_expansion", ffn_expansion)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.value_proj = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.value_modulation_proj: nn.Conv2d | None = None
        if value_modulation:
            self.value_modulation_proj = nn.Conv2d(
                9, out_channels, kernel_size=1, bias=False
            )
            nn.init.zeros_(self.value_modulation_proj.weight)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.norm = ChannelLayerNorm2d(out_channels, bias=False)
        self.act = nn.GELU()
        hidden_channels = out_channels * ffn_expansion
        self.ffn_norm = ChannelLayerNorm2d(out_channels, bias=False)
        self.ffn = PointwiseMLP2d(
            in_channels=out_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            bias=False,
        )

    def forward(
        self, x: Tensor, *, transport: PyramidTransport, gate: Tensor
    ) -> Tensor:
        """Route values, gate the update, add a skip, and apply a Conv-FFN."""
        values = self.value_proj(x)
        routed = self.spatially_varying_convolution(values, transport=transport)
        if self.value_modulation_proj is not None:
            routed = routed * self._value_modulation_gain(transport=transport).to(
                dtype=routed.dtype
            )
        out = self.act(self.norm(routed)) * gate.to(dtype=routed.dtype)
        residual = F.interpolate(x, size=transport.lookup.fine_hw, mode="nearest")
        out = out + self.skip(residual)
        return out + self.ffn(self.ffn_norm(out))

    def _value_modulation_gain(self, *, transport: PyramidTransport) -> Tensor:
        projection = cast(nn.Conv2d, self.value_modulation_proj)
        modulation = projection(transport.probs.to(dtype=projection.weight.dtype))
        return 1.0 + torch.tanh(modulation)

    @staticmethod
    def gather_local_features(x: Tensor, *, lookup: FineToCoarseLookup) -> Tensor:
        """Return local 3x3 source features for each fine-grid destination."""
        batch_size, channels, _, _ = x.shape
        fine_h, fine_w = lookup.fine_hw
        indices = lookup.target_indices.to(device=x.device).reshape(-1)
        flat = rearrange(x, "b c h w -> b c (h w)")
        gathered = flat.index_select(dim=2, index=indices)
        return (
            gathered.view(batch_size, channels, fine_h * fine_w, 9)
            .permute(0, 1, 3, 2)
            .reshape(batch_size, channels, 9, fine_h, fine_w)
        )

    @staticmethod
    def spatially_varying_convolution(
        x: Tensor, *, transport: PyramidTransport
    ) -> Tensor:
        """Average regular-grid 3x3 source neighborhoods with transport weights."""
        result_dtype = torch.promote_types(x.dtype, transport.probs.dtype)
        values = x.to(dtype=result_dtype)
        probs = transport.probs.to(dtype=result_dtype)
        fine_h, fine_w = transport.lookup.fine_hw
        coarse_h, coarse_w = transport.lookup.coarse_hw
        scale_h = fine_h // coarse_h
        scale_w = fine_w // coarse_w
        padded = F.pad(values, (1, 1, 1, 1))
        routed = values.new_zeros(
            int(values.shape[0]), int(values.shape[1]), fine_h, fine_w
        )
        edge_index = 0
        for offset_y in range(-1, 2):
            for offset_x in range(-1, 2):
                shifted = padded[
                    ...,
                    1 + offset_y : 1 + offset_y + coarse_h,
                    1 + offset_x : 1 + offset_x + coarse_w,
                ]
                shifted = shifted.repeat_interleave(scale_h, dim=-2).repeat_interleave(
                    scale_w, dim=-1
                )
                routed = routed + shifted * probs[:, edge_index : edge_index + 1]
                edge_index += 1
        return routed


class LocalTransportScorer2d(nn.Module):
    """Predict local 3x3 attention transport from query and source features."""

    def __init__(
        self, *, channels: int, attention_channels: int, use_edge_bias: bool = False
    ) -> None:
        super().__init__()
        channels = _positive_int("channels", channels)
        attention_channels = _positive_int("attention_channels", attention_channels)
        self.channels = channels
        self.attention_channels = attention_channels
        self.use_edge_bias = bool(use_edge_bias)
        self.query_proj = nn.Conv2d(
            channels, attention_channels, kernel_size=3, padding=1
        )
        self.key_proj = nn.Conv2d(
            channels, attention_channels, kernel_size=3, padding=1
        )
        if self.use_edge_bias:
            self.edge_bias = nn.Parameter(torch.zeros(1, 9, 1, 1))

    def forward(
        self,
        *,
        source: Tensor,
        query: Tensor,
        lookup: FineToCoarseLookup,
        pixel_stride: int,
    ) -> PyramidTransport:
        """Return a masked 3x3 transport distribution for each fine query site."""
        q = self.query_proj(query).float()
        k = self.key_proj(source).float()
        gathered_keys = TransportResidualBlock2d.gather_local_features(k, lookup=lookup)
        logits = torch.einsum("bdhw,bdkhw->bkhw", q, gathered_keys)
        logits = logits / math.sqrt(float(self.attention_channels))
        if self.use_edge_bias:
            logits = logits + self.edge_bias.to(dtype=logits.dtype)
        return _transport_from_logits(
            logits.to(dtype=query.dtype), lookup=lookup, pixel_stride=pixel_stride
        )


class ConvTransportScorer2d(nn.Module):
    """Predict local 3x3 transport logits from fine-grid query features."""

    def __init__(self, *, channels: int) -> None:
        super().__init__()
        channels = _positive_int("channels", channels)
        self.channels = channels
        self.block = ResidualConvBlock2d(
            in_channels=channels,
            out_channels=channels,
            norm=("GROUP", {"num_groups": 1}),
        )
        self.logits_head = nn.Conv2d(channels, 9, kernel_size=1)

    def forward(
        self,
        *,
        source: Tensor,
        query: Tensor,
        lookup: FineToCoarseLookup,
        pixel_stride: int,
    ) -> PyramidTransport:
        """Return a masked 3x3 transport distribution for each fine query site."""
        logits = self.logits_head(self.block(query))
        return _transport_from_logits(logits, lookup=lookup, pixel_stride=pixel_stride)


class PyramidFlowDecoderLayer2d(nn.Module):
    """Single transformer-style decoder layer with one local transport plan."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int,
        pixel_stride: int,
        attention_channels: int,
        use_edge_bias: bool,
        ffn_expansion: int,
        transport_predictor: Literal["attention", "conv"] = "attention",
        value_modulation: bool = False,
    ) -> None:
        super().__init__()
        in_channels = _positive_int("in_channels", in_channels)
        out_channels = _positive_int("out_channels", out_channels)
        stride = _positive_int("stride", stride)
        pixel_stride = _positive_int("pixel_stride", pixel_stride)
        attention_channels = _positive_int("attention_channels", attention_channels)
        ffn_expansion = _positive_int("ffn_expansion", ffn_expansion)
        if transport_predictor not in ("attention", "conv"):
            raise ValueError(
                "transport_predictor must be 'attention' or 'conv', got "
                f"{transport_predictor!r}"
            )
        if transport_predictor == "conv" and use_edge_bias:
            raise ValueError("use_edge_bias is only supported for attention transport")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.pixel_stride = pixel_stride
        self.transport_predictor = transport_predictor
        self.value_modulation = bool(value_modulation)
        self.transport_scorer: LocalTransportScorer2d | ConvTransportScorer2d = (
            LocalTransportScorer2d(
                channels=in_channels,
                attention_channels=attention_channels,
                use_edge_bias=use_edge_bias,
            )
            if transport_predictor == "attention"
            else ConvTransportScorer2d(channels=in_channels)
        )
        self.block = TransportResidualBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            ffn_expansion=ffn_expansion,
            value_modulation=self.value_modulation,
        )

    def forward(
        self,
        *,
        x: Tensor,
        foreground_presence: Tensor,
        transport: PyramidTransport | None,
    ) -> tuple[Tensor, Tensor, PyramidTransport]:
        """Score one transport and update a feature stream plus its alpha gate."""
        fine_h = int(x.shape[-2]) * self.stride
        fine_w = int(x.shape[-1]) * self.stride
        lookup = fine_to_coarse_lookup(
            fine_h=fine_h,
            fine_w=fine_w,
            coarse_h=int(x.shape[-2]),
            coarse_w=int(x.shape[-1]),
            device=x.device,
        )
        query = (
            x
            if self.stride == 1
            else F.interpolate(x, size=(fine_h, fine_w), mode="nearest")
        )
        if transport is None:
            query_presence = (
                foreground_presence
                if self.stride == 1
                else F.interpolate(
                    foreground_presence, size=(fine_h, fine_w), mode="nearest"
                )
            ).to(dtype=query.dtype)
            source_for_attention = x * foreground_presence.to(dtype=x.dtype)
            query_for_attention = query * query_presence
            transport = self.transport_scorer(
                source=source_for_attention,
                query=query_for_attention,
                lookup=lookup,
                pixel_stride=self.pixel_stride,
            )
        next_presence = TransportResidualBlock2d.spatially_varying_convolution(
            foreground_presence, transport=transport
        ).to(dtype=x.dtype)
        next_x = self.block(x, transport=transport, gate=next_presence)
        return next_x, next_presence, transport


class PyramidFlowDecoder2d(nn.Module):
    """
    Pyramid decoder with unrolled local-attention transport layers.

    Parameters
    ----------
    in_channels : int
        Bottleneck latent width. Must match ``channels[0]`` because foreground
        latents enter the transported feature path directly.
    out_channels : int
        Reconstruction channel count.
    channels : tuple[int, ...]
        Decoder feature widths from bottleneck to output resolution.
    strides : tuple[int, ...]
        Upsampling strides between adjacent decoder stages.
    feature_stride : int
        Pixel stride of the bottleneck grid.
    stage_blocks : Sequence[int] or None
        Number of explicit transport layers at each resolution stage.
    attention_channels : int or None
        Width used for local QK attention. Defaults to the stage input width.
    transport_predictor : {"attention", "conv"}
        Transport predictor used by decoder layers. Attention uses local QK
        scoring; conv uses a residual convolutional block plus a 1x1 logits head.
    use_edge_bias : bool
        If True, add a learned scalar bias for each local 3x3 attention edge.
    normalize_latent_blend : bool
        If True, normalize bottleneck streams with affine-free channel LayerNorm
        before decoding. Dual-stream pixel features are always L2-normalized
        immediately before mixing.
    dual_stream : bool
        If True, transport only foreground features and blend them with sampled
        pixel-resolution background features. Both streams are L2-normalized per
        pixel and mixed with variance-preserving foreground/background weights.
        If False, blend latent streams before the transported decoder.
    value_modulation : bool
        If True, modulate routed values by a bounded per-channel projection of
        the local transport probabilities at every decoder layer.
    ffn_expansion : int
        Channel expansion factor for the pointwise Conv-FFN in each block.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...],
        strides: tuple[int, ...],
        feature_stride: int,
        stage_blocks: Sequence[int] | None = None,
        attention_channels: int | None = None,
        transport_predictor: Literal["attention", "conv"] = "attention",
        use_edge_bias: bool = False,
        normalize_latent_blend: bool = True,
        dual_stream: bool = False,
        value_modulation: bool = False,
        ffn_expansion: int = 2,
    ) -> None:
        super().__init__()
        in_channels = _positive_int("in_channels", in_channels)
        out_channels = _positive_int("out_channels", out_channels)
        channels = _positive_int_sequence("channels", channels)
        strides = _positive_int_sequence("strides", strides)
        feature_stride = _positive_int("feature_stride", feature_stride)
        if len(channels) != len(strides) + 1:
            raise ValueError(
                "channels must have one entry for the bottleneck plus one per "
                f"stride, got channels={channels} and strides={strides}"
            )
        if in_channels != channels[0]:
            raise ValueError(
                "in_channels must equal channels[0] because foreground latents "
                "enter the transported path directly, got "
                f"{in_channels} and {channels[0]}"
            )
        if math.prod(strides) != feature_stride:
            raise ValueError(
                "decoder strides must multiply to feature_stride, got "
                f"strides={strides} and feature_stride={feature_stride}"
            )
        if attention_channels is not None:
            attention_channels = _positive_int("attention_channels", attention_channels)
        if transport_predictor not in ("attention", "conv"):
            raise ValueError(
                "transport_predictor must be 'attention' or 'conv', got "
                f"{transport_predictor!r}"
            )
        if transport_predictor == "conv" and use_edge_bias:
            raise ValueError("use_edge_bias is only supported for attention transport")
        ffn_expansion = _positive_int("ffn_expansion", ffn_expansion)
        stage_blocks_tuple = _validate_block_counts(
            "stage_blocks", stage_blocks, len(channels)
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.feature_stride = feature_stride
        self.stage_blocks = stage_blocks_tuple
        self.attention_channels = attention_channels
        self.transport_predictor = transport_predictor
        self.use_edge_bias = bool(use_edge_bias)
        self.normalize_latent_blend = bool(normalize_latent_blend)
        self.dual_stream = bool(dual_stream)
        self.value_modulation = bool(value_modulation)
        self.ffn_expansion = ffn_expansion
        stage_pixel_strides = [int(feature_stride)]
        cumulative = 1
        for stride in strides:
            cumulative *= int(stride)
            if int(feature_stride) % cumulative != 0:
                raise ValueError(
                    "feature_stride must be divisible by cumulative decoder stride, "
                    f"got feature_stride={feature_stride} and cumulative={cumulative}"
                )
            stage_pixel_strides.append(int(feature_stride) // cumulative)
        self.stage_pixel_strides = tuple(stage_pixel_strides)

        stage_in_channels = (channels[0], *channels[:-1])
        stage_strides = (1, *strides)
        layers: list[PyramidFlowDecoderLayer2d] = []
        layer_stage_indices: list[int] = []
        layer_strides: list[int] = []
        layer_pixel_strides: list[int] = []
        for stage_index, (
            in_stage_channels,
            out_stage_channels,
            stage_stride,
            stage_pixel_stride,
            num_blocks,
        ) in enumerate(
            zip(
                stage_in_channels,
                channels,
                stage_strides,
                self.stage_pixel_strides,
                stage_blocks_tuple,
                strict=True,
            )
        ):
            for block_index in range(num_blocks):
                layer_in_channels = (
                    in_stage_channels if block_index == 0 else out_stage_channels
                )
                layer_stride = stage_stride if block_index == 0 else 1
                layers.append(
                    PyramidFlowDecoderLayer2d(
                        in_channels=layer_in_channels,
                        out_channels=out_stage_channels,
                        stride=layer_stride,
                        pixel_stride=stage_pixel_stride,
                        attention_channels=(
                            layer_in_channels
                            if self.attention_channels is None
                            else self.attention_channels
                        ),
                        use_edge_bias=self.use_edge_bias,
                        transport_predictor=self.transport_predictor,
                        ffn_expansion=ffn_expansion,
                        value_modulation=self.value_modulation,
                    )
                )
                layer_stage_indices.append(stage_index)
                layer_strides.append(layer_stride)
                layer_pixel_strides.append(stage_pixel_stride)
        self.layer_stage_indices = tuple(layer_stage_indices)
        self.layer_strides = tuple(layer_strides)
        self.layer_pixel_strides = tuple(layer_pixel_strides)
        self.layers = nn.ModuleList(layers)
        self.latent_blend_norm: nn.Module = (
            ChannelLayerNorm2d(channels[0], affine=False)
            if self.normalize_latent_blend
            else nn.Identity()
        )
        self.output = PointwiseMLP2d(
            in_channels=channels[-1],
            hidden_channels=channels[-1],
            out_channels=out_channels,
        )

    @property
    def axis_window_px(self) -> int:
        """Maximum one-axis pixel window reachable by one latent cell."""
        return int(
            self.feature_stride
            + 2
            * sum(
                blocks * stride
                for blocks, stride in zip(
                    self.stage_blocks,
                    self.stage_pixel_strides,
                    strict=True,
                )
            )
        )

    def forward(
        self,
        foreground: Tensor,
        background: Tensor,
        foreground_presence: Tensor,
        *,
        layer_transports: tuple[PyramidTransport, ...] | None = None,
    ) -> tuple[Tensor, tuple[PyramidTransport, ...]]:
        """Decode latent streams and return one transport per decoder layer."""
        _validate_spatial_tensor("foreground", foreground, channels=self.in_channels)
        background_channels = (
            self.channels[-1] if self.dual_stream else self.in_channels
        )
        _validate_spatial_tensor("background", background, channels=background_channels)
        _validate_spatial_tensor("foreground_presence", foreground_presence, channels=1)
        if self.dual_stream:
            _validate_matching_batch_and_spatial(
                (
                    ("foreground", foreground),
                    ("foreground_presence", foreground_presence),
                )
            )
            if int(background.shape[0]) != int(foreground.shape[0]):
                raise ValueError(
                    "foreground and background batch sizes must match, got "
                    f"{foreground.shape[0]} and {background.shape[0]}"
                )
        else:
            _validate_matching_batch_and_spatial(
                (
                    ("foreground", foreground),
                    ("background", background),
                    ("foreground_presence", foreground_presence),
                )
            )
        if layer_transports is not None and len(layer_transports) != len(self.layers):
            raise ValueError(
                "layer_transports length must match decoder layers, got "
                f"{len(layer_transports)} and {len(self.layers)}"
            )

        foreground_out = self.latent_blend_norm(foreground)
        background_out = (
            background if self.dual_stream else self.latent_blend_norm(background)
        )
        presence_out = foreground_presence.to(dtype=foreground_out.dtype)
        out = (
            presence_out * foreground_out
            if self.dual_stream
            else presence_out * foreground_out + (1.0 - presence_out) * background_out
        )
        transports: list[PyramidTransport] = []
        for layer_index, layer in enumerate(self.layers):
            fixed_transport = None
            layer_typed = cast(PyramidFlowDecoderLayer2d, layer)
            if layer_transports is not None:
                fixed_transport = layer_transports[layer_index]
                self._validate_layer_transport(
                    fixed_transport,
                    batch_size=int(foreground.shape[0]),
                    fine_hw=(
                        int(out.shape[-2]) * layer_typed.stride,
                        int(out.shape[-1]) * layer_typed.stride,
                    ),
                    coarse_hw=(int(out.shape[-2]), int(out.shape[-1])),
                    pixel_stride=self.layer_pixel_strides[layer_index],
                )
            out, presence_out, transport = layer_typed(
                x=out,
                foreground_presence=presence_out,
                transport=fixed_transport,
            )
            transports.append(transport)
        if self.dual_stream:
            if tuple(background_out.shape[-2:]) != tuple(out.shape[-2:]):
                raise ValueError(
                    "dual-stream background features must be sampled on the "
                    f"decoder output grid {tuple(out.shape[-2:])}, got "
                    f"{tuple(background_out.shape[-2:])}"
                )
            foreground_out = self._unit_normalize_pixel_features(out)
            background_out = self._unit_normalize_pixel_features(background_out)
            out = self._variance_preserving_pixel_blend(
                foreground=foreground_out,
                background=background_out,
                foreground_presence=presence_out,
            )
        return self.output(out), tuple(transports)

    @staticmethod
    def _unit_normalize_pixel_features(features: Tensor) -> Tensor:
        """Return finite per-pixel unit vectors while preserving exact zeros."""
        normalized = F.normalize(features.float(), p=2.0, dim=1, eps=1e-6)
        return normalized.to(dtype=features.dtype)

    @staticmethod
    def _variance_preserving_pixel_blend(
        *, foreground: Tensor, background: Tensor, foreground_presence: Tensor
    ) -> Tensor:
        """Mix pixel features without changing variance under independent streams."""
        presence = foreground_presence.float()
        background_presence = 1.0 - presence
        denominator = (presence.square() + background_presence.square()).sqrt()
        mixed = (
            presence * foreground.float() + background_presence * background.float()
        ) / denominator
        return mixed.to(dtype=foreground.dtype)

    @staticmethod
    def _validate_layer_transport(
        transport: PyramidTransport,
        *,
        batch_size: int,
        fine_hw: tuple[int, int],
        coarse_hw: tuple[int, int],
        pixel_stride: int,
    ) -> None:
        if int(transport.probs.shape[0]) != batch_size:
            raise ValueError(
                "transport batch size must be "
                f"{batch_size}, got {transport.probs.shape[0]}"
            )
        if transport.lookup.fine_hw != fine_hw:
            raise ValueError(
                f"transport fine grid must be {fine_hw}, got {transport.lookup.fine_hw}"
            )
        if transport.lookup.coarse_hw != coarse_hw:
            raise ValueError(
                "transport coarse grid must be "
                f"{coarse_hw}, got {transport.lookup.coarse_hw}"
            )
        if int(transport.pixel_stride) != int(pixel_stride):
            raise ValueError(
                "transport pixel_stride must be "
                f"{pixel_stride}, got {transport.pixel_stride}"
            )


class PyramidFlowSystem(
    TypedLightningModule[
        TemporalBatch,
        [Tensor],
        PyramidFlowOutput,
        Tensor,
        SegmentationPredictionPayload,
    ]
):
    """
    Pyramid-flow VAE with independent foreground and background latents.

    Parameters
    ----------
    encoder : nn.Module
        Encoder producing deterministic spatial features.
    decoder : PyramidFlowDecoder2d
        Decoder consuming foreground and background latent grids.
    foreground_latent_hidden_dim : int
        Hidden width of the foreground posterior projection MLP.
    background_latent_hidden_dim : int
        Hidden width of the background posterior projection MLP.
    background_latent_pool_stride : int
        Stride of the average-pool and bilinear-upsample bottleneck applied to
        encoder features before the background posterior head.
    patch_size : int
        Required input divisibility and encoder feature stride.
    foreground_sparsity_alpha : float
        Target foreground occupancy used by the sparsity regularizer.
    latent_dim : int or None
        Width of the foreground latent and of the background latent for a
        single-stream decoder. A dual-stream background posterior instead uses
        the decoder's final feature width so it can be sampled after upsampling
        without a learned post-sampling projection. If None, use the decoder
        input width.
    latent_head_norm : str | tuple[str, dict[str, Any]]
        Normalization applied at the hidden layer of both posterior MLPs.
    recon_loss_weight : LossWeightSchedule
        Schedule for the reconstruction-loss weight.
    foreground_kl_loss_weight : LossWeightSchedule
        Schedule for the foreground KL-loss weight.
    background_kl_loss_weight : LossWeightSchedule
        Schedule for the background KL-loss weight.
    flow_l2_loss_weight : LossWeightSchedule
        Schedule for the squared-flow-loss weight.
    foreground_sparsity_loss_weight : LossWeightSchedule
        Schedule for the foreground-sparsity-loss weight.
    loss_type : {"l1", "l2"}
        Pointwise reconstruction loss.
    dense_assignment_max_elements : int
        Maximum number of elements for materializing a dense assignment map.
    eps : float
        Numerical stability constant used by probability computations.
    lr : float
        Peak learning rate.
    weight_decay : float
        AdamW weight decay.
    adam_betas : tuple[float, float]
        AdamW exponential-decay coefficients.
    adam_eps : float
        AdamW numerical stability constant.
    warmup_epochs : int
        Number of linear learning-rate warmup epochs.
    log_images_every_n_epochs : int
        Interval between reconstruction image logs. Zero disables them.
    log_image_samples : int
        Maximum number of samples in each reconstruction image log.
    log_decoder_ablation_every_n_epochs : int
        Interval between decoder-ablation logs. Zero disables them.
    log_decoder_ablation_max_batch_size : int
        Maximum batch size used for decoder-ablation logs. Must be at least two
        when decoder-ablation logging is enabled.
    segmentation_test_config : PyramidFlowSegmentationTestConfig or None
        Optional flow-induced segmentation evaluation settings.
    """

    # ruff: disable[B008]
    def __init__(
        self,
        *,
        encoder: nn.Module,
        decoder: PyramidFlowDecoder2d,
        foreground_latent_hidden_dim: int,
        background_latent_hidden_dim: int,
        background_latent_pool_stride: int = 1,
        patch_size: int = 8,
        foreground_sparsity_alpha: float = 0.5,
        latent_dim: int | None = None,
        latent_head_norm: Norm2d = "INSTANCE",
        recon_loss_weight: LossWeightSchedule = LossWeightSchedule(1.0, 1.0, 0),
        foreground_kl_loss_weight: LossWeightSchedule = LossWeightSchedule(
            0.0, 1e-4, 100
        ),
        background_kl_loss_weight: LossWeightSchedule = LossWeightSchedule(
            0.0, 1e-4, 100
        ),
        flow_l2_loss_weight: LossWeightSchedule = LossWeightSchedule(0.0, 1e-4, 100),
        foreground_sparsity_loss_weight: LossWeightSchedule = LossWeightSchedule(
            0.0, 1e-3, 100
        ),
        loss_type: Literal["l1", "l2"] = "l2",
        dense_assignment_max_elements: int = 2_000_000,
        eps: float = 1e-6,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        warmup_epochs: int = 0,
        log_images_every_n_epochs: int = 1,
        log_image_samples: int = 4,
        log_decoder_ablation_every_n_epochs: int = 0,
        log_decoder_ablation_max_batch_size: int = 8,
        segmentation_test_config: PyramidFlowSegmentationTestConfig | None = None,
    ) -> None:
        super().__init__()
        segmentation_test_config = self._validate_segmentation_test_config(
            segmentation_test_config
        )
        recon_loss_weight = self._validate_loss_weight_schedule(
            "recon_loss_weight", recon_loss_weight
        )
        foreground_kl_loss_weight = self._validate_loss_weight_schedule(
            "foreground_kl_loss_weight", foreground_kl_loss_weight
        )
        background_kl_loss_weight = self._validate_loss_weight_schedule(
            "background_kl_loss_weight", background_kl_loss_weight
        )
        flow_l2_loss_weight = self._validate_loss_weight_schedule(
            "flow_l2_loss_weight", flow_l2_loss_weight
        )
        foreground_sparsity_loss_weight = self._validate_loss_weight_schedule(
            "foreground_sparsity_loss_weight", foreground_sparsity_loss_weight
        )
        foreground_latent_hidden_dim = _positive_int(
            "foreground_latent_hidden_dim", foreground_latent_hidden_dim
        )
        background_latent_hidden_dim = _positive_int(
            "background_latent_hidden_dim", background_latent_hidden_dim
        )
        background_latent_pool_stride = _positive_int(
            "background_latent_pool_stride", background_latent_pool_stride
        )
        warmup_epochs = _nonnegative_int("warmup_epochs", warmup_epochs)
        log_decoder_ablation_every_n_epochs = _nonnegative_int(
            "log_decoder_ablation_every_n_epochs",
            log_decoder_ablation_every_n_epochs,
        )
        log_decoder_ablation_max_batch_size = _positive_int(
            "log_decoder_ablation_max_batch_size",
            log_decoder_ablation_max_batch_size,
        )
        if (
            log_decoder_ablation_every_n_epochs > 0
            and log_decoder_ablation_max_batch_size < 2
        ):
            raise ValueError(
                "log_decoder_ablation_max_batch_size must be at least 2 when "
                "decoder-ablation logging is enabled, got "
                f"{log_decoder_ablation_max_batch_size}"
            )
        self.save_hyperparameters(ignore=["encoder", "decoder"])

        encoder_dim = self._required_int_attr(encoder, "dim", "encoder")
        encoder_feature_stride = self._required_int_attr(
            encoder, "feature_stride", "encoder"
        )
        decoder_in_channels = self._required_int_attr(decoder, "in_channels", "decoder")
        decoder_out_channels = self._required_int_attr(
            decoder, "out_channels", "decoder"
        )
        encoder_in_channels = self._required_int_attr(encoder, "in_channels", "encoder")
        latent_dim_value = _positive_int(
            "latent_dim", decoder_in_channels if latent_dim is None else latent_dim
        )
        patch_size = _positive_int("patch_size", patch_size)
        dense_assignment_max_elements = _positive_int(
            "dense_assignment_max_elements", dense_assignment_max_elements
        )
        if decoder_in_channels != latent_dim_value:
            raise ValueError(
                "decoder in_channels must match latent_dim for direct alpha blending, "
                f"got {decoder_in_channels} and {latent_dim_value}"
            )
        if decoder_out_channels != encoder_in_channels:
            raise ValueError(
                "decoder out_channels must match encoder input channels, got "
                f"{decoder_out_channels} and {encoder_in_channels}"
            )
        if patch_size != encoder_feature_stride:
            raise ValueError(
                "patch_size must equal encoder feature_stride, got "
                f"patch_size={patch_size} and feature_stride={encoder_feature_stride}"
            )
        for name, value in [
            ("foreground_sparsity_alpha", foreground_sparsity_alpha),
            ("eps", eps),
            ("lr", lr),
            ("adam_eps", adam_eps),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if foreground_sparsity_alpha >= 1.0:
            raise ValueError(
                "foreground_sparsity_alpha should be in (0, 1) to reward sparse "
                f"clusters, got {foreground_sparsity_alpha}"
            )
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        if loss_type not in {"l1", "l2"}:
            raise ValueError(f"loss_type must be 'l1' or 'l2', got {loss_type!r}")

        self.encoder = encoder
        self.decoder = decoder
        self.in_channels = encoder_in_channels
        self.encoder_dim = encoder_dim
        self.dim = latent_dim_value
        self.latent_dim = latent_dim_value
        self.foreground_latent_dim = latent_dim_value
        self.background_latent_dim = (
            decoder.channels[-1] if decoder.dual_stream else latent_dim_value
        )
        self.decoder_in_channels = decoder_in_channels
        self.patch_size = patch_size
        self.foreground_sparsity_alpha = float(foreground_sparsity_alpha)
        self.foreground_latent_hidden_dim = foreground_latent_hidden_dim
        self.background_latent_hidden_dim = background_latent_hidden_dim
        self.background_latent_pool_stride = background_latent_pool_stride
        self.recon_loss_weight = recon_loss_weight
        self.foreground_kl_loss_weight = foreground_kl_loss_weight
        self.background_kl_loss_weight = background_kl_loss_weight
        self.flow_l2_loss_weight = flow_l2_loss_weight
        self.foreground_sparsity_loss_weight = foreground_sparsity_loss_weight
        self.loss_type = loss_type
        self.dense_assignment_max_elements = dense_assignment_max_elements
        self.eps = float(eps)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.adam_betas = adam_betas
        self.adam_eps = float(adam_eps)
        self.warmup_epochs = warmup_epochs
        self.log_images_every_n_epochs = int(log_images_every_n_epochs)
        self.log_image_samples = int(log_image_samples)
        self.log_decoder_ablation_every_n_epochs = log_decoder_ablation_every_n_epochs
        self.log_decoder_ablation_max_batch_size = log_decoder_ablation_max_batch_size
        self.segmentation_test_config = segmentation_test_config
        self.foreground_latent_head = PointwiseMLP2d(
            in_channels=encoder_dim,
            hidden_channels=foreground_latent_hidden_dim,
            out_channels=2 * latent_dim_value,
            hidden_norm=_make_norm(latent_head_norm, foreground_latent_hidden_dim),
        )
        self.background_latent_head = PointwiseMLP2d(
            in_channels=encoder_dim,
            hidden_channels=background_latent_hidden_dim,
            out_channels=2 * self.background_latent_dim,
            hidden_norm=_make_norm(latent_head_norm, background_latent_hidden_dim),
        )
        self.foreground_presence_head = nn.Conv2d(latent_dim_value, 1, kernel_size=1)
        self._reconstruction_input_divisibility = (
            self.patch_size * self.background_latent_pool_stride
        )
        self._segmentation_input_divisibility = self._reconstruction_input_divisibility
        self._input_divisibility_stride = self._reconstruction_input_divisibility

    def forward(
        self, image: Tensor, *, return_dense_assignment: bool = False
    ) -> PyramidFlowOutput:
        """Reconstruct an image and return pyramid-flow diagnostics."""
        _validate_spatial_tensor("image", image, channels=self.in_channels)
        encoder_output = self.encoder(image)
        if not isinstance(encoder_output, PyramidFlowEncoderOutput):
            raise TypeError(
                "encoder must return PyramidFlowEncoderOutput with deterministic "
                "bottleneck features"
            )
        encoder_features = encoder_output.features
        if int(encoder_features.shape[1]) != self.encoder_dim:
            raise RuntimeError(
                "encoder feature width does not match model encoder_dim, got "
                f"{tuple(encoder_features.shape)}"
            )
        foreground_latent_mu, foreground_latent_logvar = torch.chunk(
            self.foreground_latent_head(encoder_features), chunks=2, dim=1
        )
        background_encoder_features = self._pool_background_encoder_features(
            encoder_features
        )
        background_latent_mu, background_latent_logvar = torch.chunk(
            self.background_latent_head(background_encoder_features), chunks=2, dim=1
        )
        foreground_latents = self._sample_gaussian(
            foreground_latent_mu,
            foreground_latent_logvar,
            sample=self.training,
        )
        if self.decoder.dual_stream:
            background_latent_mu = F.interpolate(
                background_latent_mu.float(),
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(dtype=encoder_features.dtype)
            background_latent_logvar = F.interpolate(
                background_latent_logvar.float(),
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(dtype=encoder_features.dtype)
        background_latents = self._sample_gaussian(
            background_latent_mu,
            background_latent_logvar,
            sample=self.training,
        )
        foreground_logits = self.foreground_presence_head(foreground_latents)
        foreground_presence = torch.sigmoid(foreground_logits.float()).to(
            dtype=foreground_latents.dtype
        )
        recon, layer_transports = self.decoder(
            foreground_latents,
            background_latents,
            foreground_presence,
        )
        if tuple(recon.shape) != tuple(image.shape):
            raise RuntimeError(
                "decoder reconstruction shape must match image shape, got "
                f"{tuple(recon.shape)} and {tuple(image.shape)}"
            )
        p_fg = self._project_center_presence_to_pixels(
            center_presence=foreground_presence, layer_transports=layer_transports
        )
        expected_flow = self._expected_flow(
            layer_transports=layer_transports,
            output_hw=(int(image.shape[-2]), int(image.shape[-1])),
            device=image.device,
            dtype=image.dtype,
        )
        foreground_sparsity = self.foreground_sparsity_loss(foreground_presence)
        foreground_kl, foreground_kl_total = self._latent_kl_mean_and_total(
            foreground_latent_mu, foreground_latent_logvar
        )
        background_kl, background_kl_total = self._latent_kl_mean_and_total(
            background_latent_mu, background_latent_logvar
        )
        flow_l2, flow_l2_total = self._flow_l2_mean_and_total(layer_transports)
        dense_assignment = (
            self._dense_assignment(
                layer_transports=layer_transports,
                output_hw=(int(image.shape[-2]), int(image.shape[-1])),
                dtype=image.dtype,
            )
            if return_dense_assignment
            else None
        )
        return {
            "recon": recon,
            "foreground_logits": foreground_logits,
            "foreground_presence": foreground_presence,
            "p_fg": p_fg,
            "foreground_sparsity": foreground_sparsity,
            "foreground_kl": foreground_kl,
            "background_kl": background_kl,
            "flow_l2": flow_l2,
            "foreground_kl_total": foreground_kl_total,
            "background_kl_total": background_kl_total,
            "flow_l2_total": flow_l2_total,
            "expected_flow": expected_flow,
            "foreground_latents": foreground_latents,
            "foreground_latent_mu": foreground_latent_mu,
            "foreground_latent_logvar": foreground_latent_logvar,
            "background_latents": background_latents,
            "background_latent_mu": background_latent_mu,
            "background_latent_logvar": background_latent_logvar,
            "foreground_features": foreground_latents,
            "background_features": background_latents,
            "layer_transports": layer_transports,
            "dense_assignment": dense_assignment,
        }

    def _pool_background_encoder_features(self, encoder_features: Tensor) -> Tensor:
        stride = self.background_latent_pool_stride
        if stride == 1:
            return encoder_features
        height, width = (int(size) for size in encoder_features.shape[-2:])
        if height % stride != 0 or width % stride != 0:
            raise ValueError(
                "encoder feature spatial shape must be divisible by "
                f"background_latent_pool_stride={stride}, got {(height, width)}"
            )
        pooled = F.avg_pool2d(
            encoder_features.float(), kernel_size=stride, stride=stride
        )
        upsampled = F.interpolate(
            pooled,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return upsampled.to(dtype=encoder_features.dtype)

    def _get_attached_trainer(self) -> Any | None:
        try:
            return self.trainer
        except RuntimeError:
            return None

    def _get_warmup_schedule_steps(self) -> tuple[int, int] | None:
        trainer = self._get_attached_trainer()
        if trainer is None:
            return None
        raw_total_steps = trainer.estimated_stepping_batches
        try:
            total_steps = int(raw_total_steps)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "trainer.estimated_stepping_batches must be an integer-like value "
                f"for warmup scheduling, got {raw_total_steps!r}"
            ) from exc
        if total_steps < 0:
            raise ValueError(
                "trainer.estimated_stepping_batches must be non-negative for "
                f"warmup scheduling, got {total_steps}"
            )
        max_epochs = trainer.max_epochs
        if not isinstance(max_epochs, (int, float)):
            raise TypeError(
                "trainer.max_epochs must be numeric for warmup scheduling, got "
                f"{max_epochs!r}"
            )
        if max_epochs <= 0:
            if self.warmup_epochs == 0:
                return total_steps, 0
            raise ValueError(
                "trainer.max_epochs must be positive when warmup_epochs > 0, got "
                f"{max_epochs}"
            )
        warmup_steps = int(total_steps * (self.warmup_epochs / float(max_epochs)))
        if self.warmup_epochs > 0 and warmup_steps <= 0:
            raise ValueError(
                "Warmup schedule produced no warmup steps; check warmup_epochs, "
                f"trainer.max_epochs={max_epochs}, and "
                f"trainer.estimated_stepping_batches={total_steps}."
            )
        return total_steps, max(warmup_steps, 0)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=self.adam_betas,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )
        schedule_steps = self._get_warmup_schedule_steps()
        if schedule_steps is None:
            return optimizer
        total_steps, warmup_steps = schedule_steps
        if warmup_steps <= 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_steps), eta_min=0
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=max(1, warmup_steps),
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(total_steps - warmup_steps)), eta_min=0
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, cosine], milestones=[warmup_steps]
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @staticmethod
    def _validate_loss_weight_schedule(
        name: str, schedule: LossWeightSchedule
    ) -> LossWeightSchedule:
        start_weight = float(schedule.start_weight)
        end_weight = float(schedule.end_weight)
        warmup_epochs = int(schedule.warmup_epochs)
        if start_weight < 0:
            raise ValueError(f"{name}.start_weight must be non-negative")
        if end_weight < 0:
            raise ValueError(f"{name}.end_weight must be non-negative")
        if warmup_epochs < 0:
            raise ValueError(f"{name}.warmup_epochs must be non-negative")
        return LossWeightSchedule(
            start_weight=start_weight,
            end_weight=end_weight,
            warmup_epochs=warmup_epochs,
        )

    @staticmethod
    def _validate_segmentation_test_config(
        config: PyramidFlowSegmentationTestConfig | None,
    ) -> PyramidFlowSegmentationTestConfig | None:
        if config is None:
            return None
        center_threshold = _unit_interval_float(
            "center_threshold", config.center_threshold
        )
        pixel_mass_threshold = _unit_interval_float(
            "pixel_mass_threshold", config.pixel_mass_threshold
        )
        component_chunk_size = _positive_int(
            "component_chunk_size", config.component_chunk_size
        )
        min_object_area = (
            None
            if config.min_object_area is None
            else _positive_int("min_object_area", config.min_object_area)
        )
        max_object_area = (
            None
            if config.max_object_area is None
            else _positive_int("max_object_area", config.max_object_area)
        )
        if (
            min_object_area is not None
            and max_object_area is not None
            and min_object_area > max_object_area
        ):
            raise ValueError(
                "min_object_area must be <= max_object_area, got "
                f"{min_object_area} and {max_object_area}"
            )
        prediction_output_scale_factor = validate_scale_factor(
            config.prediction_output_scale_factor
        )
        return PyramidFlowSegmentationTestConfig(
            center_threshold=center_threshold,
            pixel_mass_threshold=pixel_mass_threshold,
            component_chunk_size=component_chunk_size,
            min_object_area=min_object_area,
            max_object_area=max_object_area,
            prediction_output_scale_factor=prediction_output_scale_factor,
        )

    @staticmethod
    def _required_int_attr(module: nn.Module, attr_name: str, module_name: str) -> int:
        value = getattr(module, attr_name, None)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise TypeError(
                    f"{module_name}.{attr_name} tensor must be scalar when provided"
                )
            return int(value.item())
        if isinstance(value, int):
            return int(value)
        raise TypeError(f"{module_name} must expose integer attribute `{attr_name}`")

    def _project_center_presence_to_pixels(
        self, *, center_presence: Tensor, layer_transports: tuple[PyramidTransport, ...]
    ) -> Tensor:
        p_fg = center_presence
        for transport in layer_transports:
            p_fg = self._gather_scalar_map(p_fg, transport=transport)
        return p_fg

    def foreground_sparsity_loss(self, foreground_presence: Tensor) -> Tensor:
        """Smooth monotone concave penalty for sparse foreground probabilities."""
        _validate_spatial_tensor("foreground_presence", foreground_presence, channels=1)
        presence = foreground_presence.float()
        shifted_power = (presence + self.eps).pow(self.foreground_sparsity_alpha) - (
            self.eps**self.foreground_sparsity_alpha
        )
        return shifted_power.mean().to(dtype=foreground_presence.dtype)

    def _dense_assignment(
        self,
        *,
        layer_transports: tuple[PyramidTransport, ...],
        output_hw: tuple[int, int],
        dtype: torch.dtype,
    ) -> Tensor:
        output_h, output_w = int(output_hw[0]), int(output_hw[1])
        num_pixels = output_h * output_w
        latent_h, latent_w = layer_transports[0].lookup.coarse_hw
        num_latents = int(latent_h) * int(latent_w)
        max_intermediate = num_pixels * num_pixels
        final_elements = (
            int(layer_transports[-1].probs.shape[0]) * num_pixels * num_latents
        )
        if (
            max_intermediate > self.dense_assignment_max_elements
            or final_elements > self.dense_assignment_max_elements
        ):
            raise ValueError(
                "dense assignment would be too large; increase "
                "dense_assignment_max_elements only for small diagnostic batches"
            )
        device = layer_transports[-1].probs.device
        assignment = torch.eye(num_pixels, device=device, dtype=dtype).unsqueeze(0)
        assignment = assignment.expand(int(layer_transports[-1].probs.shape[0]), -1, -1)
        for transport in reversed(layer_transports):
            assignment = self._compose_dense_assignment_layer(
                assignment=assignment,
                transport=transport,
            )
        return rearrange(
            assignment,
            "b (h w) l -> b l h w",
            h=output_h,
            w=output_w,
        )

    @staticmethod
    def _compose_dense_assignment_layer(
        *, assignment: Tensor, transport: PyramidTransport
    ) -> Tensor:
        batch_size, num_pixels, num_fine = assignment.shape
        coarse_h, coarse_w = transport.lookup.coarse_hw
        num_coarse = int(coarse_h) * int(coarse_w)
        probs = rearrange(transport.probs.float(), "b k h w -> b (h w) k")
        contribution = assignment.float().unsqueeze(3) * probs.unsqueeze(1)
        indices = transport.lookup.target_indices.to(device=assignment.device).view(
            1, 1, num_fine, 9
        )
        indices = indices.expand(batch_size, num_pixels, -1, -1)
        out = assignment.new_zeros(batch_size, num_pixels, num_coarse)
        out.scatter_add_(
            dim=2,
            index=indices.reshape(batch_size, num_pixels, -1),
            src=contribution.reshape(batch_size, num_pixels, -1).to(
                dtype=assignment.dtype
            ),
        )
        return out

    def _expected_flow(
        self,
        *,
        layer_transports: tuple[PyramidTransport, ...],
        output_hw: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        latent_h, latent_w = layer_transports[0].lookup.coarse_hw
        endpoint = self._pixel_center_coordinates(
            grid_hw=(latent_h, latent_w),
            pixel_stride=self.patch_size,
            device=device,
            dtype=dtype,
        )
        for transport in layer_transports:
            endpoint = self._gather_coordinate_map(endpoint, transport=transport)
        pixel_coords = self._pixel_center_coordinates(
            grid_hw=output_hw,
            pixel_stride=1,
            device=device,
            dtype=dtype,
        )
        return endpoint - pixel_coords

    @staticmethod
    def _pixel_center_coordinates(
        *,
        grid_hw: tuple[int, int],
        pixel_stride: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        height, width = int(grid_hw[0]), int(grid_hw[1])
        stride = int(pixel_stride)
        y = torch.arange(height, device=device, dtype=dtype) * float(
            stride
        ) + 0.5 * float(stride - 1)
        x = torch.arange(width, device=device, dtype=dtype) * float(
            stride
        ) + 0.5 * float(stride - 1)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx), dim=0).unsqueeze(0)

    @staticmethod
    def _gather_coordinate_map(
        coords: Tensor, *, transport: PyramidTransport
    ) -> Tensor:
        batch_size = int(transport.probs.shape[0])
        _, coord_channels, _, _ = coords.shape
        fine_h, fine_w = transport.lookup.fine_hw
        indices = transport.lookup.target_indices.to(device=coords.device).reshape(-1)
        flat = rearrange(coords, "b c h w -> b c (h w)")
        if int(flat.shape[0]) == 1 and batch_size > 1:
            flat = flat.expand(batch_size, -1, -1)
        gathered = flat.index_select(dim=2, index=indices)
        gathered = gathered.view(batch_size, coord_channels, fine_h * fine_w, 9)
        probs = rearrange(transport.probs, "b k h w -> b 1 (h w) k")
        out = (gathered * probs).sum(dim=3)
        return rearrange(out, "b c (h w) -> b c h w", h=fine_h, w=fine_w)

    @staticmethod
    def _gather_feature_map(values: Tensor, *, transport: PyramidTransport) -> Tensor:
        return TransportResidualBlock2d.spatially_varying_convolution(
            values, transport=transport
        )

    @staticmethod
    def _gather_scalar_map(values: Tensor, *, transport: PyramidTransport) -> Tensor:
        return PyramidFlowSystem._gather_feature_map(values, transport=transport)

    @staticmethod
    def _slice_layer_transports_batch(
        layer_transports: tuple[PyramidTransport, ...], batch_index: int
    ) -> tuple[PyramidTransport, ...]:
        def slice_tensor(value: Tensor) -> Tensor:
            if value.ndim > 0 and int(value.shape[0]) > int(batch_index):
                return value[batch_index : batch_index + 1]
            return value

        return tuple(
            PyramidTransport(
                probs=slice_tensor(transport.probs),
                logits=slice_tensor(transport.logits),
                expected_offset=slice_tensor(transport.expected_offset),
                lookup=transport.lookup,
                pixel_stride=transport.pixel_stride,
            )
            for transport in layer_transports
        )

    def flow_induced_instance_labels(
        self,
        *,
        center_presence: Tensor,
        layer_transports: tuple[PyramidTransport, ...],
        center_threshold: float,
        pixel_mass_threshold: float,
        component_chunk_size: int = 16,
        min_object_area: int | None = None,
        max_object_area: int | None = None,
        output_hw: tuple[int, int] | None = None,
    ) -> PyramidFlowSegmentationOutput:
        """Convert center probabilities and pyramid transports into instance labels.

        Parameters
        ----------
        center_presence
            Bottleneck center probabilities with shape ``(B, 1, H, W)``.
        layer_transports
            Coarse-to-fine transport kernels used to route component mass.
        center_threshold
            Probability threshold used to form connected center components.
        pixel_mass_threshold
            Minimum winning mass required to label an output pixel.
        component_chunk_size
            Number of components propagated together. Larger values trade memory
            for fewer accelerator kernel launches.
        min_object_area
            Optional minimum retained object area in output pixels.
        max_object_area
            Optional maximum retained object area in output pixels.
        output_hw
            Optional spatial crop applied before area filtering.

        Returns
        -------
        PyramidFlowSegmentationOutput
            Center components, unfiltered and area-filtered instance labels, and
            the winning propagated mass for every output pixel.
        """
        _validate_spatial_tensor("center_presence", center_presence, channels=1)
        if not layer_transports:
            raise ValueError("layer_transports must be non-empty")
        component_chunk_size = _positive_int(
            "component_chunk_size", component_chunk_size
        )

        center_components: list[Tensor] = []
        pred_unfiltered: list[Tensor] = []
        pred_filtered: list[Tensor] = []
        winning_masses: list[Tensor] = []
        batch_size = int(center_presence.shape[0])
        for batch_index in range(batch_size):
            single_transports = self._slice_layer_transports_batch(
                layer_transports, batch_index
            )
            components, labels, mass = self._flow_induced_single_label_image(
                center_presence=center_presence[batch_index : batch_index + 1],
                layer_transports=single_transports,
                center_threshold=center_threshold,
                pixel_mass_threshold=pixel_mass_threshold,
                component_chunk_size=component_chunk_size,
            )
            if output_hw is not None:
                labels = labels[: output_hw[0], : output_hw[1]]
                mass = mass[: output_hw[0], : output_hw[1]]
            filtered = _filter_label_tensor_by_area(
                labels,
                min_object_area=min_object_area,
                max_object_area=max_object_area,
            )
            center_components.append(components)
            pred_unfiltered.append(labels)
            pred_filtered.append(filtered)
            winning_masses.append(mass)

        return PyramidFlowSegmentationOutput(
            center_components=torch.stack(center_components),
            pred_labels_unfiltered=torch.stack(pred_unfiltered),
            pred_labels=torch.stack(pred_filtered),
            winning_mass=torch.stack(winning_masses),
        )

    def _flow_induced_single_label_image(
        self,
        *,
        center_presence: Tensor,
        layer_transports: tuple[PyramidTransport, ...],
        center_threshold: float,
        pixel_mass_threshold: float,
        component_chunk_size: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        center_np = center_presence[0, 0].detach().cpu().float().numpy()
        center_components_np = label(center_np >= center_threshold).astype(np.int32)
        num_components = int(center_components_np.max())
        device = center_presence.device
        center_components = torch.as_tensor(
            center_components_np,
            device=device,
            dtype=torch.int32,
        )
        output_h, output_w = layer_transports[-1].lookup.fine_hw
        if num_components == 0:
            empty_mass = torch.zeros(
                output_h, output_w, dtype=torch.float32, device=device
            )
            empty_labels = torch.zeros(
                output_h, output_w, dtype=torch.int32, device=device
            )
            return center_components, empty_labels, empty_mass

        component_ids = torch.arange(
            1,
            num_components + 1,
            device=device,
            dtype=torch.int32,
        )
        component_map = center_components.view(1, 1, *center_components.shape)
        winning_mass: Tensor | None = None
        winning_label: Tensor | None = None
        for chunk_start in range(0, num_components, component_chunk_size):
            chunk_ids = component_ids[chunk_start : chunk_start + component_chunk_size]
            component_mask = component_map == chunk_ids.view(1, -1, 1, 1)
            mass = torch.where(
                component_mask,
                center_presence.float(),
                0.0,
            )
            for transport in layer_transports:
                mass = self._gather_scalar_map(mass, transport=transport)
            chunk_mass, chunk_index = mass[0].max(dim=0)
            chunk_label = chunk_ids[chunk_index]
            if winning_mass is None or winning_label is None:
                winning_mass = chunk_mass
                winning_label = chunk_label
                continue
            replace = chunk_mass > winning_mass
            winning_mass = torch.where(replace, chunk_mass, winning_mass)
            winning_label = torch.where(replace, chunk_label, winning_label)

        assert winning_mass is not None and winning_label is not None
        pred_labels = torch.where(
            winning_mass >= float(pixel_mass_threshold),
            winning_label,
            0,
        ).to(dtype=torch.int32)
        return center_components, pred_labels, winning_mass.float()

    def _reconstruction_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        return F.mse_loss(pred, target)

    @staticmethod
    def _sample_gaussian(mu: Tensor, logvar: Tensor, *, sample: bool) -> Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar.clamp(min=-30.0, max=20.0))
        return mu + torch.randn_like(mu) * std

    @staticmethod
    def _kl_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
        mu = mu.float()
        logvar = logvar.float().clamp(min=-30.0, max=20.0)
        return 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar)

    @staticmethod
    def _latent_kl_mean_and_total(mu: Tensor, logvar: Tensor) -> tuple[Tensor, Tensor]:
        kl = PyramidFlowSystem._kl_standard_normal(mu, logvar)
        return kl.mean(), kl.sum(dim=(1, 2, 3)).mean()

    @staticmethod
    def _transport_step_squared_distance(transport: PyramidTransport) -> Tensor:
        """Squared pixel distance for each valid fine-to-coarse transport edge."""
        fine_h, fine_w = transport.lookup.fine_hw
        coarse_h, coarse_w = transport.lookup.coarse_hw
        scale_h = fine_h // coarse_h
        scale_w = fine_w // coarse_w
        coarse_coords = PyramidFlowSystem._pixel_center_coordinates_2d(
            grid_hw=transport.lookup.coarse_hw,
            pixel_stride_y=int(transport.pixel_stride) * scale_h,
            pixel_stride_x=int(transport.pixel_stride) * scale_w,
            device=transport.probs.device,
            dtype=torch.float32,
        )
        fine_coords = PyramidFlowSystem._pixel_center_coordinates_2d(
            grid_hw=transport.lookup.fine_hw,
            pixel_stride_y=int(transport.pixel_stride),
            pixel_stride_x=int(transport.pixel_stride),
            device=transport.probs.device,
            dtype=torch.float32,
        )
        indices = transport.lookup.target_indices.to(device=transport.probs.device)
        flat_coarse = rearrange(coarse_coords, "b c h w -> b c (h w)")
        gathered = flat_coarse.index_select(dim=2, index=indices.reshape(-1))
        gathered = gathered.view(1, 2, fine_h * fine_w, 9)
        gathered = rearrange(gathered, "b c (h w) k -> b c k h w", h=fine_h, w=fine_w)
        delta = gathered - fine_coords.unsqueeze(2)
        valid = transport.lookup.valid_edges.T.reshape(1, 9, fine_h, fine_w).to(
            device=transport.probs.device
        )
        return delta.square().sum(dim=1).masked_fill(~valid, 0.0)

    @staticmethod
    def _flow_l2_mean_and_total(
        layer_transports: tuple[PyramidTransport, ...],
    ) -> tuple[Tensor, Tensor]:
        """Expected squared pixel step length summed over decoder layers."""
        per_layer = []
        for transport in layer_transports:
            step_l2 = PyramidFlowSystem._transport_step_squared_distance(transport)
            per_layer.append((transport.probs.float() * step_l2).sum(dim=1))
        total = torch.stack(
            [layer_l2.sum(dim=(1, 2)) for layer_l2 in per_layer],
            dim=0,
        ).sum(dim=0)
        num_sites = sum(layer_l2[0].numel() for layer_l2 in per_layer)
        return total.mean() / float(num_sites), total.mean()

    def _loss_terms(
        self, *, image: Tensor, output: PyramidFlowOutput
    ) -> dict[str, Tensor]:
        return {
            "recon": self._reconstruction_loss(output["recon"], image).float(),
            "foreground_kl": output["foreground_kl"].float(),
            "background_kl": output["background_kl"].float(),
            "flow_l2": output["flow_l2"].float(),
            "foreground_sparsity": output["foreground_sparsity"].float(),
        }

    @staticmethod
    def _rms(value: Tensor) -> Tensor:
        return value.float().square().mean().sqrt()

    def decoder_ablation_metrics(
        self, *, image: Tensor, output: PyramidFlowOutput
    ) -> dict[str, Tensor]:
        """Return reconstruction diagnostics for causal pathway ablations."""
        if tuple(image.shape) != tuple(output["recon"].shape):
            raise ValueError(
                "image shape must match output reconstruction shape, got "
                f"{tuple(image.shape)} and {tuple(output['recon'].shape)}"
            )
        batch_size = int(image.shape[0])
        if batch_size < 2:
            raise ValueError(
                "decoder ablation diagnostics require at least 2 samples for "
                f"batch shuffling, got {batch_size}"
            )
        with torch.no_grad():
            target = image.detach().float()
            recon = output["recon"].detach()
            foreground_presence = (
                output["foreground_presence"]
                .detach()
                .to(dtype=output["background_features"].dtype)
            )
            background_features = output["background_features"].detach()
            foreground_features = output["foreground_features"].detach()
            layer_transports = output["layer_transports"]
            zero_foreground_features = torch.zeros_like(foreground_features)
            zero_background_features = torch.zeros_like(background_features)
            zero_presence = torch.zeros_like(foreground_presence)
            full_presence = torch.ones_like(foreground_presence)
            shuffled_presence = torch.roll(foreground_presence, shifts=1, dims=0)
            shuffled_transports = self._cyclically_shuffle_layer_transports(
                layer_transports
            )

            zero_foreground_recomputed_transport_recon, _ = self.decoder(
                zero_foreground_features,
                background_features,
                foreground_presence,
            )
            zero_background_recomputed_transport_recon, _ = self.decoder(
                foreground_features,
                zero_background_features,
                foreground_presence,
            )
            shuffled_presence_recomputed_transport_recon, _ = self.decoder(
                foreground_features,
                background_features,
                shuffled_presence,
            )
            shuffled_transport_recon, _ = self.decoder(
                foreground_features,
                background_features,
                foreground_presence,
                layer_transports=shuffled_transports,
            )

            zero_foreground_fixed_transport_recon, _ = self.decoder(
                zero_foreground_features,
                background_features,
                foreground_presence,
                layer_transports=layer_transports,
            )
            zero_presence_fixed_transport_recon, _ = self.decoder(
                foreground_features,
                background_features,
                zero_presence,
                layer_transports=layer_transports,
            )
            foreground_values_full_presence_fixed_transport_recon, _ = self.decoder(
                foreground_features,
                background_features,
                full_presence,
                layer_transports=layer_transports,
            )
            zero_latents_presence_fixed_transport_recon, _ = self.decoder(
                zero_foreground_features,
                zero_background_features,
                foreground_presence,
                layer_transports=layer_transports,
            )
            zero_latents_zero_presence_fixed_transport_recon, _ = self.decoder(
                zero_foreground_features,
                zero_background_features,
                zero_presence,
                layer_transports=layer_transports,
            )
            uniform_transports = self._uniform_layer_transports_like(layer_transports)
            zero_latents_presence_uniform_transport_recon, _ = self.decoder(
                zero_foreground_features,
                zero_background_features,
                foreground_presence,
                layer_transports=uniform_transports,
            )
            zero_presence_uniform_transport_recon = self.decoder(
                foreground_features,
                background_features,
                zero_presence,
                layer_transports=uniform_transports,
            )[0]

            def mse(a: Tensor, b: Tensor) -> Tensor:
                return (a.float() - b.float()).square().mean()

            zero_latents_presence_spatial = (
                zero_latents_presence_fixed_transport_recon.float().flatten(start_dim=2)
            )
            zero_latents_zero_presence_spatial = (
                zero_latents_zero_presence_fixed_transport_recon.float().flatten(
                    start_dim=2
                )
            )
            return {
                "recon_mse": mse(recon, target),
                "zero_foreground_recomputed_transport_mse": mse(
                    zero_foreground_recomputed_transport_recon, target
                ),
                "zero_foreground_recomputed_transport_delta_mse": mse(
                    zero_foreground_recomputed_transport_recon, recon
                ),
                "zero_background_recomputed_transport_mse": mse(
                    zero_background_recomputed_transport_recon, target
                ),
                "zero_background_recomputed_transport_delta_mse": mse(
                    zero_background_recomputed_transport_recon, recon
                ),
                "shuffled_presence_recomputed_transport_mse": mse(
                    shuffled_presence_recomputed_transport_recon, target
                ),
                "shuffled_presence_recomputed_transport_delta_mse": mse(
                    shuffled_presence_recomputed_transport_recon, recon
                ),
                "shuffled_transport_mse": mse(shuffled_transport_recon, target),
                "shuffled_transport_delta_mse": mse(shuffled_transport_recon, recon),
                "zero_foreground_fixed_transport_mse": mse(
                    zero_foreground_fixed_transport_recon, target
                ),
                "zero_foreground_fixed_transport_delta_mse": mse(
                    zero_foreground_fixed_transport_recon, recon
                ),
                "zero_presence_fixed_transport_mse": mse(
                    zero_presence_fixed_transport_recon, target
                ),
                "zero_presence_fixed_transport_delta_mse": mse(
                    zero_presence_fixed_transport_recon, recon
                ),
                "foreground_values_full_presence_fixed_transport_mse": mse(
                    foreground_values_full_presence_fixed_transport_recon, target
                ),
                "foreground_values_full_presence_fixed_transport_delta_mse": mse(
                    foreground_values_full_presence_fixed_transport_recon, recon
                ),
                "zero_latents_presence_fixed_transport_mse": mse(
                    zero_latents_presence_fixed_transport_recon, target
                ),
                "zero_latents_zero_presence_fixed_transport_mse": mse(
                    zero_latents_zero_presence_fixed_transport_recon,
                    target,
                ),
                "zero_latents_presence_fixed_transport_spatial_std": (
                    zero_latents_presence_spatial.std(
                        dim=2,
                        unbiased=False,
                    ).mean()
                ),
                "zero_latents_zero_presence_fixed_transport_spatial_std": (
                    zero_latents_zero_presence_spatial.std(
                        dim=2,
                        unbiased=False,
                    ).mean()
                ),
                "zero_latents_fixed_vs_uniform_transport_delta_mse": mse(
                    zero_latents_presence_fixed_transport_recon,
                    zero_latents_presence_uniform_transport_recon,
                ),
                "zero_presence_fixed_vs_uniform_transport_delta_mse": mse(
                    zero_presence_fixed_transport_recon,
                    zero_presence_uniform_transport_recon,
                ),
            }

    @staticmethod
    def _cyclically_shuffle_layer_transports(
        layer_transports: tuple[PyramidTransport, ...],
    ) -> tuple[PyramidTransport, ...]:
        return tuple(
            transport._replace(
                probs=torch.roll(transport.probs, shifts=1, dims=0),
                logits=torch.roll(transport.logits, shifts=1, dims=0),
                expected_offset=torch.roll(transport.expected_offset, shifts=1, dims=0),
            )
            for transport in layer_transports
        )

    @staticmethod
    def _uniform_layer_transports_like(
        layer_transports: tuple[PyramidTransport, ...],
    ) -> tuple[PyramidTransport, ...]:
        uniform_transports = []
        for transport in layer_transports:
            fine_h, fine_w = transport.lookup.fine_hw
            valid = transport.lookup.valid_edges.T.reshape(1, 9, fine_h, fine_w).to(
                device=transport.probs.device
            )
            valid = valid.expand(int(transport.probs.shape[0]), -1, -1, -1)
            probs = valid.to(dtype=transport.probs.dtype)
            probs = probs / probs.sum(dim=1, keepdim=True)
            expected_offset = _expected_local_offset(probs, lookup=transport.lookup)
            uniform_transports.append(
                transport._replace(probs=probs, expected_offset=expected_offset)
            )
        return tuple(uniform_transports)

    def _weighted_loss(self, terms: dict[str, Tensor]) -> Tensor:
        return (
            self._scheduled_weight(self.recon_loss_weight) * terms["recon"]
            + self._scheduled_weight(self.foreground_kl_loss_weight)
            * terms["foreground_kl"]
            + self._scheduled_weight(self.background_kl_loss_weight)
            * terms["background_kl"]
            + self._scheduled_weight(self.flow_l2_loss_weight) * terms["flow_l2"]
            + self._scheduled_weight(self.foreground_sparsity_loss_weight)
            * terms["foreground_sparsity"]
        )

    def _scheduled_weight(self, schedule: LossWeightSchedule) -> float:
        if schedule.warmup_epochs <= 0:
            return schedule.end_weight
        progress = min(
            1.0, float(self.current_epoch + 1) / float(schedule.warmup_epochs)
        )
        return (
            schedule.start_weight
            + (schedule.end_weight - schedule.start_weight) * progress
        )

    @staticmethod
    def _extract_frame(batch: TemporalBatch) -> Tensor:
        video = batch["video"]
        if video.ndim != 5:
            raise ValueError(
                "batch['video'] must have shape (B, T, C, H, W), got "
                f"{tuple(video.shape)}"
            )
        if int(video.shape[1]) < 1:
            raise ValueError("batch['video'] must contain at least one frame")
        return video[:, -1]

    @staticmethod
    def _pad_image_to_input_divisibility(
        image: Tensor, *, divisibility: int, context: str
    ) -> tuple[Tensor, tuple[int, int], tuple[int, int]]:
        original_hw = (int(image.shape[-2]), int(image.shape[-1]))
        stride = int(divisibility)
        pad_h = (stride - original_hw[0] % stride) % stride
        pad_w = (stride - original_hw[1] % stride) % stride
        if pad_h == 0 and pad_w == 0:
            return image, original_hw, original_hw
        padded_hw = (original_hw[0] + pad_h, original_hw[1] + pad_w)
        warnings.warn(
            f"PyramidFlowSystem {context} input spatial shape is not divisible "
            f"by required input divisibility {stride}; zero-padding from "
            f"{original_hw} to {padded_hw}.",
            stacklevel=2,
        )
        return (
            F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0.0),
            original_hw,
            padded_hw,
        )

    def _pad_image_to_model_stride(
        self, image: Tensor
    ) -> tuple[Tensor, tuple[int, int], tuple[int, int]]:
        return self._pad_image_to_input_divisibility(
            image,
            divisibility=self._reconstruction_input_divisibility,
            context="reconstruction",
        )

    @staticmethod
    def _crop_pixel_outputs(
        output: PyramidFlowOutput,
        *,
        original_hw: tuple[int, int],
        padded_hw: tuple[int, int],
    ) -> PyramidFlowOutput:
        if original_hw == padded_hw:
            return output
        pixel_keys = {"recon", "p_fg", "expected_flow", "dense_assignment"}
        cropped: dict[str, Any] = dict(output)
        for key in pixel_keys:
            value = cropped.get(key)
            if isinstance(value, Tensor) and tuple(value.shape[-2:]) == padded_hw:
                cropped[key] = value[..., : original_hw[0], : original_hw[1]]
        return cast(PyramidFlowOutput, cropped)

    def _forward_with_model_padding(
        self, image: Tensor
    ) -> tuple[PyramidFlowOutput, tuple[int, int], tuple[int, int]]:
        padded_image, original_hw, padded_hw = self._pad_image_to_model_stride(image)
        output = self(padded_image)
        return (
            self._crop_pixel_outputs(
                output,
                original_hw=original_hw,
                padded_hw=padded_hw,
            ),
            original_hw,
            padded_hw,
        )

    def _segmentation_forward(
        self, image: Tensor
    ) -> PyramidFlowSegmentationInferenceOutput:
        output = self(image)
        return {
            "center_logits": output["foreground_logits"],
            "center_presence": output["foreground_presence"],
            "p_fg": output["p_fg"],
            "expected_flow": output["expected_flow"],
            "layer_transports": output["layer_transports"],
        }

    @staticmethod
    def _crop_segmentation_pixel_outputs(
        output: PyramidFlowSegmentationInferenceOutput,
        *,
        original_hw: tuple[int, int],
        padded_hw: tuple[int, int],
    ) -> PyramidFlowSegmentationInferenceOutput:
        if original_hw == padded_hw:
            return output
        cropped: dict[str, Any] = dict(output)
        for key in ("p_fg", "expected_flow"):
            value = cropped[key]
            if tuple(value.shape[-2:]) == padded_hw:
                cropped[key] = value[..., : original_hw[0], : original_hw[1]]
        return cast(PyramidFlowSegmentationInferenceOutput, cropped)

    def _forward_segmentation_with_model_padding(
        self, image: Tensor
    ) -> tuple[
        PyramidFlowSegmentationInferenceOutput,
        tuple[int, int],
        tuple[int, int],
    ]:
        padded_image, original_hw, padded_hw = self._pad_image_to_input_divisibility(
            image,
            divisibility=self._segmentation_input_divisibility,
            context="segmentation",
        )
        output = self._segmentation_forward(padded_image)
        return (
            self._crop_segmentation_pixel_outputs(
                output,
                original_hw=original_hw,
                padded_hw=padded_hw,
            ),
            original_hw,
            padded_hw,
        )

    def _shared_step(
        self, batch: TemporalBatch, *, stage: str, batch_idx: int
    ) -> Tensor:
        image = self._extract_frame(batch)
        output, _original_hw, _padded_hw = self._forward_with_model_padding(image)
        terms = self._loss_terms(image=image, output=output)
        loss = self._weighted_loss(terms)
        self._raise_if_nonfinite_loss(stage=stage, loss=loss, loss_terms=terms)
        self.log(
            f"loss/{stage}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=stage == "val",
        )
        for name, value in terms.items():
            self.log(f"loss/{stage}_{name}", value, on_step=False, on_epoch=True)
        self.log(
            f"stats/foreground_sparsity_weight_{stage}",
            torch.as_tensor(
                self._scheduled_weight(self.foreground_sparsity_loss_weight),
                device=loss.device,
            ),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"stats/foreground_kl_weight_{stage}",
            torch.as_tensor(
                self._scheduled_weight(self.foreground_kl_loss_weight),
                device=loss.device,
            ),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"stats/background_kl_weight_{stage}",
            torch.as_tensor(
                self._scheduled_weight(self.background_kl_loss_weight),
                device=loss.device,
            ),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"stats/flow_l2_weight_{stage}",
            torch.as_tensor(
                self._scheduled_weight(self.flow_l2_loss_weight),
                device=loss.device,
            ),
            on_step=False,
            on_epoch=True,
        )
        self._log_stats(output=output, stage=stage)
        self._log_decoder_ablation_metrics(
            image=image, output=output, stage=stage, batch_idx=batch_idx
        )
        self._log_images(image=image, output=output, stage=stage, batch_idx=batch_idx)
        return loss

    @staticmethod
    def _raise_if_nonfinite_loss(
        *, stage: str, loss: Tensor, loss_terms: dict[str, Tensor]
    ) -> None:
        nonfinite_terms = [
            name
            for name, value in loss_terms.items()
            if not torch.isfinite(value.detach()).all().item()
        ]
        if torch.isfinite(loss.detach()).all().item() and not nonfinite_terms:
            return
        stats = ", ".join(
            f"{name}={float(value.detach().float())}"
            for name, value in {"total": loss, **loss_terms}.items()
        )
        raise RuntimeError(
            f"Non-finite PyramidFlowSystem {stage} loss detected: {stats}"
        )

    def _log_stats(self, *, output: PyramidFlowOutput, stage: str) -> None:
        with torch.no_grad():
            offsets = torch.cat(
                [
                    transport.expected_offset.float().flatten(start_dim=2)
                    for transport in output["layer_transports"]
                ],
                dim=2,
            )
            flow = output["expected_flow"].float()
            foreground_latents = output["foreground_latents"].float()
            background_latents = output["background_latents"].float()
            foreground_latent_mu = output["foreground_latent_mu"].float()
            background_latent_mu = output["background_latent_mu"].float()
            foreground_latent_logvar = output["foreground_latent_logvar"].float()
            background_latent_logvar = output["background_latent_logvar"].float()
            foreground_presence = output["foreground_presence"].float()
            foreground_presence_flat = foreground_presence.flatten()
            foreground_features = output["foreground_features"].float()
            background_features = output["background_features"].float()
            background_feature_rms = self._rms(background_features)
            stats = {
                "foreground_mean": output["p_fg"].float().mean(),
                "foreground_presence_sum": foreground_presence.sum(
                    dim=(1, 2, 3)
                ).mean(),
                "foreground_presence_mean": foreground_presence.mean(),
                "foreground_presence_p90": foreground_presence_flat.quantile(0.90),
                "foreground_presence_p99": foreground_presence_flat.quantile(0.99),
                "foreground_presence_max": foreground_presence.amax(
                    dim=(1, 2, 3)
                ).mean(),
                "foreground_presence_active_frac_005": (foreground_presence > 0.05)
                .float()
                .mean(),
                "foreground_latent_rms": self._rms(foreground_latents),
                "background_latent_rms": self._rms(background_latents),
                "foreground_feature_rms": self._rms(foreground_features),
                "background_feature_rms": background_feature_rms,
                "foreground_latent_mu_norm_mean": foreground_latent_mu.square()
                .sum(dim=1)
                .sqrt()
                .mean(),
                "background_latent_mu_norm_mean": background_latent_mu.square()
                .sum(dim=1)
                .sqrt()
                .mean(),
                "foreground_latent_logvar_mean": foreground_latent_logvar.mean(),
                "background_latent_logvar_mean": background_latent_logvar.mean(),
                "foreground_kl_mean": output["foreground_kl"].float(),
                "background_kl_mean": output["background_kl"].float(),
                "foreground_kl_total_mean": output["foreground_kl_total"].float(),
                "background_kl_total_mean": output["background_kl_total"].float(),
                "flow_l2_mean": output["flow_l2"].float(),
                "flow_l2_total_mean": output["flow_l2_total"].float(),
                "transport_expected_offset_norm_mean": offsets.square()
                .sum(dim=1)
                .sqrt()
                .mean(),
                "expected_flow_norm_mean": flow.square().sum(dim=1).sqrt().mean(),
            }
        for name, value in stats.items():
            self.log(f"stats/{name}_{stage}", value, on_step=False, on_epoch=True)

    def _should_log_decoder_ablation(self, batch_idx: int) -> bool:
        return (
            batch_idx == 0
            and self.log_decoder_ablation_every_n_epochs > 0
            and self.current_epoch % self.log_decoder_ablation_every_n_epochs == 0
        )

    def _log_decoder_ablation_metrics(
        self,
        *,
        image: Tensor,
        output: PyramidFlowOutput,
        stage: str,
        batch_idx: int,
    ) -> None:
        if not self._should_log_decoder_ablation(batch_idx):
            return
        batch_size = min(
            int(image.shape[0]),
            self.log_decoder_ablation_max_batch_size,
        )
        metrics = self.decoder_ablation_metrics(
            image=image[:batch_size],
            output=self._slice_output_batch(output=output, batch_size=batch_size),
        )
        for name, value in metrics.items():
            self.log(
                f"diagnostics/{name}_{stage}",
                value,
                on_step=False,
                on_epoch=True,
            )

    @staticmethod
    def _slice_output_batch(
        *, output: PyramidFlowOutput, batch_size: int
    ) -> PyramidFlowOutput:
        full_batch_size = int(output["recon"].shape[0])
        if batch_size >= full_batch_size:
            return output

        def slice_tensor(value: Tensor) -> Tensor:
            if value.ndim > 0 and int(value.shape[0]) == full_batch_size:
                return value[:batch_size]
            return value

        def slice_transport(transport: PyramidTransport) -> PyramidTransport:
            return PyramidTransport(
                probs=slice_tensor(transport.probs),
                logits=slice_tensor(transport.logits),
                expected_offset=slice_tensor(transport.expected_offset),
                lookup=transport.lookup,
                pixel_stride=transport.pixel_stride,
            )

        sliced: dict[str, Any] = {}
        for key, value in output.items():
            if key == "layer_transports":
                continue
            if isinstance(value, Tensor):
                sliced[key] = slice_tensor(value)
            else:
                sliced[key] = value
        sliced["layer_transports"] = tuple(
            slice_transport(transport) for transport in output["layer_transports"]
        )
        return cast(PyramidFlowOutput, sliced)

    def _should_log_images(self, batch_idx: int) -> bool:
        return (
            batch_idx == 0
            and self.log_images_every_n_epochs > 0
            and self.current_epoch % self.log_images_every_n_epochs == 0
        )

    def _log_images(
        self,
        *,
        image: Tensor,
        output: PyramidFlowOutput,
        stage: str,
        batch_idx: int,
    ) -> None:
        if not self._should_log_images(batch_idx):
            return
        logger = getattr(self, "logger", None)
        experiment = getattr(logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_image"):
            return
        num_samples = min(self.log_image_samples, int(image.shape[0]))
        recon_error = (output["recon"] - image).mean(dim=1, keepdim=True)
        foreground_presence = F.interpolate(
            output["foreground_presence"].float(),
            size=tuple(image.shape[-2:]),
            mode="nearest",
        )
        display = torch.cat(
            [
                _as_rgb_display(image[:num_samples]),
                _as_rgb_display(output["recon"][:num_samples]),
                _as_berlin_zero_display(recon_error[:num_samples]),
                _as_magma_display(output["p_fg"][:num_samples].float().clamp(0.0, 1.0)),
                _as_magma_display(
                    foreground_presence[:num_samples].float().clamp(0.0, 1.0)
                ),
                self._flow_to_rgb(output["expected_flow"][:num_samples]),
                *self._layer_flow_display_rows(
                    layer_transports=output["layer_transports"],
                    output_hw=(int(image.shape[-2]), int(image.shape[-1])),
                    num_samples=num_samples,
                    dtype=image.dtype,
                ),
            ],
            dim=0,
        )
        grid = vutils.make_grid(display, nrow=num_samples)
        experiment.add_image(stage, grid, global_step=self.global_step)

    @staticmethod
    def _normalize_for_display(x: Tensor) -> Tensor:
        denom = x.float().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return (x.float() / denom).clamp(0.0, 1.0)

    def _layer_flow_display_rows(
        self,
        *,
        layer_transports: tuple[PyramidTransport, ...],
        output_hw: tuple[int, int],
        num_samples: int,
        dtype: torch.dtype,
    ) -> list[Tensor]:
        return [
            self._flow_to_rgb(
                flow[:num_samples],
                max_magnitude_px=self._layer_flow_axis_window_px(kernel),
            )
            for kernel, flow in zip(
                layer_transports,
                self._layer_flow_maps(
                    layer_transports=layer_transports,
                    output_hw=output_hw,
                    dtype=dtype,
                ),
                strict=True,
            )
        ]

    def _layer_flow_maps(
        self,
        *,
        layer_transports: tuple[PyramidTransport, ...],
        output_hw: tuple[int, int],
        dtype: torch.dtype,
    ) -> tuple[Tensor, ...]:
        layer_flows: list[Tensor] = []
        for kernel in layer_transports:
            fine_h, fine_w = kernel.lookup.fine_hw
            coarse_h, coarse_w = kernel.lookup.coarse_hw
            scale_h = fine_h // coarse_h
            scale_w = fine_w // coarse_w
            coarse_coords = self._pixel_center_coordinates_2d(
                grid_hw=kernel.lookup.coarse_hw,
                pixel_stride_y=int(kernel.pixel_stride) * scale_h,
                pixel_stride_x=int(kernel.pixel_stride) * scale_w,
                device=kernel.probs.device,
                dtype=dtype,
            )
            fine_coords = self._pixel_center_coordinates_2d(
                grid_hw=kernel.lookup.fine_hw,
                pixel_stride_y=int(kernel.pixel_stride),
                pixel_stride_x=int(kernel.pixel_stride),
                device=kernel.probs.device,
                dtype=dtype,
            )
            layer_flow = self._gather_coordinate_map(coarse_coords, transport=kernel)
            layer_flow = layer_flow - fine_coords
            if tuple(layer_flow.shape[-2:]) != output_hw:
                layer_flow = F.interpolate(layer_flow, size=output_hw, mode="nearest")
            layer_flows.append(layer_flow)
        return tuple(layer_flows)

    @staticmethod
    def _layer_flow_axis_window_px(kernel: PyramidTransport) -> float:
        fine_h, fine_w = kernel.lookup.fine_hw
        coarse_h, coarse_w = kernel.lookup.coarse_hw
        scale_h = fine_h // coarse_h
        scale_w = fine_w // coarse_w
        fine_stride = int(kernel.pixel_stride)
        max_y = fine_stride * (float(scale_h) + 0.5 * float(scale_h - 1))
        max_x = fine_stride * (float(scale_w) + 0.5 * float(scale_w - 1))
        return max(max_y, max_x)

    @staticmethod
    def _pixel_center_coordinates_2d(
        *,
        grid_hw: tuple[int, int],
        pixel_stride_y: int,
        pixel_stride_x: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        height, width = int(grid_hw[0]), int(grid_hw[1])
        stride_y = int(pixel_stride_y)
        stride_x = int(pixel_stride_x)
        y = torch.arange(height, device=device, dtype=dtype) * float(
            stride_y
        ) + 0.5 * float(stride_y - 1)
        x = torch.arange(width, device=device, dtype=dtype) * float(
            stride_x
        ) + 0.5 * float(stride_x - 1)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx), dim=0).unsqueeze(0)

    def _flow_to_rgb(
        self, flow: Tensor, *, max_magnitude_px: float | None = None
    ) -> Tensor:
        flow = flow.float()
        flow_y = flow[:, 0:1]
        flow_x = flow[:, 1:2]
        hue = torch.remainder(torch.atan2(flow_y, flow_x) / (2.0 * math.pi) + 1.0, 1.0)
        magnitude = torch.sqrt(flow_y.square() + flow_x.square())
        scale = float(self.decoder.axis_window_px)
        if max_magnitude_px is not None:
            scale = float(max_magnitude_px)
        value = (magnitude / max(scale, self.eps)).clamp(0.0, 1.0)
        saturation = torch.ones_like(value)
        return self._hsv_to_rgb_torch(torch.cat((hue, saturation, value), dim=1))

    @staticmethod
    def _hsv_to_rgb_torch(hsv: Tensor) -> Tensor:
        h = torch.remainder(hsv[:, 0:1], 1.0)
        s = hsv[:, 1:2].clamp(0.0, 1.0)
        v = hsv[:, 2:3].clamp(0.0, 1.0)
        scaled_h = h * 6.0
        i = torch.floor(scaled_h)
        f = scaled_h - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i_mod = torch.remainder(i.to(dtype=torch.long), 6)
        r = torch.where(
            (i_mod == 0) | (i_mod == 5),
            v,
            torch.where((i_mod == 1) | (i_mod == 4), q, p),
        )
        g = torch.where(
            (i_mod == 1) | (i_mod == 2),
            v,
            torch.where((i_mod == 0) | (i_mod == 3), t, p),
        )
        b = torch.where(
            (i_mod == 3) | (i_mod == 4),
            v,
            torch.where((i_mod == 2) | (i_mod == 5), t, p),
        )
        return torch.cat((r, g, b), dim=1)

    def training_step(
        self,
        batch: TemporalBatch,
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> Tensor:
        del dataloader_idx
        return self._shared_step(batch, stage="train", batch_idx=batch_idx)

    def validation_step(
        self,
        batch: TemporalBatch,
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> Tensor:
        del dataloader_idx
        return self._shared_step(batch, stage="val", batch_idx=batch_idx)

    def test_step(
        self,
        batch: TemporalBatch,
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> Tensor:
        del dataloader_idx
        return self._shared_step(batch, stage="test", batch_idx=batch_idx)

    def predict_step(
        self,
        batch: TemporalBatch,
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> SegmentationPredictionPayload:
        """Predict instance labels without reading ground-truth annotations."""
        del batch_idx, dataloader_idx
        sequence_ids = batch.get("sequence_ids")
        frame_numbers = batch.get("frame_numbers")
        filename_padding_width = batch.get("filename_padding_width")
        if (
            sequence_ids is None
            or frame_numbers is None
            or filename_padding_width is None
        ):
            raise ValueError(
                "prediction batches must include 'sequence_ids', 'frame_numbers', "
                "and 'filename_padding_width'"
            )
        image = self._extract_frame(batch)
        output, original_hw, _padded_hw = self._forward_segmentation_with_model_padding(
            image
        )
        config = self.segmentation_test_config or PyramidFlowSegmentationTestConfig()
        segmentation = self.flow_induced_instance_labels(
            center_presence=output["center_presence"],
            layer_transports=output["layer_transports"],
            center_threshold=config.center_threshold,
            pixel_mass_threshold=config.pixel_mass_threshold,
            component_chunk_size=config.component_chunk_size,
            min_object_area=config.min_object_area,
            max_object_area=config.max_object_area,
            output_hw=original_hw,
        )
        instance_labels = segmentation.pred_labels
        instance_labels = _scale_label_images(
            instance_labels,
            scale_factor=config.prediction_output_scale_factor,
        )
        return {
            "sequence_ids": sequence_ids,
            "frame_numbers": frame_numbers[:, -1],
            "filename_padding_width": filename_padding_width,
            "instance_labels": instance_labels,
        }


__all__ = [
    "ConvTransportScorer2d",
    "FineToCoarseLookup",
    "LossWeightSchedule",
    "PyramidFlowDecoder2d",
    "PyramidFlowEncoder2d",
    "PyramidFlowEncoderOutput",
    "PyramidFlowOutput",
    "PyramidFlowSegmentationEncoderOutput",
    "PyramidFlowSegmentationInferenceOutput",
    "PyramidFlowSegmentationOutput",
    "PyramidFlowSegmentationTestConfig",
    "PyramidFlowSystem",
    "PyramidTransport",
    "fine_to_coarse_lookup",
]
