"""Shared-memory frame storage for temporal video datasets."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import cast

import numpy as np
import tifffile
import torch
import zarr
from torch import Tensor
from torch.nn import functional as F
from zarr.core.metadata import ArrayV3Metadata

from routing_pyramids.data.frame_catalog import FrameSequence


@dataclass(frozen=True)
class SharedFrameBank:
    buffer: Tensor
    videos: tuple[Tensor, ...]


def _load_frame(path) -> Tensor:
    image = tifffile.imread(str(path))
    return torch.from_numpy(image.astype(np.float32))


def _read_frame_shape(path) -> tuple[int, int]:
    with tifffile.TiffFile(str(path)) as tif:
        shape = tif.pages[0].shape

    if len(shape) != 2:
        raise RuntimeError(f"Expected 2D grayscale frame, got shape={shape} for {path}")

    return int(shape[0]), int(shape[1])


def validate_scale_factor(scale_factor: float) -> float:
    """Validate an integer or reciprocal-integer spatial output/input ratio."""
    if not isinstance(scale_factor, Real) or isinstance(scale_factor, bool):
        raise TypeError(f"scale_factor must be a real number, got {scale_factor!r}")
    scale_factor = float(scale_factor)
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError(
            f"scale_factor must be finite and greater than 0, got {scale_factor}"
        )
    integer_ratio = scale_factor if scale_factor >= 1 else 1.0 / scale_factor
    rounded_ratio = round(integer_ratio)
    if not math.isclose(integer_ratio, rounded_ratio, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "scale_factor must be an integer or the reciprocal of an integer, "
            f"got {scale_factor}"
        )
    if scale_factor >= 1:
        return float(rounded_ratio)
    return 1.0 / float(rounded_ratio)


def _scaled_hw(
    height: int,
    width: int,
    *,
    scale_factor: float,
    owner_name: str,
    sequence_id: str,
) -> tuple[int, int]:
    scale_factor = validate_scale_factor(scale_factor)
    if scale_factor >= 1:
        integer_factor = round(scale_factor)
        return height * integer_factor, width * integer_factor

    reduction_factor = round(1.0 / scale_factor)
    if height % reduction_factor != 0 or width % reduction_factor != 0:
        raise ValueError(
            f"{owner_name}: frame size {(height, width)} for sequence "
            f"{sequence_id} must be divisible by reduction factor "
            f"{reduction_factor} for scale_factor={scale_factor}"
        )
    return height // reduction_factor, width // reduction_factor


def scale_video(video: Tensor, *, scale_factor: float) -> Tensor:
    """Scale a ``(T, H, W)`` or ``(T, C, H, W)`` video before normalization."""
    scale_factor = validate_scale_factor(scale_factor)
    if scale_factor == 1:
        return video
    if video.ndim not in (3, 4):
        raise ValueError(
            f"video must have shape (T, H, W) or (T, C, H, W), got {tuple(video.shape)}"
        )
    had_channel_dimension = video.ndim == 4
    video_4d = video if had_channel_dimension else video.unsqueeze(1)
    if scale_factor < 1:
        reduction_factor = round(1.0 / scale_factor)
        scaled = F.avg_pool2d(
            video_4d,
            kernel_size=reduction_factor,
            stride=reduction_factor,
        )
    else:
        scaled = F.interpolate(
            video_4d,
            scale_factor=round(scale_factor),
            mode="bilinear",
            align_corners=False,
        )
    return scaled if had_channel_dimension else scaled.squeeze(1)


def _normalize_channels_independently(
    video: Tensor,
    normalization: Callable | None,
) -> Tensor:
    if normalization is None:
        return video
    normalized_channels = [
        normalization(video[:, channel])[0] for channel in range(int(video.shape[1]))
    ]
    return torch.stack(normalized_channels, dim=1)


def build_shared_cyx_frame_bank(
    sequences: tuple[FrameSequence, ...],
    normalization: Callable | None,
    owner_name: str,
    *,
    channels: int,
    scale_factor: float = 1.0,
) -> SharedFrameBank:
    """Load static CYX TIFF images into shared ``(T=1, C, H, W)`` storage."""
    if not sequences:
        raise RuntimeError(f"No frame sequences found for {owner_name}")
    scale_factor = validate_scale_factor(scale_factor)

    scan_results: list[tuple[FrameSequence, tuple[int, int]]] = []
    total_elements = 0
    for sequence in sequences:
        if len(sequence.frame_files) != 1:
            raise ValueError(
                f"{owner_name}: expected exactly one CYX file for sequence "
                f"{sequence.sequence_id}, got {len(sequence.frame_files)}"
            )
        image_file = sequence.frame_files[0]
        with tifffile.TiffFile(str(image_file)) as tif:
            series = tif.series[0]
            axes = series.axes
            shape = series.shape
        if axes != "CYX" or len(shape) != 3:
            raise RuntimeError(
                f"Expected CYX TIFF image at {image_file}, got "
                f"axes={axes!r}, shape={shape}."
            )
        if int(shape[0]) != channels:
            raise RuntimeError(
                f"Expected {channels} channels at {image_file}, got {int(shape[0])}."
            )
        height, width = _scaled_hw(
            int(shape[1]),
            int(shape[2]),
            scale_factor=scale_factor,
            owner_name=owner_name,
            sequence_id=sequence.sequence_id,
        )
        total_elements += channels * height * width
        scan_results.append((sequence, (height, width)))

    try:
        shared_buffer = torch.zeros(total_elements, dtype=torch.float32)
        shared_buffer.share_memory_()
    except Exception as exc:
        estimated_gib = (total_elements * 4) / (1024**3)
        raise RuntimeError(
            f"{owner_name}: failed to allocate shared frame bank with "
            f"{total_elements} float32 elements (~{estimated_gib:.2f} GiB)."
        ) from exc

    videos: list[Tensor] = []
    current_offset = 0
    for sequence, (height, width) in scan_results:
        image = tifffile.imread(str(sequence.frame_files[0]))
        video = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        video = scale_video(video, scale_factor=scale_factor)
        video = _normalize_channels_independently(video, normalization)
        flattened = video.reshape(-1)
        num_elements = flattened.numel()
        shared_buffer[current_offset : current_offset + num_elements] = flattened
        video_view = shared_buffer[current_offset : current_offset + num_elements].view(
            1, channels, height, width
        )
        videos.append(video_view)
        current_offset += num_elements

    return SharedFrameBank(buffer=shared_buffer, videos=tuple(videos))


def build_shared_frame_bank(
    sequences: tuple[FrameSequence, ...],
    normalization: Callable | None,
    owner_name: str,
    scale_factor: float = 1.0,
) -> SharedFrameBank:
    """Load frame sequences into one contiguous shared-memory tensor."""
    if not sequences:
        raise RuntimeError(f"No frame sequences found for {owner_name}")
    scale_factor = validate_scale_factor(scale_factor)

    scan_results: list[tuple[FrameSequence, tuple[int, int]]] = []
    total_elements = 0

    for sequence in sequences:
        if not sequence.frame_files:
            continue

        try:
            height, width = _read_frame_shape(sequence.frame_files[0])
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read metadata for sequence {sequence.sequence_id}"
            ) from exc

        height_scaled, width_scaled = _scaled_hw(
            height,
            width,
            scale_factor=scale_factor,
            owner_name=owner_name,
            sequence_id=sequence.sequence_id,
        )

        total_elements += len(sequence.frame_files) * height_scaled * width_scaled
        scan_results.append((sequence, (height_scaled, width_scaled)))

    if not scan_results:
        raise RuntimeError(f"No frame sequences with data found for {owner_name}")

    if total_elements <= 0:
        raise RuntimeError(f"No frame elements found for {owner_name}")

    try:
        shared_buffer = torch.zeros(total_elements, dtype=torch.float32)
        shared_buffer.share_memory_()
    except Exception as exc:
        estimated_gib = (total_elements * 4) / (1024**3)
        raise RuntimeError(
            f"{owner_name}: failed to allocate shared frame bank with "
            f"{total_elements} float32 elements (~{estimated_gib:.2f} GiB)."
        ) from exc

    videos: list[Tensor] = []
    current_offset = 0

    for sequence, (height, width) in scan_results:
        frames = [_load_frame(path) for path in sequence.frame_files]
        video = torch.stack(frames)  # (T, H, W)
        video = scale_video(video, scale_factor=scale_factor)

        if normalization is not None:
            video, _ = normalization(video)

        flattened = video.reshape(-1)
        num_elements = flattened.numel()

        shared_buffer[current_offset : current_offset + num_elements] = flattened
        video_view = shared_buffer[current_offset : current_offset + num_elements].view(
            len(sequence.frame_files), height, width
        )

        videos.append(video_view)
        current_offset += num_elements

    return SharedFrameBank(buffer=shared_buffer, videos=tuple(videos))


def _ome_zarr_array(store_dir: Path) -> zarr.Array:
    group = zarr.open_group(str(store_dir), mode="r")
    ome = group.attrs.get("ome")
    if not isinstance(ome, dict):
        raise TypeError(f"Missing OME metadata in {store_dir}")
    multiscales = ome.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise TypeError(f"Missing OME multiscales metadata in {store_dir}")
    multiscale = multiscales[0]
    if not isinstance(multiscale, dict):
        raise TypeError(f"Invalid OME multiscales metadata in {store_dir}")

    axes = multiscale.get("axes")
    if not isinstance(axes, list):
        raise TypeError(f"Missing OME axes metadata in {store_dir}")
    axis_names = tuple(
        axis.get("name") if isinstance(axis, dict) else None for axis in axes
    )
    if axis_names != ("t", "c", "y", "x"):
        raise ValueError(
            f"Expected OME axes ('t', 'c', 'y', 'x') in {store_dir}, got {axis_names}"
        )

    datasets = multiscale.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise TypeError(f"Missing OME multiscale datasets in {store_dir}")
    dataset = datasets[0]
    if not isinstance(dataset, dict) or not isinstance(dataset.get("path"), str):
        raise TypeError(f"Invalid OME multiscale dataset in {store_dir}")
    array = group[cast(str, dataset["path"])]
    if not isinstance(array, zarr.Array):
        raise TypeError(
            f"OME multiscale path {dataset['path']!r} in {store_dir} is not an array"
        )
    if array.ndim != 4:
        raise ValueError(
            f"Expected a 4D TCYX array in {store_dir}, got shape={array.shape}"
        )
    if not isinstance(array.metadata, ArrayV3Metadata):
        raise TypeError(f"Expected a Zarr v3 array in {store_dir}")
    if array.metadata.dimension_names != ("t", "c", "y", "x"):
        raise ValueError(
            f"Expected Zarr dimension names ('t', 'c', 'y', 'x') in {store_dir}, "
            f"got {array.metadata.dimension_names}"
        )
    if int(array.shape[1]) != 1:
        raise ValueError(
            f"Expected one image channel in {store_dir}, got {array.shape[1]}"
        )
    return array


def build_shared_ome_zarr_frame_bank(
    sequences: tuple[FrameSequence, ...],
    normalization: Callable | None,
    owner_name: str,
    scale_factor: float = 1.0,
) -> SharedFrameBank:
    """Load single-channel TCYX OME-Zarr stores into shared temporal storage."""
    if not sequences:
        raise RuntimeError(f"No OME-Zarr stores found for {owner_name}")
    scale_factor = validate_scale_factor(scale_factor)

    scan_results: list[tuple[FrameSequence, zarr.Array, tuple[int, int, int]]] = []
    total_elements = 0
    for sequence in sequences:
        if len(sequence.frame_files) != 1:
            raise ValueError(
                f"{owner_name}: expected one OME-Zarr store for sequence "
                f"{sequence.sequence_id}, got {len(sequence.frame_files)}"
            )
        array = _ome_zarr_array(sequence.frame_files[0])
        timesteps, _, height, width = (int(size) for size in array.shape)
        height_scaled, width_scaled = _scaled_hw(
            height,
            width,
            scale_factor=scale_factor,
            owner_name=owner_name,
            sequence_id=sequence.sequence_id,
        )
        total_elements += timesteps * height_scaled * width_scaled
        scan_results.append((sequence, array, (timesteps, height_scaled, width_scaled)))

    shared_buffer = torch.zeros(total_elements, dtype=torch.float32)
    shared_buffer.share_memory_()
    videos: list[Tensor] = []
    current_offset = 0
    for sequence, array, (timesteps, height, width) in scan_results:
        video = torch.from_numpy(np.asarray(array[:, 0], dtype=np.float32))
        video = scale_video(video, scale_factor=scale_factor)
        if normalization is not None:
            video, _ = normalization(video)

        flattened = video.reshape(-1)
        num_elements = flattened.numel()
        expected_elements = timesteps * height * width
        if num_elements != expected_elements:
            raise RuntimeError(
                f"{owner_name}: loaded sequence {sequence.sequence_id} has "
                f"{num_elements} elements, expected {expected_elements}."
            )
        shared_buffer[current_offset : current_offset + num_elements] = flattened
        videos.append(
            shared_buffer[current_offset : current_offset + num_elements].view(
                timesteps, height, width
            )
        )
        current_offset += num_elements

    return SharedFrameBank(buffer=shared_buffer, videos=tuple(videos))
