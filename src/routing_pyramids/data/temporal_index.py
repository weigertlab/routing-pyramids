"""Temporal indexing and frame selection for sequence datasets."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from routing_pyramids.data.sampling import (
    temporal_subsample_indices,
    validate_frame_indices,
)


@dataclass(frozen=True)
class TemporalWindowIndex:
    samples: tuple[tuple[int, int], ...]
    source_length: int


def resolve_temporal_source_length(
    *,
    sequence_length: int,
    temporal_source_length: int | None,
    temporal_frame_stride: int = 1,
) -> int:
    sequence_length = int(sequence_length)
    temporal_frame_stride = int(temporal_frame_stride)
    if temporal_frame_stride < 1:
        raise ValueError(
            f"temporal_frame_stride must be >= 1, got {temporal_frame_stride}"
        )

    required_source_length = (sequence_length - 1) * temporal_frame_stride + 1
    source_length = (
        required_source_length
        if temporal_source_length is None
        else int(temporal_source_length)
    )

    if source_length < required_source_length:
        raise ValueError(
            "temporal_source_length must cover sequence_length at "
            "temporal_frame_stride, got "
            f"temporal_source_length={source_length}, "
            f"sequence_length={sequence_length}, "
            f"temporal_frame_stride={temporal_frame_stride}"
        )
    if source_length > sequence_length and sequence_length < 2:
        raise ValueError(
            "sequence_length must be at least 2 when temporal subsampling is enabled, "
            f"got sequence_length={sequence_length}"
        )

    return source_length


def build_temporal_window_index(
    videos: tuple[Tensor, ...],
    source_length: int,
) -> TemporalWindowIndex:
    samples: list[tuple[int, int]] = []

    for video_index, video in enumerate(videos):
        num_frames = int(video.shape[0])
        for start in range(num_frames - source_length + 1):
            samples.append((video_index, start))

    if not samples:
        raise RuntimeError(
            f"No valid temporal windows found. Check source_length={source_length}."
        )

    return TemporalWindowIndex(samples=tuple(samples), source_length=source_length)


def _sample_temporal_indices(*, source_length: int, keep_length: int) -> Tensor:
    frame_indices = temporal_subsample_indices(
        source_length=source_length,
        keep_length=keep_length,
    )
    return validate_frame_indices(
        frame_indices,
        timesteps=keep_length,
        source_length=source_length,
    ).squeeze(0)


def _normalize_fixed_indices(
    *,
    indices: Sequence[Tensor] | None,
    dataset_length: int,
    source_length: int,
    keep_length: int,
) -> tuple[Tensor, ...] | None:
    if indices is None:
        return None

    if len(indices) != dataset_length:
        raise ValueError(
            "fixed frame index list must match dataset length, got "
            f"{len(indices)} and {dataset_length}"
        )

    normalized: list[Tensor] = []
    for frame_indices in indices:
        normalized_frame_indices = validate_frame_indices(
            frame_indices,
            timesteps=keep_length,
            source_length=source_length,
        ).squeeze(0)
        normalized.append(normalized_frame_indices.clone())

    return tuple(normalized)


def build_deterministic_frame_indices(
    *,
    num_samples: int,
    source_length: int,
    keep_length: int,
    seed: int,
) -> tuple[Tensor, ...]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    frame_indices_list: list[Tensor] = []
    for _ in range(num_samples):
        sampled = temporal_subsample_indices(
            source_length=source_length,
            keep_length=keep_length,
            generator=generator,
        )
        validated = validate_frame_indices(
            sampled,
            timesteps=keep_length,
            source_length=source_length,
        ).squeeze(0)
        frame_indices_list.append(validated)

    return tuple(frame_indices_list)


class TemporalFrameSelector:
    """Resolve sampled frame indices for each temporal window sample."""

    def __init__(
        self,
        sequence_length: int,
        source_length: int,
        temporal_frame_stride: int = 1,
    ):
        self.sequence_length = int(sequence_length)
        self.source_length = int(source_length)
        self.temporal_frame_stride = int(temporal_frame_stride)
        if self.temporal_frame_stride < 1:
            raise ValueError(
                f"temporal_frame_stride must be >= 1, got {self.temporal_frame_stride}"
            )
        self._fixed_frame_indices: tuple[Tensor, ...] | None = None

        last_frame_index = (self.sequence_length - 1) * self.temporal_frame_stride
        if last_frame_index >= self.source_length:
            raise ValueError(
                "temporal_frame_stride requires source_length to cover the sampled "
                "frames, got "
                f"sequence_length={self.sequence_length}, "
                f"source_length={self.source_length}, "
                f"temporal_frame_stride={self.temporal_frame_stride}"
            )

    @property
    def fixed_frame_indices(self) -> tuple[Tensor, ...] | None:
        return self._fixed_frame_indices

    def set_fixed_frame_indices(
        self,
        *,
        indices: Sequence[Tensor] | None,
        dataset_length: int,
    ) -> None:
        self._fixed_frame_indices = _normalize_fixed_indices(
            indices=indices,
            dataset_length=dataset_length,
            source_length=self.source_length,
            keep_length=self.sequence_length,
        )

    def sample_frame_indices(self) -> Tensor:
        if self.temporal_frame_stride > 1:
            return (
                torch.arange(
                    self.sequence_length,
                    dtype=torch.long,
                )
                * self.temporal_frame_stride
            )
        return _sample_temporal_indices(
            source_length=self.source_length,
            keep_length=self.sequence_length,
        )

    def frame_indices_for_sample(self, sample_index: int) -> Tensor:
        if self._fixed_frame_indices is not None:
            return self._fixed_frame_indices[sample_index]
        return self.sample_frame_indices()
