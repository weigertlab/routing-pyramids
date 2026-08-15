"""Shared TypedDict contracts for temporal batches and samples."""

from typing import NotRequired, TypedDict

from torch import Tensor


class TemporalSample(TypedDict):
    """One image-only temporal sample."""

    video: Tensor
    frame_indices: Tensor
    source_length: int | Tensor
    sequence_id: NotRequired[str]
    frame_numbers: NotRequired[Tensor]
    filename_padding_width: NotRequired[int | Tensor]


class TemporalBatch(TypedDict):
    video: Tensor
    frame_indices: Tensor
    source_length: int | Tensor
    sequence_ids: NotRequired[list[str]]
    frame_numbers: NotRequired[Tensor]
    filename_padding_width: NotRequired[Tensor]


class SegmentationPredictionPayload(TypedDict):
    """Ground-truth-free instance-segmentation prediction for one frame batch."""

    sequence_ids: list[str]
    frame_numbers: Tensor
    filename_padding_width: Tensor
    instance_labels: Tensor


LossStats = dict[str, Tensor]
