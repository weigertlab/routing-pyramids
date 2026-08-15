"""Video-only temporal datasets and collation."""

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from routing_pyramids.data.frame_catalog import (
    FrameSequence,
    discover_bbbc013_images,
    discover_ctc_split,
    discover_ome_zarr_stores,
)
from routing_pyramids.data.shared_frame_bank import (
    build_shared_cyx_frame_bank,
    build_shared_frame_bank,
    build_shared_ome_zarr_frame_bank,
)
from routing_pyramids.data.temporal_index import (
    TemporalFrameSelector,
    TemporalWindowIndex,
    build_temporal_window_index,
    resolve_temporal_source_length,
)
from routing_pyramids.types import TemporalBatch, TemporalSample


def _parse_frame_number_and_padding(path: Path) -> tuple[int, int]:
    suffix = path.stem.removeprefix("t")
    if suffix == path.stem or not suffix.isdigit():
        raise ValueError(
            f"Unexpected frame filename {path.name}. Expected pattern 't<digits>.tif'."
        )
    return int(suffix), len(suffix)


def _frame_numbers_and_padding(
    sequences: tuple[FrameSequence, ...],
) -> tuple[tuple[Tensor, ...], tuple[int, ...]]:
    frame_numbers: list[Tensor] = []
    padding_widths: list[int] = []
    for sequence in sequences:
        parsed = tuple(
            _parse_frame_number_and_padding(path) for path in sequence.frame_files
        )
        if not parsed:
            frame_numbers.append(torch.empty(0, dtype=torch.long))
            padding_widths.append(3)
            continue
        numbers, widths = zip(*parsed)
        if len(set(widths)) != 1:
            raise ValueError(
                "Frame filename padding width must be constant within sequence "
                f"{sequence.sequence_id}, got {sorted(set(widths))}"
            )
        frame_numbers.append(torch.tensor(numbers, dtype=torch.long))
        padding_widths.append(int(widths[0]))
    return tuple(frame_numbers), tuple(padding_widths)


def _sequential_frame_numbers_and_padding(
    videos: tuple[Tensor, ...],
    *,
    padding_width: int = 3,
) -> tuple[tuple[Tensor, ...], tuple[int, ...]]:
    frame_numbers = tuple(
        torch.arange(int(video.shape[0]), dtype=torch.long) for video in videos
    )
    return frame_numbers, tuple(padding_width for _ in videos)


class TemporalVideoDataset(Dataset):
    """Base dataset for temporal sampling over cached video tensors."""

    def __init__(
        self,
        *,
        videos: tuple[Tensor, ...],
        sequence_ids: tuple[str, ...],
        frame_numbers: tuple[Tensor, ...],
        filename_padding_widths: tuple[int, ...],
        sequence_length: int,
        temporal_source_length: int | None,
        temporal_frame_stride: int = 1,
        augmentations: Callable | None = None,
    ):
        if not videos:
            raise RuntimeError("No videos loaded")
        if len(sequence_ids) != len(videos):
            raise ValueError(
                "sequence_ids must match number of videos, got "
                f"{len(sequence_ids)} and {len(videos)}"
            )
        if len(frame_numbers) != len(videos):
            raise ValueError(
                "frame_numbers must match number of videos, got "
                f"{len(frame_numbers)} and {len(videos)}"
            )
        if len(filename_padding_widths) != len(videos):
            raise ValueError(
                "filename_padding_widths must match number of videos, got "
                f"{len(filename_padding_widths)} and {len(videos)}"
            )
        for video, numbers in zip(videos, frame_numbers):
            if int(video.shape[0]) != int(numbers.shape[0]):
                raise ValueError(
                    "frame_numbers length must match video length, got "
                    f"{int(numbers.shape[0])} and {int(video.shape[0])}"
                )
        self.videos = videos
        self.sequence_ids = sequence_ids
        self.frame_numbers = frame_numbers
        self.filename_padding_widths = filename_padding_widths
        self.sequence_length = int(sequence_length)
        self.temporal_source_length = resolve_temporal_source_length(
            sequence_length=self.sequence_length,
            temporal_source_length=temporal_source_length,
            temporal_frame_stride=temporal_frame_stride,
        )
        self.temporal_frame_stride = int(temporal_frame_stride)
        self.augmentations = augmentations
        self.window_index: TemporalWindowIndex = build_temporal_window_index(
            videos=self.videos,
            source_length=self.temporal_source_length,
        )
        self.frame_selector = TemporalFrameSelector(
            sequence_length=self.sequence_length,
            source_length=self.temporal_source_length,
            temporal_frame_stride=self.temporal_frame_stride,
        )

    @property
    def fixed_frame_indices(self) -> tuple[Tensor, ...] | None:
        return self.frame_selector.fixed_frame_indices

    def set_fixed_frame_indices(self, indices: Sequence[Tensor] | None) -> None:
        self.frame_selector.set_fixed_frame_indices(
            indices=indices,
            dataset_length=len(self),
        )

    def __len__(self) -> int:
        return len(self.window_index.samples)

    def __getitem__(self, index: int) -> TemporalSample:
        video_index, start = self.window_index.samples[index]
        source_chunk = self.videos[video_index][
            start : start + self.temporal_source_length
        ]  # (T_src, H, W)
        frame_indices = self.frame_selector.frame_indices_for_sample(index)
        chunk = source_chunk[frame_indices]  # (T, H, W)
        source_frame_indices = frame_indices.to(dtype=torch.long) + int(start)
        if self.augmentations is not None:
            chunk, _ = self.augmentations(chunk)
        if chunk.ndim == 3:
            chunk = chunk.unsqueeze(1)
        elif chunk.ndim != 4:
            raise RuntimeError(
                "Sampled video must have shape (T, H, W) or (T, C, H, W), "
                f"got {tuple(chunk.shape)}"
            )
        sample: TemporalSample = {
            "video": chunk,
            "frame_indices": frame_indices.to(dtype=torch.long).clone(),
            "source_length": self.temporal_source_length,
            "sequence_id": self.sequence_ids[video_index],
            "frame_numbers": self.frame_numbers[video_index][
                source_frame_indices
            ].clone(),
            "filename_padding_width": self.filename_padding_widths[video_index],
        }
        return sample


class CTCVideoDataset(TemporalVideoDataset):
    """Temporal dataset for Cell Tracking Challenge directory layout.

    Parameters
    ----------
    root_dir
        Root directory containing split and ground-truth directories.
    split
        Dataset split to load.
    normalization
        Optional transform applied to each complete scaled video.
    augmentations
        Optional spatial transform applied jointly to video and loaded masks.
    sequence_length
        Number of frames returned by each sample.
    temporal_source_length
        Number of source frames from which output frames are selected.
    temporal_frame_stride
        Stride between selected source frames.
    scale_factor
        Spatial output/input ratio applied before normalization.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        normalization: Callable | None = None,
        augmentations: Callable | None = None,
        sequence_length: int = 8,
        temporal_source_length: int | None = None,
        temporal_frame_stride: int = 1,
        scale_factor: float = 1.0,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        sequences = discover_ctc_split(self.root_dir, split)
        frame_bank = build_shared_frame_bank(
            sequences=sequences,
            normalization=normalization,
            owner_name=f"CTCVideoDataset[{split}]",
            scale_factor=scale_factor,
        )
        self.shared_buffer = frame_bank.buffer
        frame_numbers, filename_padding_widths = _frame_numbers_and_padding(sequences)
        super().__init__(
            videos=frame_bank.videos,
            sequence_ids=tuple(sequence.sequence_id for sequence in sequences),
            frame_numbers=frame_numbers,
            filename_padding_widths=filename_padding_widths,
            sequence_length=sequence_length,
            temporal_source_length=temporal_source_length,
            temporal_frame_stride=temporal_frame_stride,
            augmentations=augmentations,
        )


class BBBC013VideoDataset(TemporalVideoDataset):
    """Single-frame image dataset for selected BBBC013 plate wells.

    Parameters
    ----------
    root_dir
        Directory containing converted TIFF files named like ``A1.tif``.
    first_characters
        Filename-first-character values selecting plate rows to include.
    columns
        Optional plate-column indices to include. By default, include all columns.
    normalization
        Optional transform applied independently to each image.
    augmentations
        Optional spatial transform applied to each image sample.
    scale_factor
        Spatial output/input scale factor applied before normalization.
    """

    def __init__(
        self,
        root_dir: str,
        first_characters: Sequence[str],
        columns: Sequence[int] | None = None,
        normalization: Callable | None = None,
        augmentations: Callable | None = None,
        scale_factor: float = 1.0,
    ):
        self.root_dir = Path(root_dir)
        self.first_characters = tuple(first_characters)
        self.columns = None if columns is None else tuple(columns)
        sequences = discover_bbbc013_images(
            self.root_dir,
            self.first_characters,
            self.columns,
        )
        column_label = (
            "all" if self.columns is None else ",".join(map(str, self.columns))
        )
        frame_bank = build_shared_cyx_frame_bank(
            sequences=sequences,
            normalization=normalization,
            owner_name=(
                f"BBBC013VideoDataset[{''.join(self.first_characters)};"
                f"columns={column_label}]"
            ),
            channels=2,
            scale_factor=scale_factor,
        )
        self.shared_buffer = frame_bank.buffer
        frame_numbers, filename_padding_widths = _sequential_frame_numbers_and_padding(
            frame_bank.videos
        )

        super().__init__(
            videos=frame_bank.videos,
            sequence_ids=tuple(sequence.sequence_id for sequence in sequences),
            frame_numbers=frame_numbers,
            filename_padding_widths=filename_padding_widths,
            sequence_length=1,
            temporal_source_length=None,
            temporal_frame_stride=1,
            augmentations=augmentations,
        )


class OMEZarrVideoDataset(TemporalVideoDataset):
    """
    Temporal dataset for explicitly selected single-channel OME-Zarr stores.

    Parameters
    ----------
    root_dir
        Directory containing the OME-Zarr stores.
    store_names
        Names of the ``.ome.zarr`` directories to include as video sequences.
    normalization
        Optional transform applied independently to each complete video.
    augmentations
        Optional spatial transform applied to each sampled temporal window.
    sequence_length
        Number of frames returned by each sample.
    temporal_source_length
        Number of source frames from which output frames are selected.
    temporal_frame_stride
        Stride between selected source frames.
    scale_factor
        Spatial output/input scale factor applied before normalization.
    """

    def __init__(
        self,
        root_dir: str,
        store_names: Sequence[str],
        normalization: Callable | None = None,
        augmentations: Callable | None = None,
        sequence_length: int = 8,
        temporal_source_length: int | None = None,
        temporal_frame_stride: int = 1,
        scale_factor: float = 1.0,
    ):
        self.root_dir = Path(root_dir)
        self.store_names = tuple(store_names)
        sequences = discover_ome_zarr_stores(self.root_dir, self.store_names)
        frame_bank = build_shared_ome_zarr_frame_bank(
            sequences=sequences,
            normalization=normalization,
            owner_name="OMEZarrVideoDataset",
            scale_factor=scale_factor,
        )
        self.shared_buffer = frame_bank.buffer
        frame_numbers, filename_padding_widths = _sequential_frame_numbers_and_padding(
            frame_bank.videos,
        )

        super().__init__(
            videos=frame_bank.videos,
            sequence_ids=tuple(sequence.sequence_id for sequence in sequences),
            frame_numbers=frame_numbers,
            filename_padding_widths=filename_padding_widths,
            sequence_length=sequence_length,
            temporal_source_length=temporal_source_length,
            temporal_frame_stride=temporal_frame_stride,
            augmentations=augmentations,
        )


def collate_temporal_video_batch(batch: list[TemporalSample]) -> TemporalBatch:
    videos = torch.stack([sample["video"] for sample in batch])
    frame_indices = torch.stack([sample["frame_indices"] for sample in batch])
    source_length = int(batch[0]["source_length"])
    out: TemporalBatch = {
        "video": videos,
        "frame_indices": frame_indices,
        "source_length": source_length,
        "sequence_ids": [str(sample["sequence_id"]) for sample in batch],
        "frame_numbers": torch.stack([sample["frame_numbers"] for sample in batch]),
        "filename_padding_width": torch.tensor(
            [int(sample["filename_padding_width"]) for sample in batch],
            dtype=torch.long,
        ),
    }
    return out
