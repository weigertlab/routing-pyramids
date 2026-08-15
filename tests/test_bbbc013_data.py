"""Tests for two-channel BBBC013 image loading."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from routing_pyramids.data.frame_catalog import discover_bbbc013_images
from routing_pyramids.data.temporal_datamodule import BBBC013VideoDataModule
from routing_pyramids.data.temporal_video_dataset import BBBC013VideoDataset


def _write_well(path: Path, *, offset: int) -> None:
    grid = np.arange(12 * 16, dtype=np.uint16).reshape(12, 16)
    image = np.stack((grid + offset, 2 * grid + offset + 1000))
    tifffile.imwrite(
        path,
        image,
        imagej=True,
        metadata={"axes": "CYX"},
    )


def _write_plate(root: Path, *, columns: tuple[int, ...] = (1, 2)) -> None:
    for row_index, row in enumerate("ABCDEFGH"):
        for column in columns:
            _write_well(
                root / f"{row}{column}.tif",
                offset=100 * row_index + column,
            )


def test_bbbc013_discovery_selects_rows_and_columns_and_sorts_numerically(
    tmp_path: Path,
) -> None:
    _write_plate(tmp_path, columns=(10, 2, 1))

    sequences = discover_bbbc013_images(tmp_path, ("B", "A"), columns=(10, 1))

    assert tuple(sequence.sequence_id for sequence in sequences) == (
        "B1",
        "B10",
        "A1",
        "A10",
    )


def test_bbbc013_dataset_loads_both_channels_as_one_frame(tmp_path: Path) -> None:
    _write_well(tmp_path / "A1.tif", offset=7)

    dataset = BBBC013VideoDataset(
        root_dir=str(tmp_path),
        first_characters=("A",),
    )
    sample = dataset[0]

    expected = torch.from_numpy(tifffile.imread(tmp_path / "A1.tif")).float()
    assert dataset.sequence_ids == ("A1",)
    assert dataset.shared_buffer.is_shared()
    assert sample["video"].shape == (1, 2, 12, 16)
    assert torch.equal(sample["video"][0], expected)
    assert sample["sequence_id"] == "A1"
    assert torch.equal(sample["frame_numbers"], torch.tensor([0]))


def test_bbbc013_dataset_normalizes_channels_independently(tmp_path: Path) -> None:
    _write_well(tmp_path / "A1.tif", offset=7)

    def normalize(video: torch.Tensor) -> tuple[torch.Tensor, None]:
        return video / video.max(), None

    dataset = BBBC013VideoDataset(
        root_dir=str(tmp_path),
        first_characters=("A",),
        normalization=normalize,
    )

    channel_maxima = dataset[0]["video"].amax(dim=(0, 2, 3))
    assert torch.equal(channel_maxima, torch.ones(2))


def test_bbbc013_datamodule_uses_default_train_and_eval_columns(
    tmp_path: Path,
) -> None:
    _write_plate(tmp_path, columns=tuple(range(1, 13)))
    datamodule = BBBC013VideoDataModule(
        data_dir=str(tmp_path),
        batch_size=3,
        num_workers=0,
        crop_size=8,
        drop_last=False,
    )

    datamodule.setup("fit")

    assert datamodule.train_ds is not None
    assert datamodule.val_ds is not None
    assert datamodule.train_ds.sequence_ids == tuple(
        f"{row}{column}" for row in "ABCDEFGH" for column in range(2, 12)
    )
    assert datamodule.val_ds.sequence_ids == tuple(
        f"{row}{column}" for row in "ABCDEFGH" for column in (1, 12)
    )
    assert next(iter(datamodule.train_dataloader()))["video"].shape == (
        3,
        1,
        2,
        8,
        8,
    )
    assert next(iter(datamodule.val_dataloader()))["video"].shape == (
        3,
        1,
        2,
        8,
        8,
    )


@pytest.mark.parametrize(
    "train_columns,eval_columns,match",
    [
        ((), (1,), "at least one"),
        ((2, 2), (1,), "unique"),
        ((0, 2), (1,), "between 1 and 12"),
        ((2, 3), (1, 3), "disjoint"),
    ],
)
def test_bbbc013_datamodule_rejects_invalid_split_columns(
    tmp_path: Path,
    train_columns: tuple[int, ...],
    eval_columns: tuple[int, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        BBBC013VideoDataModule(
            data_dir=str(tmp_path),
            train_columns=train_columns,
            eval_columns=eval_columns,
        )


def test_bbbc013_dataset_rejects_single_channel_conversion(tmp_path: Path) -> None:
    tifffile.imwrite(tmp_path / "A1.tif", np.zeros((12, 16), dtype=np.uint16))

    with pytest.raises(RuntimeError, match="Expected CYX TIFF image"):
        BBBC013VideoDataset(
            root_dir=str(tmp_path),
            first_characters=("A",),
        )
