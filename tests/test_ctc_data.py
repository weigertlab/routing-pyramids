"""Unit tests for CTC temporal data loading."""

from functools import partial

import numpy as np
import pytest
import tifffile
import torch
from torch.utils.data import Subset

from routing_pyramids.data.sampling import fixed_shuffled_indices
from routing_pyramids.data.temporal_datamodule import VideoTemporalDataModule
from routing_pyramids.data.temporal_video_dataset import CTCVideoDataset


@pytest.fixture
def mock_ctc_data_dir(tmp_path):
    """Create a temporary CTC-style directory with multiple videos."""
    for split in ["train", "test"]:
        for video_name in ["01", "02"]:
            video_dir = tmp_path / split / video_name
            video_dir.mkdir(parents=True)
            for i in range(16):
                img = np.random.randint(0, 255, (96, 96), dtype=np.uint8)
                tifffile.imwrite(video_dir / f"t{i:03d}.tif", img)

    yield tmp_path


@pytest.fixture
def mock_ctc_data_with_gt_dir(tmp_path):
    """Create a CTC-style directory with TRA masks and sparse SEG masks."""
    for split in ["train", "test"]:
        for video_name in ["01", "02"]:
            video_dir = tmp_path / split / video_name
            tra_dir = tmp_path / split / f"{video_name}_GT" / "TRA"
            seg_dir = tmp_path / split / f"{video_name}_GT" / "SEG"
            video_dir.mkdir(parents=True)
            tra_dir.mkdir(parents=True)
            seg_dir.mkdir(parents=True)
            for frame_idx in range(16):
                img = np.full((96, 96), frame_idx, dtype=np.uint8)
                tifffile.imwrite(video_dir / f"t{frame_idx:03d}.tif", img)

                tra = np.zeros((96, 96), dtype=np.uint16)
                tra[20, 21] = frame_idx + 1
                tra[32, 33] = frame_idx + 17
                tifffile.imwrite(tra_dir / f"man_track{frame_idx:03d}.tif", tra)

                if frame_idx == 0:
                    seg = np.zeros((96, 96), dtype=np.uint16)
                    seg[22:25, 26:29] = 7
                    tifffile.imwrite(seg_dir / "man_seg000.tif", seg)

    yield tmp_path


def test_ctc_dataset_contract_and_shared_storage(mock_ctc_data_dir):
    ds = CTCVideoDataset(
        root_dir=str(mock_ctc_data_dir),
        split="train",
        normalization=None,
        augmentations=None,
        sequence_length=4,
    )

    assert len(ds.videos) == 2
    assert len(ds) == 26
    assert ds.shared_buffer.is_shared()

    buffer_ptr = ds.shared_buffer.untyped_storage().data_ptr()
    for video in ds.videos:
        assert video.untyped_storage().data_ptr() == buffer_ptr

    sample = ds[0]
    assert sample["video"].shape == (4, 1, 96, 96)
    assert sample["frame_indices"].shape == (4,)
    assert sample["frame_indices"].dtype == torch.long
    assert int(sample["source_length"]) == 4
    assert sample["sequence_id"] == "01"
    assert torch.equal(sample["frame_numbers"], torch.tensor([0, 1, 2, 3]))
    assert int(sample["filename_padding_width"]) == 3


def test_ctc_val_order_fixed_and_non_sequential(mock_ctc_data_dir):
    seed = 123
    torch.manual_seed(seed)

    dm = VideoTemporalDataModule(
        data_dir=str(mock_ctc_data_dir),
        dataset_class=CTCVideoDataset,
        train_split="test",
        val_split="train",
        batch_size=2,
        num_workers=0,
        crop_size=64,
        sequence_length=4,
    )
    dm.setup("fit")

    assert dm.val_ds is not None
    expected = fixed_shuffled_indices(len(dm.val_ds), seed=seed)
    assert dm._val_indices == expected
    assert expected != list(range(len(dm.val_ds)))

    val_loader_1 = dm.val_dataloader()
    val_loader_2 = dm.val_dataloader()

    assert isinstance(val_loader_1.dataset, Subset)
    assert isinstance(val_loader_2.dataset, Subset)
    assert list(val_loader_1.dataset.indices) == expected
    assert list(val_loader_2.dataset.indices) == expected


def test_ctc_temporal_subsample_fixed_endpoints_and_val_deterministic(
    mock_ctc_data_dir,
):
    torch.manual_seed(77)
    dm = VideoTemporalDataModule(
        data_dir=str(mock_ctc_data_dir),
        dataset_class=CTCVideoDataset,
        train_split="test",
        val_split="train",
        batch_size=3,
        num_workers=0,
        crop_size=64,
        sequence_length=4,
        temporal_source_length=8,
    )
    dm.setup("fit")

    train_batch = next(iter(dm.train_dataloader()))
    assert train_batch["video"].shape[1] == 4
    assert train_batch["frame_indices"].shape == (3, 4)
    assert int(train_batch["source_length"]) == 8
    assert torch.all(train_batch["frame_indices"][:, 0] == 0)
    assert torch.all(train_batch["frame_indices"][:, -1] == 7)
    assert torch.all(torch.diff(train_batch["frame_indices"], dim=1) > 0)

    val_batch_1 = next(iter(dm.val_dataloader()))
    val_batch_2 = next(iter(dm.val_dataloader()))
    assert torch.equal(val_batch_1["frame_indices"], val_batch_2["frame_indices"])


def test_ctc_predict_loader_emits_full_frames_and_metadata(mock_ctc_data_dir):
    dm = VideoTemporalDataModule(
        data_dir=str(mock_ctc_data_dir),
        dataset_class=CTCVideoDataset,
        train_split="test",
        val_split="train",
        test_split="test",
        batch_size=2,
        num_workers=0,
        crop_size=64,
        sequence_length=4,
    )

    dm.setup("predict")
    batch = next(iter(dm.predict_dataloader()))

    assert batch["video"].shape == (2, 4, 1, 96, 96)
    assert batch["sequence_ids"] == ["01", "01"]
    assert batch["frame_numbers"].shape == (2, 4)
    assert torch.equal(batch["frame_numbers"][0], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(batch["filename_padding_width"], torch.tensor([3, 3]))


def test_ctc_val_temporal_indices_not_affected_by_prior_rng_consumption(
    mock_ctc_data_dir,
):
    dm = partial(
        VideoTemporalDataModule,
        data_dir=str(mock_ctc_data_dir),
        dataset_class=CTCVideoDataset,
        train_split="test",
        val_split="train",
        batch_size=2,
        num_workers=0,
        crop_size=64,
        sequence_length=4,
        temporal_source_length=8,
    )

    torch.manual_seed(123)
    dm_a = dm()
    dm_a.setup("fit")
    assert dm_a.val_ds is not None
    idx_a = dm_a.val_ds.fixed_frame_indices
    assert idx_a is not None

    torch.manual_seed(123)
    _ = torch.rand(10_000)
    dm_b = dm()
    dm_b.setup("fit")
    assert dm_b.val_ds is not None
    idx_b = dm_b.val_ds.fixed_frame_indices
    assert idx_b is not None

    assert len(idx_a) == len(idx_b)
    for a, b in zip(idx_a, idx_b):
        assert torch.equal(a, b)
