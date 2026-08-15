"""Focused tests for shared temporal datamodule wiring."""

import numpy as np
import pytest
import tifffile
from torch.utils.data import ConcatDataset, Subset

from routing_pyramids.augmentation import PhotometricAugmentationConfig
from routing_pyramids.data.temporal_datamodule import VideoTemporalDataModule
from routing_pyramids.data.temporal_video_dataset import CTCVideoDataset
from routing_pyramids.data.transforms import DetectionTransform


def _write_frames(base_dir, num_frames: int, value_offset: int) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        img = np.full((64, 64), fill_value=(value_offset + i), dtype=np.uint8)
        tifffile.imwrite(base_dir / f"t{i:06d}.tif", img)


def _prepare_ctc_layout(root):
    for split in ["train", "test"]:
        for video_name in ["01", "02"]:
            _write_frames(root / split / video_name, num_frames=12, value_offset=10)


def test_video_temporal_datamodule_wiring(tmp_path):
    _prepare_ctc_layout(tmp_path)

    dm = VideoTemporalDataModule(
        data_dir=str(tmp_path),
        dataset_class=CTCVideoDataset,
        train_split="test",
        val_split="train",
        batch_size=2,
        num_workers=0,
        crop_size=32,
        sequence_length=4,
        temporal_source_length=8,
    )
    dm.setup("fit")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    assert train_loader.multiprocessing_context is None
    assert val_loader.multiprocessing_context is None
    assert isinstance(val_loader.dataset, Subset)
    assert dm.val_ds is not None
    assert dm.val_ds.fixed_frame_indices is not None


def test_video_temporal_datamodule_fixed_stride_all_stages(tmp_path):
    _prepare_ctc_layout(tmp_path)

    dm = VideoTemporalDataModule(
        data_dir=str(tmp_path),
        dataset_class=CTCVideoDataset,
        train_split="test",
        val_split="train",
        test_split="train",
        batch_size=2,
        num_workers=0,
        crop_size=32,
        sequence_length=4,
        temporal_frame_stride=2,
    )
    dm.setup(None)

    expected = [[0, 2, 4, 6], [0, 2, 4, 6]]
    train_batch = next(iter(dm.train_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))

    assert train_batch["frame_indices"].tolist() == expected
    assert val_batch["frame_indices"].tolist() == expected
    assert test_batch["frame_indices"].tolist() == expected
    assert dm.val_ds is not None
    assert dm.val_ds.fixed_frame_indices is None


def test_video_temporal_datamodule_default_train_repeat_factor(tmp_path):
    _prepare_ctc_layout(tmp_path)

    dm = VideoTemporalDataModule(
        data_dir=str(tmp_path),
        dataset_class=CTCVideoDataset,
        train_split="test",
        val_split="train",
        batch_size=2,
        num_workers=0,
        crop_size=32,
        sequence_length=4,
        temporal_source_length=8,
    )
    dm.setup("fit")

    train_loader = dm.train_dataloader()

    assert dm.train_ds is not None
    assert train_loader.dataset is dm.train_ds
    assert len(train_loader) == len(dm.train_ds) // dm.batch_size


def test_video_temporal_datamodule_test_crop_size_none_emits_full_frames(tmp_path):
    _prepare_ctc_layout(tmp_path)

    dm = VideoTemporalDataModule(
        data_dir=str(tmp_path),
        dataset_class=CTCVideoDataset,
        train_split="test",
        val_split="train",
        test_split="train",
        batch_size=2,
        num_workers=0,
        crop_size=None,
        sequence_length=1,
    )
    dm.setup("test")

    batch = next(iter(dm.test_dataloader()))

    assert batch["video"].shape[-2:] == (64, 64)


def test_video_temporal_datamodule_train_repeat_factor_reuses_cache(tmp_path):
    _prepare_ctc_layout(tmp_path)

    dm = VideoTemporalDataModule(
        data_dir=str(tmp_path),
        dataset_class=CTCVideoDataset,
        train_split="test",
        val_split="train",
        batch_size=2,
        num_workers=0,
        crop_size=32,
        sequence_length=4,
        temporal_source_length=8,
        train_repeat_factor=3,
    )
    dm.setup("fit")
    assert dm.train_ds is not None

    shared_buffer = dm.train_ds.shared_buffer
    buffer_ptr = shared_buffer.untyped_storage().data_ptr()
    train_loader = dm.train_dataloader()

    assert isinstance(train_loader.dataset, ConcatDataset)
    assert len(train_loader.dataset) == len(dm.train_ds) * dm.train_repeat_factor
    assert len(train_loader) == len(train_loader.dataset) // dm.batch_size
    assert all(dataset is dm.train_ds for dataset in train_loader.dataset.datasets)

    _ = next(iter(train_loader))
    assert shared_buffer.untyped_storage().data_ptr() == buffer_ptr


@pytest.mark.parametrize("train_repeat_factor", [0, -1, 1.5, True])
def test_video_temporal_datamodule_rejects_invalid_train_repeat_factor(
    tmp_path, train_repeat_factor
):
    _prepare_ctc_layout(tmp_path)
    expected_error = TypeError if train_repeat_factor in (True, 1.5) else ValueError
    with pytest.raises(expected_error, match="train_repeat_factor"):
        VideoTemporalDataModule(
            data_dir=str(tmp_path),
            dataset_class=CTCVideoDataset,
            train_split="test",
            val_split="train",
            batch_size=2,
            num_workers=0,
            crop_size=32,
            sequence_length=4,
            train_repeat_factor=train_repeat_factor,
        )


def test_video_datamodule_wires_photometric_augmentation_to_training_only(tmp_path):
    _prepare_ctc_layout(tmp_path)
    augmentation = PhotometricAugmentationConfig.from_ranges(
        scale=0.0,
        shift=(1.0, 1.0),
        apply_prob=1.0,
    )
    dm = VideoTemporalDataModule(
        data_dir=str(tmp_path),
        dataset_class=CTCVideoDataset,
        train_split="test",
        val_split="train",
        batch_size=2,
        num_workers=0,
        crop_size=32,
        sequence_length=4,
        photometric_augmentation=augmentation,
    )

    dm.setup("fit")

    assert dm.train_ds is not None
    assert dm.val_ds is not None
    train_augmentations = dm.train_ds.augmentations
    val_augmentations = dm.val_ds.augmentations
    assert isinstance(train_augmentations, DetectionTransform)
    assert isinstance(val_augmentations, DetectionTransform)
    assert train_augmentations.photometric_augmentation is augmentation
    assert val_augmentations.photometric_augmentation is None
