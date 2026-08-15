"""Tests for temporal loading of explicitly selected OME-Zarr stores."""

from pathlib import Path

import numpy as np
import pytest
import torch
import zarr
from torch.nn import functional as F
from torch.utils.data import Subset

from routing_pyramids.data.temporal_datamodule import OMEZarrVideoDataModule
from routing_pyramids.data.temporal_video_dataset import OMEZarrVideoDataset


def _write_ome_zarr(
    root: Path,
    store_name: str,
    data: np.ndarray,
    *,
    ome_axes: tuple[str, ...] = ("t", "c", "y", "x"),
    dimension_names: tuple[str, ...] = ("t", "c", "y", "x"),
    include_multiscales: bool = True,
) -> Path:
    store_dir = root / store_name
    group = zarr.create_group(str(store_dir), overwrite=True)
    if include_multiscales:
        group.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [{"name": axis} for axis in ome_axes],
                    "datasets": [{"path": "0"}],
                }
            ],
        }
    group.create_array(
        "0",
        data=data,
        chunks=(1, *data.shape[1:]),
        dimension_names=dimension_names,
    )
    return store_dir


class _AddOne:
    def __call__(
        self,
        video: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return video + 1, None


def test_ome_zarr_dataset_loads_exact_temporal_windows_into_shared_memory(
    tmp_path: Path,
) -> None:
    data = np.arange(4 * 1 * 4 * 6, dtype=np.uint16).reshape(4, 1, 4, 6)
    store_name = "video_a.ome.zarr"
    _write_ome_zarr(tmp_path, store_name, data)

    dataset = OMEZarrVideoDataset(
        root_dir=str(tmp_path),
        store_names=(store_name,),
        normalization=_AddOne(),
        sequence_length=2,
    )

    assert len(dataset) == 3
    assert dataset.sequence_ids == (store_name,)
    assert dataset.shared_buffer.is_shared()
    sample = dataset[1]
    expected = torch.from_numpy(data[1:3, 0].astype(np.float32)) + 1
    assert sample["video"].dtype == torch.float32
    assert torch.equal(sample["video"][:, 0], expected)
    assert torch.equal(sample["frame_numbers"], torch.tensor([1, 2]))
    assert sample["sequence_id"] == store_name


def test_ome_zarr_dataset_average_pooling_is_numerically_exact(
    tmp_path: Path,
) -> None:
    data = np.arange(3 * 1 * 4 * 6, dtype=np.uint16).reshape(3, 1, 4, 6)
    store_name = "video_a.ome.zarr"
    _write_ome_zarr(tmp_path, store_name, data)

    dataset = OMEZarrVideoDataset(
        root_dir=str(tmp_path),
        store_names=(store_name,),
        sequence_length=2,
        scale_factor=0.5,
    )

    expected = F.avg_pool2d(
        torch.from_numpy(data[:2].astype(np.float32)),
        kernel_size=2,
        stride=2,
    )
    assert torch.equal(dataset[0]["video"], expected)


def test_ome_zarr_datamodule_respects_explicit_splits(tmp_path: Path) -> None:
    train_name = "train.ome.zarr"
    val_name = "val.ome.zarr"
    train_data = np.arange(5 * 1 * 8 * 8, dtype=np.uint16).reshape(5, 1, 8, 8)
    val_data = train_data + 1000
    _write_ome_zarr(tmp_path, train_name, train_data)
    _write_ome_zarr(tmp_path, val_name, val_data)

    datamodule = OMEZarrVideoDataModule(
        data_dir=str(tmp_path),
        train_store_names=(train_name,),
        val_store_names=(val_name,),
        batch_size=2,
        num_workers=0,
        crop_size=4,
        sequence_length=2,
        temporal_source_length=3,
        drop_last=False,
    )
    datamodule.setup("fit")

    assert datamodule.train_ds is not None
    assert datamodule.val_ds is not None
    assert datamodule.train_ds.sequence_ids == (train_name,)
    assert datamodule.val_ds.sequence_ids == (val_name,)
    assert datamodule.val_ds.fixed_frame_indices is not None
    val_loader = datamodule.val_dataloader()
    assert isinstance(val_loader.dataset, Subset)
    assert next(iter(datamodule.train_dataloader()))["video"].shape == (2, 2, 1, 4, 4)
    assert next(iter(val_loader))["video"].shape == (2, 2, 1, 4, 4)


def test_ome_zarr_datamodule_builds_ground_truth_free_prediction_data(
    tmp_path: Path,
) -> None:
    store_name = "predict.ome.zarr"
    data = np.arange(3 * 1 * 8 * 8, dtype=np.uint16).reshape(3, 1, 8, 8)
    _write_ome_zarr(tmp_path, store_name, data)
    datamodule = OMEZarrVideoDataModule(
        data_dir=str(tmp_path),
        predict_store_names=(store_name,),
        batch_size=2,
        num_workers=0,
        crop_size=None,
        sequence_length=1,
        input_scale_factor=0.5,
    )

    datamodule.setup("predict")
    batch = next(iter(datamodule.predict_dataloader()))

    assert datamodule.train_ds is None
    assert datamodule.val_ds is None
    assert datamodule.predict_ds is not None
    assert batch["video"].shape == (2, 1, 1, 4, 4)
    assert batch["sequence_ids"] == [store_name, store_name]
    assert "instance_masks" not in batch


@pytest.mark.parametrize(
    "train_names,val_names,match",
    [
        ((), ("val.ome.zarr",), "train_store_names"),
        (("train.ome.zarr", "train.ome.zarr"), ("val.ome.zarr",), "unique"),
        (("same.ome.zarr",), ("same.ome.zarr",), "disjoint"),
    ],
)
def test_ome_zarr_datamodule_rejects_invalid_split_assignments(
    tmp_path: Path,
    train_names: tuple[str, ...],
    val_names: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        OMEZarrVideoDataModule(
            data_dir=str(tmp_path),
            train_store_names=train_names,
            val_store_names=val_names,
        )


def test_ome_zarr_dataset_rejects_missing_store(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        OMEZarrVideoDataset(
            root_dir=str(tmp_path),
            store_names=("missing.ome.zarr",),
        )


def test_ome_zarr_dataset_rejects_missing_multiscales(tmp_path: Path) -> None:
    data = np.zeros((3, 1, 4, 4), dtype=np.uint16)
    store_name = "video.ome.zarr"
    _write_ome_zarr(tmp_path, store_name, data, include_multiscales=False)

    with pytest.raises(TypeError, match="OME metadata"):
        OMEZarrVideoDataset(root_dir=str(tmp_path), store_names=(store_name,))


def test_ome_zarr_dataset_rejects_invalid_ome_axes(tmp_path: Path) -> None:
    data = np.zeros((3, 1, 4, 4), dtype=np.uint16)
    store_name = "video.ome.zarr"
    _write_ome_zarr(tmp_path, store_name, data, ome_axes=("t", "z", "y", "x"))

    with pytest.raises(ValueError, match="Expected OME axes"):
        OMEZarrVideoDataset(
            root_dir=str(tmp_path),
            store_names=(store_name,),
        )


def test_ome_zarr_dataset_rejects_invalid_array_dimension_names(
    tmp_path: Path,
) -> None:
    data = np.zeros((3, 1, 4, 4), dtype=np.uint16)
    store_name = "video.ome.zarr"
    _write_ome_zarr(
        tmp_path,
        store_name,
        data,
        dimension_names=("t", "z", "y", "x"),
    )

    with pytest.raises(ValueError, match="Expected Zarr dimension names"):
        OMEZarrVideoDataset(
            root_dir=str(tmp_path),
            store_names=(store_name,),
        )


def test_ome_zarr_dataset_rejects_multiple_channels(tmp_path: Path) -> None:
    data = np.zeros((3, 2, 4, 4), dtype=np.uint16)
    store_name = "video.ome.zarr"
    _write_ome_zarr(tmp_path, store_name, data)

    with pytest.raises(ValueError, match="Expected one image channel"):
        OMEZarrVideoDataset(
            root_dir=str(tmp_path),
            store_names=(store_name,),
        )


def test_ome_zarr_dataset_rejects_incompatible_pooling(tmp_path: Path) -> None:
    data = np.zeros((3, 1, 5, 4), dtype=np.uint16)
    store_name = "video.ome.zarr"
    _write_ome_zarr(tmp_path, store_name, data)

    with pytest.raises(ValueError, match="must be divisible"):
        OMEZarrVideoDataset(
            root_dir=str(tmp_path),
            store_names=(store_name,),
            scale_factor=0.5,
        )
