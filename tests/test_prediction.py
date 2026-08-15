from typing import Any, cast

import tifffile
import torch

from routing_pyramids.prediction import CTCSegmentationPredictionWriter


def test_segmentation_prediction_writer_writes_sequence_frames(tmp_path):
    labels = torch.zeros(2, 4, 5, dtype=torch.int32)
    labels[0, :2, :2] = 1
    labels[1, 2:, 2:] = 2
    writer = CTCSegmentationPredictionWriter(tmp_path)
    writer.write_on_batch_end(
        trainer=cast(Any, None),
        pl_module=cast(Any, None),
        prediction={
            "sequence_ids": ["a.ome.zarr", "b.ome.zarr"],
            "frame_numbers": torch.tensor([3, 12]),
            "filename_padding_width": torch.tensor([2, 2]),
            "instance_labels": labels,
        },
        batch_indices=None,
        batch=None,
        batch_idx=0,
        dataloader_idx=0,
    )

    first_dir = tmp_path / "a.ome.zarr_RES"
    second_dir = tmp_path / "b.ome.zarr_RES"
    first = tifffile.imread(first_dir / "mask03.tif")
    second = tifffile.imread(second_dir / "mask12.tif")
    assert first.dtype.name == "uint32"
    assert first.max() == 1
    assert second.max() == 1
    assert (first_dir / "res_track.txt").read_text() == "1 3 3 0\n"
    assert (second_dir / "res_track.txt").read_text() == "1 12 12 0\n"
