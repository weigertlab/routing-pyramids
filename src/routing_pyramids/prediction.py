"""Prediction-time instance filtering, tracking, and CTC export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import tifffile
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from .types import SegmentationPredictionPayload


class CTCSegmentationPredictionWriter(BasePredictionWriter):
    """Write independent spatial segmentations in CTC result format."""

    def __init__(self, output_root: str | Path):
        """
        Initialize the segmentation writer.

        Parameters
        ----------
        output_root
            Root containing one output directory per input sequence.
        """
        super().__init__(write_interval="batch")
        self.output_root = Path(output_root)
        self._next_track_id: dict[str, int] = {}
        self._track_lines: dict[str, list[str]] = {}

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Write a prediction payload without retaining the full dataset in memory."""
        del trainer, pl_module, batch_indices, batch, batch_idx, dataloader_idx
        payload = self._validate_segmentation_payload(prediction)
        labels = payload["instance_labels"].detach().cpu().to(dtype=torch.int32)
        frame_numbers = payload["frame_numbers"].detach().cpu().to(dtype=torch.long)
        padding_widths = (
            payload["filename_padding_width"].detach().cpu().to(dtype=torch.long)
        )
        if labels.ndim != 3:
            raise ValueError(
                f"instance_labels must have shape (B, H, W), got {tuple(labels.shape)}"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        for index, sequence_id in enumerate(payload["sequence_ids"]):
            label_image = labels[index].numpy().astype("int64", copy=False)
            object_ids = np.unique(label_image)
            object_ids = object_ids[object_ids > 0]
            next_track_id = self._next_track_id.get(sequence_id, 1)
            last_track_id = next_track_id + len(object_ids) - 1
            if last_track_id > np.iinfo(np.uint32).max:
                raise ValueError(
                    "CTC mask export exceeded uint32 track IDs for sequence "
                    f"{sequence_id!r}"
                )
            relabeled = np.zeros(label_image.shape, dtype=np.uint32)
            if len(object_ids) > 0:
                track_ids = np.arange(
                    next_track_id,
                    next_track_id + len(object_ids),
                    dtype=np.uint32,
                )
                foreground = label_image > 0
                positions = np.searchsorted(object_ids, label_image[foreground])
                relabeled[foreground] = track_ids[positions]
            sequence_dir = self.output_root / f"{sequence_id}_RES"
            sequence_dir.mkdir(parents=True, exist_ok=True)
            frame_number = int(frame_numbers[index].item())
            padding_width = int(padding_widths[index].item())
            output_path = sequence_dir / f"mask{frame_number:0{padding_width}d}.tif"
            tifffile.imwrite(str(output_path), relabeled)
            track_lines = self._track_lines.setdefault(sequence_id, [])
            track_lines.extend(
                f"{track_id} {frame_number} {frame_number} 0"
                for track_id in range(next_track_id, last_track_id + 1)
            )
            self._next_track_id[sequence_id] = last_track_id + 1
            (sequence_dir / "res_track.txt").write_text(
                "\n".join(track_lines) + ("\n" if track_lines else "")
            )

    @staticmethod
    def _validate_segmentation_payload(
        prediction: Any,
    ) -> SegmentationPredictionPayload:
        if not isinstance(prediction, dict):
            raise TypeError(
                f"segmentation prediction payload must be a dict, got {type(prediction)}"
            )
        required = {
            "sequence_ids",
            "frame_numbers",
            "filename_padding_width",
            "instance_labels",
        }
        missing = required.difference(prediction)
        if missing:
            raise KeyError(
                "segmentation prediction payload is missing required keys: "
                f"{sorted(missing)}"
            )
        return cast(SegmentationPredictionPayload, prediction)


__all__ = ["CTCSegmentationPredictionWriter"]
