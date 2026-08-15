"""Tests for shared-memory contiguous frame storage."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
from torch import Tensor

from routing_pyramids.data.frame_catalog import FrameSequence
from routing_pyramids.data.shared_frame_bank import build_shared_frame_bank


def _write_frame_series(base_dir: Path, values: list[int]) -> tuple[Path, ...]:
    base_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for index, value in enumerate(values):
        frame = np.full((16, 16), fill_value=value, dtype=np.uint8)
        path = base_dir / f"t{index:06d}.tif"
        tifffile.imwrite(path, frame)
        files.append(path)
    return tuple(files)


def _write_frame(path: Path, frame: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, frame)
    return path


def test_shared_frame_bank_is_contiguous_and_shared(tmp_path):
    seq_a = FrameSequence(
        sequence_id="a",
        frame_files=_write_frame_series(tmp_path / "a", [1, 2, 3]),
    )
    seq_b = FrameSequence(
        sequence_id="b",
        frame_files=_write_frame_series(tmp_path / "b", [4, 5]),
    )

    bank = build_shared_frame_bank(
        sequences=(seq_a, seq_b),
        normalization=None,
        owner_name="test-bank",
        scale_factor=1,
    )

    assert bank.buffer.is_shared()
    assert bank.buffer.numel() == 5 * 16 * 16
    assert len(bank.videos) == 2
    assert bank.videos[0].shape == (3, 16, 16)
    assert bank.videos[1].shape == (2, 16, 16)

    buffer_ptr = bank.buffer.untyped_storage().data_ptr()
    for video in bank.videos:
        assert video.untyped_storage().data_ptr() == buffer_ptr

    assert bank.videos[0][0, 0, 0].item() == 1.0
    assert bank.videos[0][2, 0, 0].item() == 3.0
    assert bank.videos[1][0, 0, 0].item() == 4.0


def test_shared_frame_bank_avg_pools_before_normalization(tmp_path):
    frame = np.arange(16, dtype=np.uint8).reshape(4, 4)
    sequence = FrameSequence(
        sequence_id="pooled",
        frame_files=(_write_frame(tmp_path / "pooled" / "t000000.tif", frame),),
    )
    expected = torch.tensor(
        [[[2.5, 4.5], [10.5, 12.5]]],
        dtype=torch.float32,
    )
    seen_by_normalization: list[Tensor] = []

    def normalize(video):
        seen_by_normalization.append(video.clone())
        return video + 1.0, None

    bank = build_shared_frame_bank(
        sequences=(sequence,),
        normalization=normalize,
        owner_name="pooled-test",
        scale_factor=0.5,
    )

    assert bank.buffer.numel() == 4
    assert bank.videos[0].shape == (1, 2, 2)
    assert torch.allclose(seen_by_normalization[0], expected)
    assert torch.allclose(bank.videos[0], expected + 1.0)


def test_shared_frame_bank_bilinearly_upsamples_before_normalization(tmp_path):
    frame = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    sequence = FrameSequence(
        sequence_id="upsampled",
        frame_files=(_write_frame(tmp_path / "upsampled" / "t000000.tif", frame),),
    )
    expected = torch.tensor(
        [
            [
                [0.0, 0.25, 0.75, 1.0],
                [0.5, 0.75, 1.25, 1.5],
                [1.5, 1.75, 2.25, 2.5],
                [2.0, 2.25, 2.75, 3.0],
            ]
        ]
    )
    seen_by_normalization: list[Tensor] = []

    def normalize(video):
        seen_by_normalization.append(video.clone())
        return video + 1.0, None

    bank = build_shared_frame_bank(
        sequences=(sequence,),
        normalization=normalize,
        owner_name="upsampled-test",
        scale_factor=2,
    )

    assert bank.buffer.numel() == 16
    assert bank.videos[0].shape == (1, 4, 4)
    torch.testing.assert_close(seen_by_normalization[0], expected)
    torch.testing.assert_close(bank.videos[0], expected + 1.0)


def test_shared_frame_bank_downscale_requires_divisible_shape(tmp_path):
    sequence = FrameSequence(
        sequence_id="bad-shape",
        frame_files=(
            _write_frame(
                tmp_path / "bad-shape" / "t000000.tif",
                np.zeros((5, 4), dtype=np.uint8),
            ),
        ),
    )

    with pytest.raises(ValueError, match="must be divisible by reduction factor 2"):
        build_shared_frame_bank(
            sequences=(sequence,),
            normalization=None,
            owner_name="bad-shape-test",
            scale_factor=0.5,
        )


def test_shared_frame_bank_scale_factor_must_be_positive(tmp_path):
    sequence = FrameSequence(
        sequence_id="a",
        frame_files=_write_frame_series(tmp_path / "a", [1]),
    )

    with pytest.raises(ValueError, match="greater than 0"):
        build_shared_frame_bank(
            sequences=(sequence,),
            normalization=None,
            owner_name="bad-factor-test",
            scale_factor=0,
        )


@pytest.mark.parametrize(
    "scale_factor", [-1, 0.75, 1.5, float("inf"), float("nan"), True]
)
def test_shared_frame_bank_rejects_invalid_scale_factors(tmp_path, scale_factor):
    sequence = FrameSequence(
        sequence_id="a",
        frame_files=_write_frame_series(tmp_path / "a", [1]),
    )
    error_type = TypeError if scale_factor is True else ValueError
    with pytest.raises(error_type, match="scale_factor"):
        build_shared_frame_bank(
            sequences=(sequence,),
            normalization=None,
            owner_name="bad-factor-test",
            scale_factor=scale_factor,
        )


def test_shared_frame_bank_fails_fast_when_share_memory_fails(tmp_path, monkeypatch):
    seq = FrameSequence(
        sequence_id="a",
        frame_files=_write_frame_series(tmp_path / "a", [7, 8]),
    )

    def raise_share_memory(self):
        raise OSError("no shared memory")

    monkeypatch.setattr(torch.Tensor, "share_memory_", raise_share_memory)

    with pytest.raises(RuntimeError, match=r"failed to allocate shared frame bank"):
        build_shared_frame_bank(
            sequences=(seq,),
            normalization=None,
            owner_name="ctc-failure-test",
        )
