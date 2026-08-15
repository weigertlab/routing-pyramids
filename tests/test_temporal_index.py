import pytest
import torch

from routing_pyramids.data.temporal_index import (
    TemporalFrameSelector,
    resolve_temporal_source_length,
)


def test_resolve_temporal_source_length_derives_stride_span():
    assert (
        resolve_temporal_source_length(
            sequence_length=4,
            temporal_source_length=None,
            temporal_frame_stride=2,
        )
        == 7
    )


def test_resolve_temporal_source_length_rejects_short_stride_span():
    with pytest.raises(ValueError, match="temporal_frame_stride"):
        resolve_temporal_source_length(
            sequence_length=4,
            temporal_source_length=6,
            temporal_frame_stride=2,
        )


def test_temporal_frame_selector_fixed_stride_indices():
    selector = TemporalFrameSelector(
        sequence_length=4,
        source_length=7,
        temporal_frame_stride=2,
    )

    assert torch.equal(
        selector.sample_frame_indices(),
        torch.tensor([0, 2, 4, 6], dtype=torch.long),
    )
    assert torch.equal(
        selector.frame_indices_for_sample(0), selector.sample_frame_indices()
    )


def test_temporal_frame_selector_rejects_invalid_stride():
    with pytest.raises(ValueError, match="temporal_frame_stride"):
        TemporalFrameSelector(
            sequence_length=4,
            source_length=4,
            temporal_frame_stride=0,
        )
