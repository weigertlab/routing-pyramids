import pytest
import torch

from routing_pyramids.data.sampling import (
    fixed_shuffled_indices,
    temporal_subsample_indices,
)


def test_fixed_shuffled_indices_same_seed_same_permutation():
    n = 32
    seed = 123

    a = fixed_shuffled_indices(n, seed=seed)
    b = fixed_shuffled_indices(n, seed=seed)

    assert a == b
    assert sorted(a) == list(range(n))


def test_fixed_shuffled_indices_different_seed_different_permutation():
    n = 32
    a = fixed_shuffled_indices(n, seed=1)
    b = fixed_shuffled_indices(n, seed=2)

    assert a != b
    assert sorted(a) == list(range(n))
    assert sorted(b) == list(range(n))


def test_fixed_shuffled_indices_uses_global_seed_when_not_provided():
    n = 20
    torch.manual_seed(77)

    expected = fixed_shuffled_indices(n, seed=torch.initial_seed())
    observed = fixed_shuffled_indices(n)

    assert observed == expected


def test_temporal_subsample_indices_fixed_endpoints_and_sorted_unique():
    torch.manual_seed(42)
    idx = temporal_subsample_indices(source_length=8, keep_length=4)
    assert idx.shape == (4,)
    assert idx[0].item() == 0
    assert idx[-1].item() == 7
    assert torch.all(torch.diff(idx) > 0)
    assert len(torch.unique(idx)) == 4


def test_temporal_subsample_indices_identity_when_no_subsampling():
    idx = temporal_subsample_indices(source_length=4, keep_length=4)
    assert torch.equal(idx, torch.tensor([0, 1, 2, 3]))


def test_temporal_subsample_indices_rejects_invalid_lengths():
    with pytest.raises(ValueError):
        temporal_subsample_indices(source_length=4, keep_length=5)
    with pytest.raises(ValueError):
        temporal_subsample_indices(source_length=8, keep_length=1)
