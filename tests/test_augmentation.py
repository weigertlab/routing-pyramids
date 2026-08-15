"""Tests for shared photometric augmentation."""

from typing import Any

import pytest
import torch
from torch.testing import assert_close

from routing_pyramids.augmentation import PhotometricAugmentationConfig


@pytest.fixture(autouse=True)
def _preserve_torch_rng_state():
    with torch.random.fork_rng():
        yield


def test_photometric_strength_is_independent_per_channel_and_spatially_coherent():
    augmentation = PhotometricAugmentationConfig.from_ranges(
        scale=0.0,
        shift=1.0,
        noise_std=0.0,
        apply_prob=1.0,
    )
    torch.manual_seed(0)
    image = torch.zeros(2, 8, 3, 4, 4)

    out = augmentation(image)

    channel_values = out[..., 0, 0, 0]
    assert torch.unique(channel_values).numel() > 2
    assert_close(out, channel_values[..., None, None, None].expand_as(out))


def test_photometric_apply_probability_is_independent_per_channel():
    augmentation = PhotometricAugmentationConfig.from_ranges(
        scale=0.0,
        shift=(1.0, 1.0),
        noise_std=0.0,
        apply_prob=0.5,
    )
    torch.manual_seed(0)
    image = torch.zeros(1, 32, 4, 4)

    out = augmentation(image)

    channel_changed = out[:, :, 0, 0] > 0.0
    assert channel_changed.any()
    assert (~channel_changed).any()
    assert_close(out, out[:, :, :1, :1].expand_as(out))


def test_photometric_gamma_preserves_each_channel_range():
    augmentation = PhotometricAugmentationConfig.from_ranges(
        scale=0.0,
        shift=0.0,
        gamma=2.0,
        noise_std=0.0,
        apply_prob=1.0,
    )
    image = torch.stack(
        [
            torch.linspace(-2.0, 3.0, 16).view(4, 4),
            torch.linspace(10.0, 30.0, 16).view(4, 4),
        ],
        dim=0,
    ).unsqueeze(0)

    out = augmentation(image)

    assert_close(out.amin(dim=(-2, -1)), image.amin(dim=(-2, -1)))
    assert_close(out.amax(dim=(-2, -1)), image.amax(dim=(-2, -1)))
    assert not torch.equal(out, image)


def test_photometric_noise_samples_strength_per_channel_and_noise_per_pixel():
    augmentation = PhotometricAugmentationConfig.from_ranges(
        scale=0.0,
        shift=0.0,
        noise_std=1.0,
        apply_prob=1.0,
    )
    torch.manual_seed(0)
    image = torch.zeros(1, 8, 64, 64)

    out = augmentation(image)

    channel_std = out.std(dim=(-2, -1))
    assert torch.unique(channel_std).numel() == image.shape[1]
    assert (channel_std > 0.0).all()
    assert (out[:, :, 0, 0] != out[:, :, 0, 1]).all()


def test_photometric_apply_probability_zero_keeps_image():
    augmentation = PhotometricAugmentationConfig.from_ranges(
        scale=1.0,
        shift=1.0,
        gamma=(0.5, 2.0),
        noise_std=1.0,
        apply_prob=0.0,
    )
    image = torch.randn(4, 3, 4, 4)

    assert augmentation(image) is image


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"scale": (1.0, 0.5)}, "photometric_scale range"),
        ({"scale": (-0.1, 1.0)}, "photometric_scale lower bound"),
        ({"shift": (1.0, -1.0)}, "photometric_shift range"),
        ({"gamma": (2.0, 1.0)}, "photometric_gamma range"),
        ({"gamma": (-0.1, 1.0)}, "photometric_gamma lower bound"),
        ({"noise_std": -0.1}, "photometric_noise_std"),
        ({"apply_prob": 1.5}, "photometric_apply_prob"),
    ],
)
def test_photometric_rejects_invalid_config(kwargs: dict[str, Any], match: str):
    with pytest.raises(ValueError, match=match):
        PhotometricAugmentationConfig.from_ranges(**kwargs)


def test_photometric_rejects_tensor_without_batch_and_channel_dimensions():
    augmentation = PhotometricAugmentationConfig.from_ranges()

    with pytest.raises(ValueError, match="batch, channels"):
        augmentation(torch.zeros(2, 2))
