"""Unit tests for data transforms."""

import pytest
import torch
import torchvision.transforms.v2 as T
from torch import Tensor
from torch.testing import assert_close

from routing_pyramids.augmentation import PhotometricAugmentationConfig
from routing_pyramids.data.transforms import (
    DetectionTransform,
    QuantileNormalize,
    RandomTemporalCrop,
    _get_training_crop,
    get_normalization,
    get_transforms,
)


def test_get_normalization():
    """Verify get_normalization returns a working transform."""
    norm = get_normalization()
    # (T, H, W)
    data = torch.tensor([0.0, 50.0, 100.0]).view(1, 1, 3)
    out, _ = norm(data)
    assert out.shape == data.shape
    assert torch.abs(out[0, 0, 1]) < 1e-5  # p50 maps to 0


def test_quantile_normalize_clips_and_reuses_quantiles(monkeypatch):
    """Verify clipping uses a single sampled-quantile pass."""
    data = torch.arange(1000, dtype=torch.float32)
    data[0] = -1000.0
    data[-1] = 10_000.0
    video = data.view(1, 20, 50)

    quantile_calls: list[Tensor] = []
    original_quantile = torch.quantile

    def wrapped_quantile(input: Tensor, q: Tensor | float, *args, **kwargs):
        quantile_calls.append(torch.as_tensor(q).detach().cpu())
        return original_quantile(input, q, *args, **kwargs)

    monkeypatch.setattr(torch, "quantile", wrapped_quantile)

    out = QuantileNormalize()(video)

    assert len(quantile_calls) == 1
    assert_close(quantile_calls[0], torch.tensor([0.001, 0.50, 0.99, 0.999]))

    p001, p50, p99, p999 = original_quantile(
        data, torch.tensor([0.001, 0.50, 0.99, 0.999])
    )
    expected = video.clamp(min=p001, max=p999)
    expected = (expected - p50) / (p99 - p50 + 1e-8)
    assert_close(out, expected)


def test_frame_consistency():
    """Verify that transforms are applied identically across frames."""
    # Create identical frames with distinct markings (T, H, W)
    T, H, W = 4, 100, 100
    crop_size = 50

    # Create a pattern: gradient in H, W
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    pattern = (y + x).float()

    # Stack to create T identical frames
    chunk = pattern.expand(T, H, W)

    # Get training transforms (random)
    transforms = get_transforms(is_train=True, crop_size=crop_size)

    # Apply transforms multiple times to ensure robustness
    for t_idx in range(5):
        out, _ = transforms(chunk)

        # Verify shape
        assert out.shape == (T, crop_size, crop_size)

        # Verify all frames are identical
        frame0 = out[0]
        for t in range(1, T):
            # Use a small tolerance for float comparison strictly
            diff = torch.abs(out[t] - frame0).max()
            assert diff < 1e-6, f"Frame {t} differs from Frame 0 in iteration {t_idx}"


def test_eval_transforms_crop_size_none_preserves_full_frame() -> None:
    chunk = torch.arange(3 * 7 * 9, dtype=torch.float32).view(3, 7, 9)
    transforms = get_transforms(is_train=False, crop_size=None)

    out, _ = transforms(chunk)

    torch.testing.assert_close(out, chunk)


def test_train_transforms_reject_crop_size_none() -> None:
    with pytest.raises(ValueError, match="crop_size=None"):
        get_transforms(is_train=True, crop_size=None)


def test_random_temporal_crop_interpolates_crop_locations(monkeypatch):
    """Verify shifted crops linearly interpolate between first and last frames."""
    pattern = torch.arange(64, dtype=torch.float32).view(8, 8)
    chunk = pattern.expand(4, 8, 8)
    transform = DetectionTransform(
        T.Compose(
            [
                RandomTemporalCrop(
                    (4, 4),
                    max_shift=2,
                )
            ]
        )
    )

    values = [0, 0, 2, 2]

    def fake_randint(low, high, size, **kwargs):
        del high, size, kwargs
        return torch.tensor(values.pop(0) + int(low))

    monkeypatch.setattr(torch, "randint", fake_randint)

    out, _ = transform(chunk)

    assert out.shape == (4, 4, 4)
    assert_close(out[0], pattern[0:4, 0:4])
    assert_close(out[1], pattern[1:5, 1:5])
    assert_close(out[2], pattern[1:5, 1:5])
    assert_close(out[3], pattern[2:6, 2:6])


def test_random_temporal_crop_preserves_channel_dimension(monkeypatch):
    """Verify shifted crops work for video tensors with channels."""
    pattern = torch.arange(64, dtype=torch.float32).view(8, 8)
    video = torch.stack([pattern, pattern + 100], dim=0)
    video = video.unsqueeze(0).expand(4, 2, 8, 8)
    transform = RandomTemporalCrop((4, 4), max_shift=2)

    values = [0, 0, 2, 2]

    def fake_randint(low, high, size, **kwargs):
        del high, size, kwargs
        return torch.tensor(values.pop(0) + int(low))

    monkeypatch.setattr(torch, "randint", fake_randint)

    out = transform(video)

    assert isinstance(out, Tensor)
    assert out.shape == (4, 2, 4, 4)
    assert_close(out[0, 0], pattern[0:4, 0:4])
    assert_close(out[3, 1], pattern[2:6, 2:6] + 100)


@pytest.mark.parametrize(
    ("probability", "max_shift"),
    [
        (0.0, 2),
        (0.5, 0),
    ],
)
def test_training_crop_uses_static_crop_when_shift_is_disabled(probability, max_shift):
    """Verify disabled shift settings produce the box-compatible crop branch."""
    crop = _get_training_crop(
        4,
        temporal_crop_shift_probability=probability,
        temporal_crop_max_shift=max_shift,
    )

    assert isinstance(crop, T.RandomCrop)


def test_training_crop_uses_temporal_crop_when_shift_is_certain():
    """Verify probability 1 selects the temporal crop directly."""
    crop = _get_training_crop(
        4,
        temporal_crop_shift_probability=1.0,
        temporal_crop_max_shift=(2, 3),
    )

    assert isinstance(crop, RandomTemporalCrop)
    assert crop.max_shift == (2, 3)


def test_training_crop_uses_random_choice_for_fractional_shift_probability():
    """Verify fractional shift probability builds the intended branch mix."""
    crop = _get_training_crop(
        4,
        temporal_crop_shift_probability=0.25,
        temporal_crop_max_shift=2,
    )

    assert isinstance(crop, T.RandomChoice)
    assert len(crop.transforms) == 2
    assert isinstance(crop.transforms[0], T.RandomCrop)
    assert isinstance(crop.transforms[1], RandomTemporalCrop)
    assert crop.p == [0.75, 0.25]


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_get_transforms_rejects_invalid_shift_probability(probability):
    """Verify invalid probabilities fail before building the pipeline."""
    with pytest.raises(ValueError, match="temporal_crop_shift_probability"):
        get_transforms(
            is_train=True,
            crop_size=4,
            temporal_crop_shift_probability=probability,
            temporal_crop_max_shift=2,
        )


def test_get_transforms_shift_probability_branch():
    """Verify fractional temporal shift probability builds a working transform."""
    video = torch.randn(4, 8, 8)
    transform = get_transforms(
        is_train=True,
        crop_size=4,
        temporal_crop_shift_probability=0.5,
        temporal_crop_max_shift=2,
    )

    out_video, _ = transform(video)

    assert out_video.shape == (4, 4, 4)


def test_photometric_augmentation_treats_grayscale_clip_as_one_channel():
    augmentation = PhotometricAugmentationConfig.from_ranges(
        scale=0.0,
        shift=1.0,
        noise_std=0.0,
        apply_prob=1.0,
    )
    transform = DetectionTransform(
        T.Compose([T.Identity()]), photometric_augmentation=augmentation
    )
    video = torch.zeros(8, 4, 4)

    with torch.random.fork_rng():
        torch.manual_seed(0)
        out, _ = transform(video)

    assert_close(out, out[:1].expand_as(out))


def test_photometric_augmentation_samples_multichannel_clip_per_channel():
    augmentation = PhotometricAugmentationConfig.from_ranges(
        scale=0.0,
        shift=1.0,
        noise_std=0.0,
        apply_prob=1.0,
    )
    transform = DetectionTransform(
        T.Compose([T.Identity()]), photometric_augmentation=augmentation
    )
    video = torch.zeros(4, 8, 4, 4)

    with torch.random.fork_rng():
        torch.manual_seed(0)
        out, _ = transform(video)

    channel_values = out[0, :, 0, 0]
    assert torch.unique(channel_values).numel() > 1
    assert_close(out, channel_values[None, :, None, None].expand_as(out))


def test_evaluation_transforms_ignore_photometric_augmentation():
    augmentation = PhotometricAugmentationConfig.from_ranges(
        scale=0.0,
        shift=(1.0, 1.0),
        noise_std=0.0,
        apply_prob=1.0,
    )
    transform = get_transforms(
        is_train=False,
        crop_size=4,
        photometric_augmentation=augmentation,
    )
    video = torch.zeros(2, 1, 4, 4)

    out, _ = transform(video)

    assert_close(out, video)


@pytest.mark.parametrize("is_train", [True, False])
def test_transforms_shape(is_train):
    """Verify output shapes for train and val transforms."""
    T, H, W = 4, 100, 100
    crop_size = 64
    chunk = torch.randn(T, H, W)

    transforms = get_transforms(is_train=is_train, crop_size=crop_size)
    out, _ = transforms(chunk)
    assert out.shape == (T, crop_size, crop_size)


def test_transforms_multichannel():
    """Verify that transforms work with arbitrary leading dimensions (T, C, H, W)."""
    T, C, H, W = 4, 3, 100, 100
    crop_size = 50
    chunk = torch.randn(T, C, H, W)

    transforms = get_transforms(is_train=True, crop_size=crop_size)
    out, _ = transforms(chunk)

    assert out.shape == (T, C, crop_size, crop_size)
    assert torch.is_tensor(out)


def test_quantile_normalize_handles_outlier_without_nan():
    """Verify quantile normalization remains finite with extreme outliers."""
    # (T, H, W)
    T, H, W = 4, 100, 100
    # Create data with range 0-1000
    data = torch.linspace(0, 1000, steps=T * H * W).view(T, H, W)

    # Add outlier
    data[0, 0, 0] = 1_000_000

    normalization = get_normalization()
    out, _ = normalization(data)

    assert torch.is_tensor(out)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
