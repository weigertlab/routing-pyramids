"""Shared image augmentation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PhotometricAugmentationConfig:
    """Per-sample, per-channel photometric jitter for image-like tensors."""

    scale: tuple[float, float]
    shift: tuple[float, float]
    gamma: tuple[float, float]
    noise_std: float
    apply_prob: float

    @classmethod
    def from_ranges(
        cls,
        *,
        scale: float | tuple[float, float] = 0.1,
        shift: float | tuple[float, float] = 0.05,
        gamma: float | tuple[float, float] = (1.0, 1.0),
        noise_std: float = 0.0,
        apply_prob: float = 0.5,
    ) -> PhotometricAugmentationConfig:
        """Create an augmentation from scalar magnitudes or explicit ranges."""
        scale_range = cls._scale_range(scale)
        shift_range = cls._symmetric_range("photometric_shift", shift)
        gamma_range = cls._positive_range("photometric_gamma", gamma)
        if scale_range[0] < 0.0:
            raise ValueError(
                f"photometric_scale lower bound must be non-negative, got {scale_range}"
            )
        if noise_std < 0.0:
            raise ValueError(
                f"photometric_noise_std must be non-negative, got {noise_std}"
            )
        if not 0.0 <= apply_prob <= 1.0:
            raise ValueError(
                f"photometric_apply_prob must be in [0, 1], got {apply_prob}"
            )
        return cls(
            scale=scale_range,
            shift=shift_range,
            gamma=gamma_range,
            noise_std=float(noise_std),
            apply_prob=float(apply_prob),
        )

    def __call__(self, image: Tensor) -> Tensor:
        """Apply jitter to an image tensor shaped ``(batch, channels, ...)``."""
        if image.ndim < 3:
            raise ValueError(
                "image must have shape (batch, channels, ...), got "
                f"{tuple(image.shape)}"
            )
        if self.apply_prob == 0.0:
            return image
        parameter_shape = image.shape[:2] + (1,) * (image.ndim - 2)
        out = image
        if self.gamma != (1.0, 1.0):
            gamma_active = self._apply_mask(
                parameter_shape=parameter_shape, reference=image
            )
            gamma = self._sample_range(
                self.gamma, parameter_shape=parameter_shape, reference=image
            )
            out = torch.where(
                gamma_active,
                self._adjust_gamma_preserve_range(out, gamma=gamma),
                out,
            )
        if self.scale != (1.0, 1.0):
            scale_active = self._apply_mask(
                parameter_shape=parameter_shape, reference=image
            )
            scale = self._sample_range(
                self.scale, parameter_shape=parameter_shape, reference=image
            )
            out = out * torch.where(
                scale_active,
                scale,
                torch.ones(parameter_shape, device=image.device, dtype=image.dtype),
            )
        if self.shift != (0.0, 0.0):
            shift_active = self._apply_mask(
                parameter_shape=parameter_shape, reference=image
            )
            shift = self._sample_range(
                self.shift, parameter_shape=parameter_shape, reference=image
            )
            out = out + torch.where(
                shift_active,
                shift,
                torch.zeros(parameter_shape, device=image.device, dtype=image.dtype),
            )
        if self.noise_std > 0.0:
            noise_active = self._apply_mask(
                parameter_shape=parameter_shape, reference=image
            )
            noise_std = torch.empty(
                parameter_shape, device=image.device, dtype=image.dtype
            ).uniform_(0.0, self.noise_std)
            noise = torch.randn_like(out) * noise_std
            out = out + torch.where(noise_active, noise, torch.zeros_like(noise))
        return out

    @classmethod
    def _scale_range(cls, value: float | tuple[float, float]) -> tuple[float, float]:
        if isinstance(value, tuple):
            return cls._ordered_range("photometric_scale", value)
        magnitude = float(value)
        if magnitude < 0.0:
            raise ValueError(f"photometric_scale must be non-negative, got {value}")
        return 1.0 - magnitude, 1.0 + magnitude

    @classmethod
    def _symmetric_range(
        cls, name: str, value: float | tuple[float, float]
    ) -> tuple[float, float]:
        if isinstance(value, tuple):
            return cls._ordered_range(name, value)
        magnitude = float(value)
        if magnitude < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")
        return -magnitude, magnitude

    @classmethod
    def _positive_range(
        cls, name: str, value: float | tuple[float, float]
    ) -> tuple[float, float]:
        if isinstance(value, tuple):
            low, high = cls._ordered_range(name, value)
        else:
            low = high = float(value)
        if low < 0.0:
            raise ValueError(
                f"{name} lower bound must be non-negative, got {(low, high)}"
            )
        return low, high

    @staticmethod
    def _ordered_range(name: str, value: tuple[float, float]) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError(f"{name} range must contain two values, got {value}")
        low, high = float(value[0]), float(value[1])
        if low > high:
            raise ValueError(f"{name} range must be ordered, got {value}")
        return low, high

    @staticmethod
    def _sample_range(
        bounds: tuple[float, float],
        *,
        parameter_shape: tuple[int, ...],
        reference: Tensor,
    ) -> Tensor:
        low, high = bounds
        if low == high:
            return torch.full(
                parameter_shape, low, device=reference.device, dtype=reference.dtype
            )
        return torch.empty(
            parameter_shape, device=reference.device, dtype=reference.dtype
        ).uniform_(low, high)

    @staticmethod
    def _adjust_gamma_preserve_range(image: Tensor, *, gamma: Tensor) -> Tensor:
        parameter_shape = image.shape[:2] + (1,) * (image.ndim - 2)
        flat = image.flatten(start_dim=2)
        sample_min = flat.amin(dim=2).view(parameter_shape)
        sample_max = flat.amax(dim=2).view(parameter_shape)
        sample_range = sample_max - sample_min
        nonconstant = sample_range > 0.0
        denom = torch.where(
            nonconstant,
            sample_range,
            torch.ones(parameter_shape, device=image.device, dtype=image.dtype),
        )
        normalized = ((image - sample_min) / denom).clamp(0.0, 1.0)
        adjusted = normalized.pow(gamma) * sample_range + sample_min
        return torch.where(nonconstant, adjusted, image)

    def _apply_mask(
        self, *, parameter_shape: tuple[int, ...], reference: Tensor
    ) -> Tensor:
        return (
            torch.rand(parameter_shape, device=reference.device, dtype=reference.dtype)
            < self.apply_prob
        )


__all__ = ["PhotometricAugmentationConfig"]
