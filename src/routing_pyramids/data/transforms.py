"""Data augmentation transforms using torchvision.transforms.v2."""

from collections.abc import Sequence
from typing import Any, cast

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torch import Tensor
from torchvision import tv_tensors

from routing_pyramids.augmentation import PhotometricAugmentationConfig


class DetectionTransform:
    """
    Transform pipeline for object detection (Video + Bounding Boxes).

    Parameters
    ----------
    transforms : T.Compose
        The composition of transforms to apply.
    """

    def __init__(
        self,
        transforms: T.Compose,
        photometric_augmentation: PhotometricAugmentationConfig | None = None,
    ):
        self.transforms = transforms
        self.photometric_augmentation = photometric_augmentation

    def _apply_photometric_augmentation(self, video: Tensor) -> Tensor:
        if self.photometric_augmentation is None:
            return video
        if video.ndim == 3:
            return self.photometric_augmentation(video[None, None])[0, 0]
        if video.ndim == 4:
            channel_first_clip = video.movedim(1, 0).unsqueeze(0)
            augmented = self.photometric_augmentation(channel_first_clip)[0]
            return augmented.movedim(0, 1)
        raise ValueError(
            f"video must have shape (T, H, W) or (T, C, H, W), got {tuple(video.shape)}"
        )

    def __call__(
        self, video: Tensor, boxes: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """
        Apply transforms to video and optional bounding boxes.

        Parameters
        ----------
        video : Tensor
            (T, C, H, W)
        boxes : Tensor, optional
            (N, 4) in XYXY format absolute coordinates.

        Returns
        -------
        video : Tensor
            Transformed video (T, C, H, W)
        boxes : Tensor, optional
            Transformed boxes (N, 4) in XYXY format.
        """
        # Wrap video in tv_tensors.Video to ensure correct transformation handling
        # structure is (T, C, H, W) or (C, H, W).
        # We assume input is (T, C, H, W).
        if video.ndim == 4:
            video_tv = tv_tensors.Video(video)
        else:
            video_tv = tv_tensors.Image(video)

        if boxes is not None and boxes.numel() > 0:
            # Boxes need to be wrapped. Format explicitly set to XYXY.
            # Canvas size (H, W) is required for some transforms (like flip/crop safety checks)
            H, W = video.shape[-2:]
            boxes_tv = cast(Any, tv_tensors.BoundingBoxes)(
                boxes,
                format="XYXY",
                canvas_size=(H, W),
            )
            out_video, out_boxes = self.transforms(video_tv, boxes_tv)
            return self._apply_photometric_augmentation(out_video), out_boxes
        else:
            out_video = self.transforms(video_tv)
            return self._apply_photometric_augmentation(out_video), boxes

    def apply_video_and_mask(
        self, video: Tensor, mask: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        """
        Apply transforms to a video tensor and an optional segmentation mask.

        Parameters
        ----------
        video : Tensor
            ``(T, C, H, W)`` or ``(T, H, W)``.
        mask : Tensor, optional
            Integer mask tensor with shape ``(T, H, W)``.
        """
        if video.ndim == 4:
            video_tv = tv_tensors.Video(video)
        else:
            video_tv = tv_tensors.Image(video)

        if mask is None:
            out_video = self.transforms(video_tv)
            return self._apply_photometric_augmentation(out_video), None

        mask_tv = tv_tensors.Mask(mask)
        out_video, out_mask = self.transforms(video_tv, mask_tv)
        return self._apply_photometric_augmentation(out_video), out_mask

    def apply_video_and_masks(
        self,
        video: Tensor,
        masks: Sequence[Tensor | None],
    ) -> tuple[Tensor, tuple[Tensor | None, ...]]:
        """
        Apply transforms to a video tensor and multiple segmentation masks.

        Parameters
        ----------
        video : Tensor
            ``(T, C, H, W)`` or ``(T, H, W)``.
        masks : sequence of Tensor or None
            Integer mask tensors with shape ``(T, H, W)``. ``None`` entries are
            preserved.
        """
        if video.ndim == 4:
            video_tv = tv_tensors.Video(video)
        else:
            video_tv = tv_tensors.Image(video)

        present_indices: list[int] = []
        present_masks: list[tv_tensors.Mask] = []
        for mask_index, mask in enumerate(masks):
            if mask is None:
                continue
            present_indices.append(mask_index)
            present_masks.append(tv_tensors.Mask(mask))

        if not present_masks:
            out_video = self.transforms(video_tv)
            return self._apply_photometric_augmentation(out_video), tuple(
                None for _ in masks
            )

        outputs = self.transforms(video_tv, *present_masks)
        if not isinstance(outputs, tuple):
            raise TypeError("Expected tuple output when transforming masks")
        out_video = outputs[0]
        out_present_masks = outputs[1:]
        if len(out_present_masks) != len(present_masks):
            raise RuntimeError(
                "Transformed mask count mismatch, got "
                f"{len(out_present_masks)} and expected {len(present_masks)}"
            )

        out_masks: list[Tensor | None] = [None] * len(masks)
        for mask_index, out_mask in zip(
            present_indices, out_present_masks, strict=True
        ):
            out_masks[mask_index] = out_mask
        return self._apply_photometric_augmentation(out_video), tuple(out_masks)


def get_normalization(
    clip_quantile_low: float = 0.001,
    norm_quantile_low: float = 0.50,
    norm_quantile_high: float = 0.99,
    clip_quantile_high: float = 0.999,
) -> DetectionTransform:
    """
    Get the normalization transform.

    Parameters
    ----------
    clip_quantile_low : float, optional
        Lower quantile for clipping
    clip_quantile_high : float, optional
        Higher quantile for clipping
    norm_quantile_low : float, optional
        Lower quantile for normalization scaling
    norm_quantile_high : float, optional
        Higher quantile for normalization scaling

    Returns
    -------
    DetectionTransform
        Normalization pipeline.
    """
    return DetectionTransform(
        T.Compose(
            [
                QuantileNormalize(
                    clip_quantile_low=clip_quantile_low,
                    norm_quantile_low=norm_quantile_low,
                    norm_quantile_high=norm_quantile_high,
                    clip_quantile_high=clip_quantile_high,
                )
            ]
        )
    )


class QuantileNormalize(torch.nn.Module):
    """
    Clip to specified quantile range, then scale based on quantile values.

    The default behavior clips to the 0.1th-99.9th percentile range and
    scales so that the 50th percentile maps to 0 and 99th percentile to 1.

    Parameters
    ----------
    clip_quantile_low : float, optional
        Lower quantile for clipping (default: 0.001)
    clip_quantile_high : float, optional
        Higher quantile for clipping (default: 0.999)
    norm_quantile_low : float, optional
        Lower quantile for normalization scaling (default: 0.50)
    norm_quantile_high : float, optional
        Higher quantile for normalization scaling (default: 0.99)
    """

    eps = 1e-8

    def __init__(
        self,
        clip_quantile_low: float = 0.001,
        norm_quantile_low: float = 0.50,
        norm_quantile_high: float = 0.99,
        clip_quantile_high: float = 0.999,
    ):
        super().__init__()
        self.clip_quantile_low = clip_quantile_low
        self.norm_quantile_low = norm_quantile_low
        self.norm_quantile_high = norm_quantile_high
        self.clip_quantile_high = clip_quantile_high

    def forward(
        self, x: Tensor, boxes: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        x_flat = x.flatten()
        # torch.quantile has a size limit
        # use strided sampling for large tensors
        n = x_flat.numel()
        if n > 1_000_000:
            stride = n // 1_000_000
            x_sample = x_flat[::stride]
        else:
            x_sample = x_flat
        p_low, p_norm_low, p_norm_high, p_high = torch.quantile(
            x_sample,
            torch.tensor(
                [
                    self.clip_quantile_low,
                    self.norm_quantile_low,
                    self.norm_quantile_high,
                    self.clip_quantile_high,
                ],
                device=x_sample.device,
            ),
        )
        x = x.clamp(min=p_low, max=p_high)
        x = (x - p_norm_low) / (p_norm_high - p_norm_low + self.eps)

        if boxes is not None:
            return x, boxes
        return x


class RandomTemporalCrop(torch.nn.Module):
    """
    Random crop with linear crop-window drift across a clip.

    Parameters
    ----------
    size : int | Sequence[int]
        Spatial size of the crop (height, width).
        The same size is used for both dimensions if an int is provided.
    max_shift : int | Sequence[int]
        Maximum shift for temporal drift.
    """

    def __init__(
        self,
        size: int | Sequence[int],
        *,
        max_shift: int | Sequence[int] = 0,
    ):
        super().__init__()
        self.size = self._normalize_size(size)
        self.max_shift = self._normalize_shift(max_shift)

    @staticmethod
    def _normalize_size(size: int | Sequence[int]) -> tuple[int, int]:
        if isinstance(size, int):
            size = (size, size)
        if len(size) != 2:
            raise ValueError(f"size must have length 2, got {size}")
        crop_h, crop_w = int(size[0]), int(size[1])
        if crop_h <= 0 or crop_w <= 0:
            raise ValueError(f"size must be positive, got {(crop_h, crop_w)}")
        return crop_h, crop_w

    @staticmethod
    def _normalize_shift(max_shift: int | Sequence[int]) -> tuple[int, int]:
        if isinstance(max_shift, int):
            max_shift = (max_shift, max_shift)
        if len(max_shift) != 2:
            raise ValueError(f"max_shift must have length 2, got {max_shift}")
        shift_h, shift_w = int(max_shift[0]), int(max_shift[1])
        if shift_h < 0 or shift_w < 0:
            raise ValueError(f"max_shift must be non-negative, got {max_shift}")
        return shift_h, shift_w

    @staticmethod
    def _randint(low: int, high: int, *, device: torch.device) -> int:
        if high <= low:
            return int(low)
        return int(torch.randint(low, high, size=(), device=device).item())

    @staticmethod
    def _has_bounding_boxes(inputs: tuple[Tensor, ...]) -> bool:
        return any(isinstance(input, tv_tensors.BoundingBoxes) for input in inputs)

    @staticmethod
    def _temporal_length(input: Tensor) -> int:
        if input.ndim < 3:
            return 1
        return int(input.shape[0])

    def _sample_static_location(
        self, *, height: int, width: int, device: torch.device
    ) -> tuple[int, int]:
        crop_h, crop_w = self.size
        max_top = height - crop_h
        max_left = width - crop_w
        if max_top < 0 or max_left < 0:
            raise ValueError(
                "Crop size must not exceed input spatial size, got "
                f"crop_size={self.size}, input_size={(height, width)}"
            )
        top = self._randint(0, max_top + 1, device=device)
        left = self._randint(0, max_left + 1, device=device)
        return top, left

    def _sample_temporal_locations(
        self, *, timesteps: int, height: int, width: int, device: torch.device
    ) -> tuple[list[int], list[int]]:
        top0, left0 = self._sample_static_location(
            height=height,
            width=width,
            device=device,
        )

        if timesteps == 1:
            return [top0], [left0]

        crop_h, crop_w = self.size
        shift_h, shift_w = self.max_shift
        max_top = height - crop_h
        max_left = width - crop_w
        top1_min = max(0, top0 - shift_h)
        top1_max = min(max_top, top0 + shift_h)
        left1_min = max(0, left0 - shift_w)
        left1_max = min(max_left, left0 + shift_w)
        top1 = self._randint(top1_min, top1_max + 1, device=device)
        left1 = self._randint(left1_min, left1_max + 1, device=device)

        alphas = torch.linspace(0.0, 1.0, timesteps, device=device)
        tops = torch.round(top0 + (top1 - top0) * alphas).to(dtype=torch.long)
        lefts = torch.round(left0 + (left1 - left0) * alphas).to(dtype=torch.long)
        return tops.cpu().tolist(), lefts.cpu().tolist()

    def _crop_per_timestep(
        self, input: Tensor, *, tops: list[int], lefts: list[int]
    ) -> Tensor:
        crop_h, crop_w = self.size
        if input.ndim < 3:
            return F.crop(input, tops[0], lefts[0], crop_h, crop_w)

        stacked = torch.stack(
            [
                F.crop(input[t], top, left, crop_h, crop_w)
                for t, (top, left) in enumerate(zip(tops, lefts, strict=True))
            ],
            dim=0,
        )
        if isinstance(input, tv_tensors.Video):
            return tv_tensors.Video(stacked)
        if isinstance(input, tv_tensors.Mask):
            return tv_tensors.Mask(stacked)
        return stacked

    def forward(self, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        if len(inputs) == 0:
            raise ValueError("RandomTemporalCrop requires at least one input")

        reference = inputs[0]
        height, width = int(reference.shape[-2]), int(reference.shape[-1])
        device = reference.device
        timesteps = self._temporal_length(reference)

        if self._has_bounding_boxes(inputs):
            raise ValueError(
                "RandomTemporalCrop does not support temporal crop drift with "
                "BoundingBoxes inputs."
            )

        tops, lefts = self._sample_temporal_locations(
            timesteps=timesteps,
            height=height,
            width=width,
            device=device,
        )
        outputs = tuple(
            self._crop_per_timestep(input, tops=tops, lefts=lefts) for input in inputs
        )

        if len(outputs) == 1:
            return outputs[0]
        return outputs


def _get_training_crop(
    crop_size: int,
    *,
    temporal_crop_shift_probability: float,
    temporal_crop_max_shift: int | Sequence[int],
) -> torch.nn.Module:
    if not 0.0 <= temporal_crop_shift_probability <= 1.0:
        raise ValueError(
            "temporal_crop_shift_probability must be in [0, 1], got "
            f"{temporal_crop_shift_probability}"
        )

    crop = T.RandomCrop((crop_size, crop_size))
    shift_h, shift_w = RandomTemporalCrop._normalize_shift(temporal_crop_max_shift)
    shift_configured = temporal_crop_shift_probability > 0.0 and (
        shift_h > 0 or shift_w > 0
    )
    if not shift_configured:
        return crop

    shifted_crop = RandomTemporalCrop(
        (crop_size, crop_size),
        max_shift=(shift_h, shift_w),
    )
    if temporal_crop_shift_probability == 1.0:
        return shifted_crop

    return T.RandomChoice(
        [crop, shifted_crop],
        p=[
            1.0 - temporal_crop_shift_probability,
            temporal_crop_shift_probability,
        ],
    )


def get_transforms(
    is_train: bool,
    crop_size: int | None,
    *,
    temporal_crop_shift_probability: float = 0.0,
    temporal_crop_max_shift: int | Sequence[int] = 0,
    photometric_augmentation: PhotometricAugmentationConfig | None = None,
) -> DetectionTransform:
    """
    Get composable transforms for data augmentation (augmentations only).

    Parameters
    ----------
    is_train : bool
        Whether to return training transforms (augmentations).
    crop_size : int or None
        Size of the spatial crop. ``None`` disables evaluation-time cropping.
    temporal_crop_shift_probability : float
        Probability of selecting a linearly drifting training crop.
    temporal_crop_max_shift : int or sequence of int
        Maximum end-frame crop displacement from the first-frame crop.
    photometric_augmentation : PhotometricAugmentationConfig, optional
        Training-only photometric augmentation applied per channel.

    Returns
    -------
    DetectionTransform
        Transform pipeline.
    """
    if crop_size is None:
        if is_train:
            raise ValueError(
                "crop_size=None is only supported for evaluation transforms"
            )
        return DetectionTransform(T.Compose([T.Identity()]))

    if is_train:
        transforms = T.Compose(
            [
                _get_training_crop(
                    crop_size,
                    temporal_crop_shift_probability=temporal_crop_shift_probability,
                    temporal_crop_max_shift=temporal_crop_max_shift,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                # Discrete 90-degree rotations (0, 90, 180, 270) matching MONAI RandRotate90
                T.RandomChoice(
                    [
                        T.Identity(),
                        T.RandomRotation(degrees=(90, 90)),
                        T.RandomRotation(degrees=(180, 180)),
                        T.RandomRotation(degrees=(270, 270)),
                    ]
                ),
            ]
        )
    else:
        transforms = T.Compose(
            [
                T.CenterCrop((crop_size, crop_size)),
            ]
        )

    return DetectionTransform(
        transforms,
        photometric_augmentation=photometric_augmentation if is_train else None,
    )
