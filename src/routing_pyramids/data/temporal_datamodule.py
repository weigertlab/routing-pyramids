"""Configurable temporal data modules for video and detection datasets."""

import warnings
from collections.abc import Sequence, Sized
from typing import cast

import lightning as L
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from routing_pyramids.augmentation import PhotometricAugmentationConfig
from routing_pyramids.data.sampling import fixed_shuffled_indices
from routing_pyramids.data.shared_frame_bank import validate_scale_factor
from routing_pyramids.data.temporal_index import build_deterministic_frame_indices
from routing_pyramids.data.temporal_video_dataset import (
    BBBC013VideoDataset,
    CTCVideoDataset,
    OMEZarrVideoDataset,
    TemporalVideoDataset,
    collate_temporal_video_batch,
)
from routing_pyramids.data.transforms import get_normalization, get_transforms


class BaseTemporalDataModule(L.LightningDataModule):
    """
    Base Lightning data module for temporal video-style datasets.

    This class centralizes common loader configuration, deterministic validation
    ordering, and temporal frame-index handling. Concrete subclasses are
    responsible for constructing train, validation, test, or prediction
    datasets.

    Parameters
    ----------
    data_dir
        Root directory containing the dataset files.
    batch_size
        Number of samples yielded by each dataloader batch.
    num_workers
        Number of worker processes used by each dataloader.
    crop_size
        Spatial crop size used by augmentation pipelines. ``None`` disables
        evaluation-time cropping and is only valid for non-fit stages.
    sequence_length
        Number of frames returned by each sampled clip.
    temporal_source_length
        Number of source frames available before frame subsampling. If ``None``,
        the dataset defaults to ``sequence_length``.
    temporal_frame_stride
        Stride between sampled frames in each clip.
    input_scale_factor
        Spatial output/input scale factor applied while loading frames.
    temporal_crop_shift_probability
        Probability that training crops drift linearly across a clip.
    temporal_crop_max_shift
        Maximum end-frame crop displacement from the first-frame crop, in pixels.
    train_repeat_factor
        Number of times to repeat the training dataset per epoch. A value of
        ``1`` keeps the natural training epoch length.
    photometric_augmentation
        Optional training-only per-channel photometric augmentation.
    pin_memory
        Whether dataloaders should pin host memory.
    drop_last
        Whether training and validation dataloaders should drop incomplete final
        batches by default.

    Raises
    ------
    ValueError
        If ``temporal_frame_stride`` or ``input_scale_factor`` is invalid.
    """

    def __init__(
        self,
        *,
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 4,
        crop_size: int | None = 256,
        sequence_length: int = 8,
        temporal_source_length: int | None = None,
        temporal_frame_stride: int = 1,
        input_scale_factor: float = 1.0,
        temporal_crop_shift_probability: float = 0.0,
        temporal_crop_max_shift: int | tuple[int, int] = 0,
        train_repeat_factor: int = 1,
        photometric_augmentation: PhotometricAugmentationConfig | None = None,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()
        if temporal_frame_stride < 1:
            raise ValueError(
                f"temporal_frame_stride must be >= 1, got {temporal_frame_stride}"
            )
        input_scale_factor = validate_scale_factor(input_scale_factor)
        if not isinstance(train_repeat_factor, int) or isinstance(
            train_repeat_factor, bool
        ):
            raise TypeError(
                f"train_repeat_factor must be an integer, got {train_repeat_factor!r}"
            )
        if train_repeat_factor < 1:
            raise ValueError(
                f"train_repeat_factor must be >= 1, got {train_repeat_factor}"
            )
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.sequence_length = sequence_length
        self.temporal_source_length = temporal_source_length
        self.temporal_frame_stride = int(temporal_frame_stride)
        self.input_scale_factor = input_scale_factor
        self.temporal_crop_shift_probability = float(temporal_crop_shift_probability)
        self.temporal_crop_max_shift = temporal_crop_max_shift
        self.train_repeat_factor = train_repeat_factor
        self.photometric_augmentation = photometric_augmentation
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self._mp_context = self._resolve_multiprocessing_context()
        self._val_indices: list[int] | None = None
        self._ordered_val_subset: Subset | None = None
        self._test_indices: list[int] | None = None
        self._ordered_test_subset: Subset | None = None

    def _resolve_multiprocessing_context(self) -> str | None:
        if self.num_workers <= 0:
            return None

        try:
            methods = torch.multiprocessing.get_all_start_methods()
        except RuntimeError:
            return None

        if "fork" in methods:
            return "fork"

        warnings.warn(
            "Fork start method not available; DataLoader workers may duplicate "
            "the dataset cache in memory."
        )
        return None

    def _apply_deterministic_validation_indices(self, dataset: Dataset[object]) -> None:
        if not hasattr(dataset, "set_fixed_frame_indices"):
            return

        if (
            self.temporal_frame_stride > 1
            or self.temporal_source_length is None
            or self.temporal_source_length <= self.sequence_length
        ):
            return

        dataset_sized = cast(Sized, dataset)
        frame_indices = build_deterministic_frame_indices(
            num_samples=len(dataset_sized),
            source_length=self.temporal_source_length,
            keep_length=self.sequence_length,
            seed=int(torch.initial_seed()),
        )
        cast(TemporalVideoDataset, dataset).set_fixed_frame_indices(frame_indices)

    def _build_validation_order(self, dataset: Dataset[object]) -> None:
        dataset_sized = cast(Sized, dataset)
        self._val_indices = fixed_shuffled_indices(len(dataset_sized))
        self._ordered_val_subset = Subset(dataset, self._val_indices)

    def _build_test_order(self, dataset: Dataset[object]) -> None:
        dataset_sized = cast(Sized, dataset)
        self._test_indices = fixed_shuffled_indices(len(dataset_sized))
        self._ordered_test_subset = Subset(dataset, self._test_indices)

    def _build_train_epoch_dataset(self, dataset: Dataset[object]) -> Dataset[object]:
        if self.train_repeat_factor == 1:
            return dataset
        return ConcatDataset([dataset] * self.train_repeat_factor)

    def _build_dataloader(
        self,
        dataset: Dataset[object],
        *,
        shuffle: bool,
        collate_fn,
        drop_last: bool | None = None,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
            multiprocessing_context=self._mp_context,
            drop_last=self.drop_last if drop_last is None else drop_last,
        )


class VideoTemporalDataModule(BaseTemporalDataModule):
    """
    Lightning data module for temporal video-only datasets.

    The module builds video datasets for training, validation, testing, and
    prediction using shared normalization settings. Training data receives random
    spatial augmentations, optionally including temporal crop drift, while
    validation and test data use deterministic center crops.

    Parameters
    ----------
    data_dir
        Root directory containing the dataset files.
    dataset_class
        CTC dataset implementation to instantiate.
    train_split
        Split name used for the training dataset.
    val_split
        Split name used for the validation dataset.
    test_split
        Split name used for testing. If ``None``, ``val_split`` is reused.
    batch_size
        Number of samples yielded by each dataloader batch.
    num_workers
        Number of worker processes used by each dataloader.
    crop_size
        Spatial crop size used by augmentation pipelines.
    sequence_length
        Number of frames returned by each sampled clip.
    temporal_source_length
        Number of source frames available before frame subsampling. If ``None``,
        the dataset defaults to ``sequence_length``.
    temporal_frame_stride
        Stride between sampled frames in each clip.
    input_scale_factor
        Spatial output/input scale factor applied while loading frames.
    temporal_crop_shift_probability
        Probability that training crops drift linearly across a clip.
    temporal_crop_max_shift
        Maximum end-frame crop displacement from the first-frame crop, in pixels.
    train_repeat_factor
        Number of times to repeat the training dataset per epoch. A value of
        ``1`` keeps the natural training epoch length.
    photometric_augmentation
        Optional training-only per-channel photometric augmentation.
    pin_memory
        Whether dataloaders should pin host memory.
    drop_last
        Whether training and validation dataloaders should drop incomplete final
        batches by default.
    clip_quantile_low
        Lower quantile used for intensity clipping.
    norm_quantile_low
        Lower quantile mapped to zero during intensity normalization.
    norm_quantile_high
        Upper quantile mapped to one during intensity normalization.
    clip_quantile_high
        Upper quantile used for intensity clipping.
    """

    def __init__(
        self,
        *,
        data_dir: str,
        dataset_class: type[CTCVideoDataset],
        train_split: str,
        val_split: str,
        test_split: str | None = None,
        batch_size: int = 4,
        num_workers: int = 4,
        crop_size: int | None = 256,
        sequence_length: int = 8,
        temporal_source_length: int | None = None,
        temporal_frame_stride: int = 1,
        input_scale_factor: float = 1.0,
        temporal_crop_shift_probability: float = 0.0,
        temporal_crop_max_shift: int | tuple[int, int] = 0,
        train_repeat_factor: int = 1,
        photometric_augmentation: PhotometricAugmentationConfig | None = None,
        pin_memory: bool = True,
        drop_last: bool = True,
        clip_quantile_low: float = 0.001,
        norm_quantile_low: float = 0.50,
        norm_quantile_high: float = 0.99,
        clip_quantile_high: float = 0.999,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            crop_size=crop_size,
            sequence_length=sequence_length,
            temporal_source_length=temporal_source_length,
            temporal_frame_stride=temporal_frame_stride,
            input_scale_factor=input_scale_factor,
            temporal_crop_shift_probability=temporal_crop_shift_probability,
            temporal_crop_max_shift=temporal_crop_max_shift,
            train_repeat_factor=train_repeat_factor,
            photometric_augmentation=photometric_augmentation,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        self.save_hyperparameters(ignore=["dataset_class"])
        self.dataset_class = dataset_class
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = val_split if test_split is None else test_split
        self.clip_quantile_low = clip_quantile_low
        self.norm_quantile_low = norm_quantile_low
        self.norm_quantile_high = norm_quantile_high
        self.clip_quantile_high = clip_quantile_high
        self.train_ds: CTCVideoDataset | None = None
        self.val_ds: CTCVideoDataset | None = None
        self.test_ds: CTCVideoDataset | None = None
        self.predict_ds: CTCVideoDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit", "test", "predict"):
            return

        normalization = get_normalization(
            clip_quantile_low=self.clip_quantile_low,
            norm_quantile_low=self.norm_quantile_low,
            norm_quantile_high=self.norm_quantile_high,
            clip_quantile_high=self.clip_quantile_high,
        )
        val_augment = get_transforms(is_train=False, crop_size=self.crop_size)
        if stage in (None, "fit"):
            train_augment = get_transforms(
                is_train=True,
                crop_size=self.crop_size,
                temporal_crop_shift_probability=self.temporal_crop_shift_probability,
                temporal_crop_max_shift=self.temporal_crop_max_shift,
                photometric_augmentation=self.photometric_augmentation,
            )

            self.train_ds = self.dataset_class(
                root_dir=self.data_dir,
                split=self.train_split,
                normalization=normalization,
                augmentations=train_augment,
                sequence_length=self.sequence_length,
                temporal_source_length=self.temporal_source_length,
                temporal_frame_stride=self.temporal_frame_stride,
                scale_factor=self.input_scale_factor,
            )
            self.val_ds = self.dataset_class(
                root_dir=self.data_dir,
                split=self.val_split,
                normalization=normalization,
                augmentations=val_augment,
                sequence_length=self.sequence_length,
                temporal_source_length=self.temporal_source_length,
                temporal_frame_stride=self.temporal_frame_stride,
                scale_factor=self.input_scale_factor,
            )

            self._apply_deterministic_validation_indices(self.val_ds)
            self._build_validation_order(self.val_ds)

        if stage in (None, "test"):
            self.test_ds = self.dataset_class(
                root_dir=self.data_dir,
                split=self.test_split,
                normalization=normalization,
                augmentations=val_augment,
                sequence_length=self.sequence_length,
                temporal_source_length=self.temporal_source_length,
                temporal_frame_stride=self.temporal_frame_stride,
                scale_factor=self.input_scale_factor,
            )
            self._apply_deterministic_validation_indices(self.test_ds)
            self._build_test_order(self.test_ds)

        if stage == "predict":
            self.predict_ds = self.dataset_class(
                root_dir=self.data_dir,
                split=self.test_split,
                normalization=normalization,
                augmentations=None,
                sequence_length=self.sequence_length,
                temporal_source_length=self.temporal_source_length,
                temporal_frame_stride=self.temporal_frame_stride,
                scale_factor=self.input_scale_factor,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError(
                "setup() must be called before requesting train_dataloader"
            )
        return self._build_dataloader(
            self._build_train_epoch_dataset(self.train_ds),
            shuffle=True,
            collate_fn=collate_temporal_video_batch,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError(
                "setup() must be called before requesting val_dataloader"
            )

        dataset = (
            self._ordered_val_subset
            if self._ordered_val_subset is not None
            else self.val_ds
        )
        return self._build_dataloader(
            dataset,
            shuffle=False,
            collate_fn=collate_temporal_video_batch,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError(
                "setup() must be called before requesting test_dataloader"
            )

        dataset = (
            self._ordered_test_subset
            if self._ordered_test_subset is not None
            else self.test_ds
        )
        return self._build_dataloader(
            dataset,
            shuffle=False,
            collate_fn=collate_temporal_video_batch,
        )

    def predict_dataloader(self) -> DataLoader:
        if self.predict_ds is None:
            raise RuntimeError(
                "setup() must be called before requesting predict_dataloader"
            )

        return self._build_dataloader(
            self.predict_ds,
            shuffle=False,
            collate_fn=collate_temporal_video_batch,
            drop_last=False,
        )


class BBBC013VideoDataModule(BaseTemporalDataModule):
    """Lightning data module for static BBBC013 well images.

    Each TIFF file is exposed as an independent single-frame sample. Splits are
    selected by plate column across all rows.

    Parameters
    ----------
    data_dir
        Directory containing converted TIFF files named like ``A1.tif``.
    train_columns
        Plate columns assigned to training.
    eval_columns
        Plate columns assigned to validation, testing, and prediction.
    batch_size
        Number of samples yielded by each dataloader batch.
    num_workers
        Number of worker processes used by each dataloader.
    crop_size
        Spatial crop size used by training and evaluation transforms.
    input_scale_factor
        Spatial output/input scale factor applied while loading images.
    train_repeat_factor
        Number of times to repeat the training dataset per epoch.
    photometric_augmentation
        Optional training-only per-channel photometric augmentation.
    pin_memory
        Whether dataloaders should pin host memory.
    drop_last
        Whether training, validation, and test dataloaders drop incomplete final
        batches by default.
    clip_quantile_low
        Lower quantile used for intensity clipping.
    norm_quantile_low
        Lower quantile mapped to zero during intensity normalization.
    norm_quantile_high
        Upper quantile mapped to one during intensity normalization.
    clip_quantile_high
        Upper quantile used for intensity clipping.
    """

    def __init__(
        self,
        *,
        data_dir: str,
        train_columns: Sequence[int] = tuple(range(2, 12)),
        eval_columns: Sequence[int] = (1, 12),
        batch_size: int = 4,
        num_workers: int = 4,
        crop_size: int | None = 256,
        input_scale_factor: float = 1.0,
        train_repeat_factor: int = 1,
        photometric_augmentation: PhotometricAugmentationConfig | None = None,
        pin_memory: bool = True,
        drop_last: bool = True,
        clip_quantile_low: float = 0.001,
        norm_quantile_low: float = 0.50,
        norm_quantile_high: float = 0.99,
        clip_quantile_high: float = 0.999,
    ):
        validated_train_columns = self._validate_columns(
            train_columns,
            argument_name="train_columns",
        )
        validated_eval_columns = self._validate_columns(
            eval_columns,
            argument_name="eval_columns",
        )
        overlap = sorted(
            set(validated_train_columns).intersection(validated_eval_columns)
        )
        if overlap:
            raise ValueError(
                f"train_columns and eval_columns must be disjoint, overlap={overlap}"
            )

        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            crop_size=crop_size,
            sequence_length=1,
            temporal_source_length=None,
            temporal_frame_stride=1,
            input_scale_factor=input_scale_factor,
            temporal_crop_shift_probability=0.0,
            temporal_crop_max_shift=0,
            train_repeat_factor=train_repeat_factor,
            photometric_augmentation=photometric_augmentation,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        self.save_hyperparameters()
        self.train_columns = validated_train_columns
        self.eval_columns = validated_eval_columns
        self.clip_quantile_low = clip_quantile_low
        self.norm_quantile_low = norm_quantile_low
        self.norm_quantile_high = norm_quantile_high
        self.clip_quantile_high = clip_quantile_high

        self.train_ds: BBBC013VideoDataset | None = None
        self.val_ds: BBBC013VideoDataset | None = None
        self.test_ds: BBBC013VideoDataset | None = None
        self.predict_ds: BBBC013VideoDataset | None = None

    @staticmethod
    def _validate_columns(
        columns: Sequence[int],
        *,
        argument_name: str,
    ) -> tuple[int, ...]:
        validated_columns = tuple(columns)
        if not validated_columns:
            raise ValueError(f"{argument_name} must contain at least one column")
        if any(
            not isinstance(column, int)
            or isinstance(column, bool)
            or not 1 <= column <= 12
            for column in validated_columns
        ):
            raise ValueError(
                f"{argument_name} entries must be integer columns between 1 and 12, "
                f"got {validated_columns!r}"
            )
        if len(set(validated_columns)) != len(validated_columns):
            raise ValueError(f"{argument_name} must contain unique columns")
        return validated_columns

    def _build_dataset(
        self,
        *,
        columns: tuple[int, ...],
        normalization,
        augmentations,
    ) -> BBBC013VideoDataset:
        return BBBC013VideoDataset(
            root_dir=self.data_dir,
            first_characters=tuple("ABCDEFGH"),
            columns=columns,
            normalization=normalization,
            augmentations=augmentations,
            scale_factor=self.input_scale_factor,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit", "test", "predict"):
            return

        normalization = get_normalization(
            clip_quantile_low=self.clip_quantile_low,
            norm_quantile_low=self.norm_quantile_low,
            norm_quantile_high=self.norm_quantile_high,
            clip_quantile_high=self.clip_quantile_high,
        )
        eval_augment = get_transforms(is_train=False, crop_size=self.crop_size)
        if stage in (None, "fit"):
            train_augment = get_transforms(
                is_train=True,
                crop_size=self.crop_size,
                photometric_augmentation=self.photometric_augmentation,
            )
            self.train_ds = self._build_dataset(
                columns=self.train_columns,
                normalization=normalization,
                augmentations=train_augment,
            )
            self.val_ds = self._build_dataset(
                columns=self.eval_columns,
                normalization=normalization,
                augmentations=eval_augment,
            )
            self._build_validation_order(self.val_ds)

        if stage in (None, "test"):
            self.test_ds = self._build_dataset(
                columns=self.eval_columns,
                normalization=normalization,
                augmentations=eval_augment,
            )
            self._build_test_order(self.test_ds)

        if stage == "predict":
            self.predict_ds = self._build_dataset(
                columns=self.eval_columns,
                normalization=normalization,
                augmentations=None,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError(
                "setup() must be called before requesting train_dataloader"
            )
        return self._build_dataloader(
            self._build_train_epoch_dataset(self.train_ds),
            shuffle=True,
            collate_fn=collate_temporal_video_batch,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError(
                "setup() must be called before requesting val_dataloader"
            )
        dataset = (
            self._ordered_val_subset
            if self._ordered_val_subset is not None
            else self.val_ds
        )
        return self._build_dataloader(
            dataset,
            shuffle=False,
            collate_fn=collate_temporal_video_batch,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError(
                "setup() must be called before requesting test_dataloader"
            )
        dataset = (
            self._ordered_test_subset
            if self._ordered_test_subset is not None
            else self.test_ds
        )
        return self._build_dataloader(
            dataset,
            shuffle=False,
            collate_fn=collate_temporal_video_batch,
        )

    def predict_dataloader(self) -> DataLoader:
        if self.predict_ds is None:
            raise RuntimeError(
                "setup() must be called before requesting predict_dataloader"
            )
        return self._build_dataloader(
            self.predict_ds,
            shuffle=False,
            collate_fn=collate_temporal_video_batch,
            drop_last=False,
        )


class OMEZarrVideoDataModule(BaseTemporalDataModule):
    """
    Lightning data module for configurable OME-Zarr data splits.

    Parameters
    ----------
    data_dir
        Directory containing the OME-Zarr stores.
    train_store_names
        Unique ``.ome.zarr`` directory names assigned to training.
    val_store_names
        Unique ``.ome.zarr`` directory names assigned to validation.
    predict_store_names
        Unique ``.ome.zarr`` directory names used for prediction. Prediction
        samples are loaded without spatial augmentation or ground truth.
    batch_size
        Number of temporal samples per batch.
    num_workers
        Number of worker processes used by each dataloader.
    crop_size
        Spatial crop size for training and validation samples.
    sequence_length
        Number of frames returned by each sample.
    temporal_source_length
        Number of source frames from which output frames are selected.
    temporal_frame_stride
        Stride between selected source frames.
    input_scale_factor
        Spatial output/input scale factor applied while loading videos.
    temporal_crop_shift_probability
        Probability that a training crop drifts across a temporal sample.
    temporal_crop_max_shift
        Maximum end-frame crop displacement from the first frame.
    train_repeat_factor
        Number of times the natural training dataset is repeated per epoch.
    photometric_augmentation
        Optional training-only per-channel photometric augmentation.
    pin_memory
        Whether dataloaders pin host memory.
    drop_last
        Whether dataloaders drop incomplete final batches.
    clip_quantile_low
        Lower quantile used for intensity clipping.
    norm_quantile_low
        Lower quantile mapped to zero during intensity normalization.
    norm_quantile_high
        Upper quantile mapped to one during intensity normalization.
    clip_quantile_high
        Upper quantile used for intensity clipping.
    """

    def __init__(
        self,
        *,
        data_dir: str,
        train_store_names: Sequence[str] = (),
        val_store_names: Sequence[str] = (),
        predict_store_names: Sequence[str] = (),
        batch_size: int = 4,
        num_workers: int = 4,
        crop_size: int | None = 256,
        sequence_length: int = 8,
        temporal_source_length: int | None = None,
        temporal_frame_stride: int = 1,
        input_scale_factor: float = 1.0,
        temporal_crop_shift_probability: float = 0.0,
        temporal_crop_max_shift: int | tuple[int, int] = 0,
        train_repeat_factor: int = 1,
        photometric_augmentation: PhotometricAugmentationConfig | None = None,
        pin_memory: bool = True,
        drop_last: bool = True,
        clip_quantile_low: float = 0.001,
        norm_quantile_low: float = 0.50,
        norm_quantile_high: float = 0.99,
        clip_quantile_high: float = 0.999,
    ):
        train_names = self._validate_store_names(
            train_store_names,
            argument_name="train_store_names",
            allow_empty=True,
        )
        val_names = self._validate_store_names(
            val_store_names,
            argument_name="val_store_names",
            allow_empty=True,
        )
        predict_names = self._validate_store_names(
            predict_store_names,
            argument_name="predict_store_names",
            allow_empty=True,
        )
        if bool(train_names) != bool(val_names):
            missing_argument = "val_store_names" if train_names else "train_store_names"
            raise ValueError(
                f"{missing_argument} must contain at least one store when configuring fit data"
            )
        if not train_names and not predict_names:
            raise ValueError(
                "configure train_store_names and val_store_names, "
                "predict_store_names, or both"
            )
        overlap = sorted(set(train_names).intersection(val_names))
        if overlap:
            raise ValueError(
                "train_store_names and val_store_names must be disjoint, "
                f"overlap={overlap}"
            )
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            crop_size=crop_size,
            sequence_length=sequence_length,
            temporal_source_length=temporal_source_length,
            temporal_frame_stride=temporal_frame_stride,
            input_scale_factor=input_scale_factor,
            temporal_crop_shift_probability=temporal_crop_shift_probability,
            temporal_crop_max_shift=temporal_crop_max_shift,
            train_repeat_factor=train_repeat_factor,
            photometric_augmentation=photometric_augmentation,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        self.save_hyperparameters()
        self.train_store_names = train_names
        self.val_store_names = val_names
        self.predict_store_names = predict_names
        self.clip_quantile_low = clip_quantile_low
        self.norm_quantile_low = norm_quantile_low
        self.norm_quantile_high = norm_quantile_high
        self.clip_quantile_high = clip_quantile_high
        self.train_ds: OMEZarrVideoDataset | None = None
        self.val_ds: OMEZarrVideoDataset | None = None
        self.predict_ds: OMEZarrVideoDataset | None = None

    @staticmethod
    def _validate_store_names(
        store_names: Sequence[str],
        *,
        argument_name: str,
        allow_empty: bool = False,
    ) -> tuple[str, ...]:
        names = tuple(store_names)
        if not names and not allow_empty:
            raise ValueError(f"{argument_name} must contain at least one store")
        if len(set(names)) != len(names):
            raise ValueError(f"{argument_name} must contain unique store names")
        return names

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit", "predict"):
            return

        normalization = get_normalization(
            clip_quantile_low=self.clip_quantile_low,
            norm_quantile_low=self.norm_quantile_low,
            norm_quantile_high=self.norm_quantile_high,
            clip_quantile_high=self.clip_quantile_high,
        )
        if stage in (None, "fit"):
            if not self.train_store_names:
                if stage == "fit":
                    raise RuntimeError(
                        "train_store_names and val_store_names are required for setup('fit')"
                    )
            else:
                train_augment = get_transforms(
                    is_train=True,
                    crop_size=self.crop_size,
                    temporal_crop_shift_probability=self.temporal_crop_shift_probability,
                    temporal_crop_max_shift=self.temporal_crop_max_shift,
                    photometric_augmentation=self.photometric_augmentation,
                )
                val_augment = get_transforms(
                    is_train=False,
                    crop_size=self.crop_size,
                )
                self.train_ds = OMEZarrVideoDataset(
                    root_dir=self.data_dir,
                    store_names=self.train_store_names,
                    normalization=normalization,
                    augmentations=train_augment,
                    sequence_length=self.sequence_length,
                    temporal_source_length=self.temporal_source_length,
                    temporal_frame_stride=self.temporal_frame_stride,
                    scale_factor=self.input_scale_factor,
                )
                self.val_ds = OMEZarrVideoDataset(
                    root_dir=self.data_dir,
                    store_names=self.val_store_names,
                    normalization=normalization,
                    augmentations=val_augment,
                    sequence_length=self.sequence_length,
                    temporal_source_length=self.temporal_source_length,
                    temporal_frame_stride=self.temporal_frame_stride,
                    scale_factor=self.input_scale_factor,
                )
                self._apply_deterministic_validation_indices(self.val_ds)
                self._build_validation_order(self.val_ds)

        if stage in (None, "predict"):
            if not self.predict_store_names:
                if stage == "predict":
                    raise RuntimeError(
                        "predict_store_names is required for setup('predict')"
                    )
            else:
                self.predict_ds = OMEZarrVideoDataset(
                    root_dir=self.data_dir,
                    store_names=self.predict_store_names,
                    normalization=normalization,
                    augmentations=None,
                    sequence_length=self.sequence_length,
                    temporal_source_length=self.temporal_source_length,
                    temporal_frame_stride=self.temporal_frame_stride,
                    scale_factor=self.input_scale_factor,
                )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError(
                "setup() must be called before requesting train_dataloader"
            )
        return self._build_dataloader(
            self._build_train_epoch_dataset(self.train_ds),
            shuffle=True,
            collate_fn=collate_temporal_video_batch,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError(
                "setup() must be called before requesting val_dataloader"
            )
        dataset = (
            self._ordered_val_subset
            if self._ordered_val_subset is not None
            else self.val_ds
        )
        return self._build_dataloader(
            dataset,
            shuffle=False,
            collate_fn=collate_temporal_video_batch,
        )

    def predict_dataloader(self) -> DataLoader:
        if self.predict_ds is None:
            raise RuntimeError(
                "setup() must be called before requesting predict_dataloader"
            )
        return self._build_dataloader(
            self.predict_ds,
            shuffle=False,
            collate_fn=collate_temporal_video_batch,
            drop_last=False,
        )


__all__ = [
    "BBBC013VideoDataModule",
    "BaseTemporalDataModule",
    "OMEZarrVideoDataModule",
    "VideoTemporalDataModule",
]
