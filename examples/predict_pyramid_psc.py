"""Write CTC segmentations from a PhC-C2DL-PSC routing-pyramid checkpoint."""

from pathlib import Path

import lightning as L
import torch

from routing_pyramids.data.temporal_datamodule import VideoTemporalDataModule
from routing_pyramids.data.temporal_video_dataset import CTCVideoDataset
from routing_pyramids.prediction import CTCSegmentationPredictionWriter
from routing_pyramids.pyramid_flow_system import (
    PyramidFlowDecoder2d,
    PyramidFlowEncoder2d,
    PyramidFlowSegmentationTestConfig,
    PyramidFlowSystem,
)

DATA_DIR = Path("data/PhC-C2DL-PSC")
CHECKPOINT = Path(
    "outputs/psc/pyramid_flow_vae/"
    "dim_256_8x8_64-fg_1e-2-bg_5e-2-flow_5e-3-entropy_0-sparsity_5e-1/"
    "checkpoints/last.ckpt"
)
OUTPUT_DIR = Path("predictions/psc")


def main() -> None:
    """Run ground-truth-free prediction and write one label TIFF per frame."""
    torch.set_float32_matmul_precision("high")
    input_scale_factor = 2.0

    latent_dim = 64
    encoder = PyramidFlowEncoder2d(
        in_channels=1,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        down_blocks=(2, 2, 2, 2),
        norm="GROUP",
    )
    decoder = PyramidFlowDecoder2d(
        in_channels=latent_dim,
        out_channels=1,
        channels=(latent_dim, 128, 64, 32),
        strides=(2, 2, 2),
        feature_stride=encoder.feature_stride,
        transport_predictor="conv",
        stage_blocks=(1, 4, 2, 1),
        normalize_latent_blend=False,
        dual_stream=False,
        value_modulation=False,
    )
    system = PyramidFlowSystem.load_from_checkpoint(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
        encoder=encoder,
        decoder=decoder,
        segmentation_test_config=PyramidFlowSegmentationTestConfig(
            center_threshold=0.5,
            pixel_mass_threshold=0.1,
            min_object_area=100,
            max_object_area=None,
            prediction_output_scale_factor=1.0 / input_scale_factor,
        ),
    )
    data = VideoTemporalDataModule(
        data_dir=str(DATA_DIR),
        dataset_class=CTCVideoDataset,
        train_split="test",
        val_split="train",
        test_split="train",
        batch_size=4,
        num_workers=4,
        crop_size=None,
        sequence_length=1,
        temporal_source_length=1,
        temporal_frame_stride=1,
        input_scale_factor=input_scale_factor,
        pin_memory=True,
        drop_last=False,
        clip_quantile_low=0.001,
        clip_quantile_high=0.999,
        norm_quantile_low=0.50,
        norm_quantile_high=0.99,
    )
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        callbacks=[CTCSegmentationPredictionWriter(OUTPUT_DIR)],
        logger=False,
    )
    trainer.predict(model=system, datamodule=data, return_predictions=False)


if __name__ == "__main__":
    main()
