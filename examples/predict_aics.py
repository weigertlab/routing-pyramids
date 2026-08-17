"""Write CTC instance segmentations from an AICS routing-pyramid checkpoint."""

from pathlib import Path

import lightning as L
import torch

from routing_pyramids.data.temporal_datamodule import OMEZarrVideoDataModule
from routing_pyramids.prediction import CTCSegmentationPredictionWriter
from routing_pyramids.pyramid_flow_system import (
    PyramidFlowDecoder2d,
    PyramidFlowEncoder2d,
    PyramidFlowSegmentationTestConfig,
    PyramidFlowSystem,
)

DATA_DIR = Path("data/aics")
CHECKPOINT = Path(
    "outputs/aics/pyramid_flow_vae/"
    "dim_256_8x8_64-fg_1e-2-bg_5e-2-flow_5e-3-entropy_0-sparsity_5e-1/"
    "checkpoints/last.ckpt"
)
OUTPUT_DIR = Path("predictions/aics")


def main() -> None:
    """Run ground-truth-free prediction and write one label TIFF per frame."""
    torch.set_float32_matmul_precision("high")
    input_scale_factor = 0.5

    latent_dim = 64
    encoder_channels = (32, 64, 128, 256)
    encoder_strides = (2, 2, 2)
    encoder_blocks = (2, 2, 2, 2)
    decoder_channels = (latent_dim, *tuple(reversed(encoder_channels[:-1])))
    decoder_strides = tuple(reversed(encoder_strides))

    encoder = PyramidFlowEncoder2d(
        in_channels=1,
        channels=encoder_channels,
        strides=encoder_strides,
        down_blocks=encoder_blocks,
        norm="GROUP",
    )
    decoder = PyramidFlowDecoder2d(
        in_channels=latent_dim,
        out_channels=1,
        channels=decoder_channels,
        strides=decoder_strides,
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
    data = OMEZarrVideoDataModule(
        data_dir=str(DATA_DIR),
        predict_store_names=("20200323_06_medium_mip.ome.zarr",),
        batch_size=4,
        num_workers=4,
        crop_size=None,
        sequence_length=1,
        input_scale_factor=input_scale_factor,
        temporal_frame_stride=1,
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
