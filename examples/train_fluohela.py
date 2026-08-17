"""Train pyramid-flow spatial VAE on Fluo-N2DL-HeLa."""

from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from routing_pyramids.augmentation import PhotometricAugmentationConfig
from routing_pyramids.data.temporal_datamodule import VideoTemporalDataModule
from routing_pyramids.data.temporal_video_dataset import CTCVideoDataset
from routing_pyramids.pyramid_flow_system import (
    LossWeightSchedule,
    PyramidFlowDecoder2d,
    PyramidFlowEncoder2d,
    PyramidFlowSystem,
)

DATA_DIR = Path("data/Fluo-N2DL-HeLa")
OUTPUT_DIR = Path("outputs/hela")


def main() -> None:
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

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
    system = PyramidFlowSystem(
        encoder=encoder,
        decoder=decoder,
        foreground_latent_hidden_dim=256,
        background_latent_hidden_dim=32,
        background_latent_pool_stride=8,
        patch_size=8,
        latent_dim=latent_dim,
        foreground_sparsity_alpha=0.5,
        recon_loss_weight=LossWeightSchedule(1.0, 1.0, 0),
        foreground_kl_loss_weight=LossWeightSchedule(0, 1e-2, 10),
        background_kl_loss_weight=LossWeightSchedule(0, 5e-2, 10),
        flow_l2_loss_weight=LossWeightSchedule(0.0, 5e-3, 10),
        foreground_sparsity_loss_weight=LossWeightSchedule(0, 2e-1, 10),
        loss_type="l1",
        lr=1e-4,
        warmup_epochs=10,
        log_images_every_n_epochs=1,
        log_image_samples=3,
        log_decoder_ablation_every_n_epochs=10,
        log_decoder_ablation_max_batch_size=8,
    )
    system.compile()

    data = VideoTemporalDataModule(
        data_dir=str(DATA_DIR),
        dataset_class=CTCVideoDataset,
        train_split="test",
        val_split="train",
        test_split="train",
        batch_size=64,
        num_workers=4,
        crop_size=256,
        sequence_length=1,
        temporal_frame_stride=1,
        temporal_crop_shift_probability=0.0,
        train_repeat_factor=50,
        photometric_augmentation=PhotometricAugmentationConfig.from_ranges(
            scale=0.5,
            shift=0.2,
            gamma=(0.5, 2.0),
            noise_std=0.1,
            apply_prob=0.8,
        ),
        clip_quantile_low=0.05,
        clip_quantile_high=0.999,
        norm_quantile_low=0.50,
        norm_quantile_high=0.99,
    )

    callbacks: list[Callback] = [
        ModelCheckpoint(
            monitor="loss/val",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="{epoch}-{loss/val:.4f}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = TensorBoardLogger(
        save_dir=str(OUTPUT_DIR),
        name="pyramid_flow_vae",
        version="dim_256_8x8_64-fg_1e-2-bg_5e-2-flow_5e-3-entropy_0-sparsity_2e-1-200ep",
    )
    trainer = L.Trainer(
        # fast_dev_run=True,
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        strategy="auto",
        precision="bf16-mixed",
        max_epochs=200,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    trainer.fit(system, datamodule=data)


if __name__ == "__main__":
    main()
