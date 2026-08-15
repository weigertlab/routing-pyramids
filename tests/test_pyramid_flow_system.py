from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Literal, cast

import lightning
import pytest
import torch
from einops import rearrange

import routing_pyramids.pyramid_flow_system as pfs
from routing_pyramids.pyramid_flow_system import (
    LossWeightSchedule,
    PointwiseMLP2d,
    PyramidFlowDecoder2d,
    PyramidFlowEncoder2d,
    PyramidFlowEncoderOutput,
    PyramidFlowSegmentationTestConfig,
    PyramidFlowSystem,
    PyramidTransport,
    fine_to_coarse_lookup,
)
from routing_pyramids.types import TemporalBatch

from ._stubs import TrainerStub as _TrainerStub

LATENT_DIM = 10
FOREGROUND_LATENT_HIDDEN_DIM = 32
BACKGROUND_LATENT_HIDDEN_DIM = 5


def _tiny_encoder(*, image_channels: int = 1) -> PyramidFlowEncoder2d:
    return PyramidFlowEncoder2d(
        in_channels=image_channels,
        channels=(4, 8, 12, 16),
        strides=(2, 2, 2),
        down_blocks=(1, 1, 1, 1),
        norm=("GROUP", {"num_groups": 2}),
    )


def _tiny_decoder(
    *,
    image_channels: int = 1,
    stage_blocks: tuple[int, ...] | None = None,
    transport_predictor: Literal["attention", "conv"] = "attention",
    dual_stream: bool = False,
    value_modulation: bool = False,
) -> PyramidFlowDecoder2d:
    return PyramidFlowDecoder2d(
        in_channels=LATENT_DIM,
        out_channels=image_channels,
        channels=(LATENT_DIM, 12, 8, 4),
        strides=(2, 2, 2),
        feature_stride=8,
        stage_blocks=stage_blocks,
        transport_predictor=transport_predictor,
        dual_stream=dual_stream,
        value_modulation=value_modulation,
    )


def _tiny_system(
    *,
    image_channels: int = 1,
    stage_blocks: tuple[int, ...] | None = None,
    warmup_epochs: int = 0,
    transport_predictor: Literal["attention", "conv"] = "attention",
    dual_stream: bool = False,
    value_modulation: bool = False,
    foreground_latent_hidden_dim: int = FOREGROUND_LATENT_HIDDEN_DIM,
    background_latent_hidden_dim: int = BACKGROUND_LATENT_HIDDEN_DIM,
    background_latent_pool_stride: int = 1,
    segmentation_test_config: PyramidFlowSegmentationTestConfig | None = None,
    log_decoder_ablation_every_n_epochs: int = 0,
    log_decoder_ablation_max_batch_size: int = 8,
) -> PyramidFlowSystem:
    return PyramidFlowSystem(
        encoder=_tiny_encoder(image_channels=image_channels),
        decoder=_tiny_decoder(
            image_channels=image_channels,
            stage_blocks=stage_blocks,
            transport_predictor=transport_predictor,
            dual_stream=dual_stream,
            value_modulation=value_modulation,
        ),
        foreground_latent_hidden_dim=foreground_latent_hidden_dim,
        background_latent_hidden_dim=background_latent_hidden_dim,
        background_latent_pool_stride=background_latent_pool_stride,
        patch_size=8,
        latent_dim=LATENT_DIM,
        latent_head_norm=("GROUP", {"num_groups": 2}),
        foreground_sparsity_alpha=0.5,
        foreground_sparsity_loss_weight=LossWeightSchedule(0.0, 1e-3, 10),
        dense_assignment_max_elements=100_000,
        warmup_epochs=warmup_epochs,
        log_images_every_n_epochs=0,
        log_decoder_ablation_every_n_epochs=log_decoder_ablation_every_n_epochs,
        log_decoder_ablation_max_batch_size=log_decoder_ablation_max_batch_size,
        segmentation_test_config=segmentation_test_config,
    )


def _one_stage_system() -> PyramidFlowSystem:
    return PyramidFlowSystem(
        encoder=PyramidFlowEncoder2d(
            in_channels=1,
            channels=(2,),
            strides=(),
            norm=("GROUP", {"num_groups": 1}),
        ),
        decoder=PyramidFlowDecoder2d(
            in_channels=2,
            out_channels=1,
            channels=(2,),
            strides=(),
            feature_stride=1,
            stage_blocks=(1,),
        ),
        foreground_latent_hidden_dim=4,
        background_latent_hidden_dim=1,
        patch_size=1,
        latent_dim=2,
        latent_head_norm=("GROUP", {"num_groups": 1}),
        log_images_every_n_epochs=0,
    )


def _decoder_stream_inputs(
    *,
    batch_size: int = 2,
    channels: int = LATENT_DIM,
    grid_hw: tuple[int, int] = (2, 2),
    background_channels: int | None = None,
    background_grid_hw: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    foreground = torch.randn(batch_size, channels, *grid_hw)
    background = torch.randn(
        batch_size,
        channels if background_channels is None else background_channels,
        *(grid_hw if background_grid_hw is None else background_grid_hw),
    )
    foreground_presence = torch.rand(batch_size, 1, *grid_hw)
    return foreground, background, foreground_presence


def _single_frame_batch(image: torch.Tensor) -> TemporalBatch:
    batch_size = int(image.shape[0])
    return {
        "video": image.unsqueeze(1),
        "frame_indices": torch.zeros(batch_size, 1, dtype=torch.long),
        "source_length": 1,
    }


def _attach_trainer(
    system: PyramidFlowSystem,
    *,
    estimated_stepping_batches: int | None = 400,
    max_epochs: float | None = 100,
) -> None:
    system._trainer = _TrainerStub(
        experiment=object(),
        global_step=0,
        estimated_stepping_batches=estimated_stepping_batches,
        max_epochs=max_epochs,
    )


def test_lightning_checkpoint_round_trip(tmp_path: Path) -> None:
    system = _tiny_system(transport_predictor="conv")
    trainer = lightning.Trainer(
        accelerator="cpu", logger=False, enable_checkpointing=False, max_epochs=0
    )
    trainer.strategy.connect(system)
    checkpoint = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint)

    loaded = PyramidFlowSystem.load_from_checkpoint(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        encoder=_tiny_encoder(),
        decoder=_tiny_decoder(transport_predictor="conv"),
    )
    loaded.eval()
    output = loaded(torch.randn(1, 1, 32, 32))
    assert output["recon"].shape == (1, 1, 32, 32)


def test_image_logging_supports_two_channel_fluorescence() -> None:
    class ImageExperiment:
        def __init__(self) -> None:
            self.images: list[torch.Tensor] = []

        def add_image(self, tag: str, image: torch.Tensor, global_step: int) -> None:
            self.images.append(image)

    model = _tiny_system(image_channels=2)
    model.log_images_every_n_epochs = 1
    image = torch.rand(2, 2, 16, 16)
    output = model(image)
    experiment = ImageExperiment()
    model._trainer = _TrainerStub(
        experiment=experiment,
        global_step=0,
        current_epoch=0,
    )

    model._log_images(image=image, output=output, stage="train", batch_idx=0)

    assert len(experiment.images) == 1
    assert experiment.images[0].shape[0] == 3


def _gather_test_transport(
    *,
    batch_size: int,
    fine_hw: tuple[int, int],
    coarse_hw: tuple[int, int],
    pixel_stride: int = 1,
) -> PyramidTransport:
    fine_h, fine_w = fine_hw
    lookup = fine_to_coarse_lookup(
        fine_h=fine_h,
        fine_w=fine_w,
        coarse_h=coarse_hw[0],
        coarse_w=coarse_hw[1],
        device=torch.device("cpu"),
    )
    logits = torch.linspace(
        -1.25,
        1.25,
        steps=batch_size * 9 * fine_h * fine_w,
        dtype=torch.float32,
    ).view(batch_size, 9, fine_h, fine_w)
    valid = lookup.valid_edges.T.reshape(1, 9, fine_h, fine_w)
    probs = torch.softmax(
        logits.masked_fill(~valid, -torch.finfo(torch.float32).max),
        dim=1,
    ) * valid.to(dtype=torch.float32)
    return PyramidTransport(
        probs=probs,
        logits=logits,
        expected_offset=pfs._expected_local_offset(probs, lookup=lookup),
        lookup=lookup,
        pixel_stride=pixel_stride,
    )


def _center_one_hot_transport(
    *,
    fine_hw: tuple[int, int],
    coarse_hw: tuple[int, int],
    pixel_stride: int = 1,
) -> PyramidTransport:
    lookup = fine_to_coarse_lookup(
        fine_h=fine_hw[0],
        fine_w=fine_hw[1],
        coarse_h=coarse_hw[0],
        coarse_w=coarse_hw[1],
        device=torch.device("cpu"),
    )
    center_index = int(
        ((lookup.offsets_y == 0) & (lookup.offsets_x == 0))
        .nonzero(as_tuple=False)
        .item()
    )
    probs = torch.zeros(1, 9, *fine_hw)
    probs[:, center_index] = 1.0
    return PyramidTransport(
        probs=probs,
        logits=probs,
        expected_offset=pfs._expected_local_offset(probs, lookup=lookup),
        lookup=lookup,
        pixel_stride=pixel_stride,
    )


def _edge_index(lookup: pfs.FineToCoarseLookup, *, dy: int, dx: int) -> int:
    return int(
        ((lookup.offsets_y == dy) & (lookup.offsets_x == dx))
        .nonzero(as_tuple=False)
        .item()
    )


def _one_hot_transport_from_edges(
    *,
    fine_hw: tuple[int, int],
    coarse_hw: tuple[int, int],
    edges: dict[tuple[int, int], tuple[int, int]],
    pixel_stride: int = 1,
) -> PyramidTransport:
    lookup = fine_to_coarse_lookup(
        fine_h=fine_hw[0],
        fine_w=fine_hw[1],
        coarse_h=coarse_hw[0],
        coarse_w=coarse_hw[1],
        device=torch.device("cpu"),
    )
    center_index = _edge_index(lookup, dy=0, dx=0)
    probs = torch.zeros(1, 9, *fine_hw)
    probs[:, center_index] = 1.0
    for (y, x), (dy, dx) in edges.items():
        edge = _edge_index(lookup, dy=dy, dx=dx)
        assert bool(lookup.valid_edges[y * fine_hw[1] + x, edge])
        probs[:, :, y, x] = 0.0
        probs[:, edge, y, x] = 1.0
    return PyramidTransport(
        probs=probs,
        logits=probs,
        expected_offset=pfs._expected_local_offset(probs, lookup=lookup),
        lookup=lookup,
        pixel_stride=pixel_stride,
    )


def _manual_spatial_conv(
    coarse: torch.Tensor, *, transport: PyramidTransport
) -> torch.Tensor:
    batch_size, channels, coarse_h, coarse_w = coarse.shape
    fine_h, fine_w = transport.lookup.fine_hw
    indices = transport.lookup.target_indices.to(device=coarse.device).reshape(-1)
    flat_coarse = coarse.reshape(batch_size, channels, coarse_h * coarse_w)
    gathered = flat_coarse.index_select(dim=2, index=indices)
    gathered = gathered.view(batch_size, channels, fine_h * fine_w, 9)
    probs = rearrange(transport.probs, "b k h w -> b 1 (h w) k")
    out = (gathered * probs).sum(dim=3)
    return rearrange(out, "b c (h w) -> b c h w", h=fine_h, w=fine_w)


def test_configure_optimizers_without_trainer_returns_optimizer() -> None:
    model = _tiny_system()

    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)


def test_configure_optimizers_adds_step_warmup_cosine_scheduler() -> None:
    model = _tiny_system(warmup_epochs=10)
    _attach_trainer(model, estimated_stepping_batches=400, max_epochs=100)

    optimizers, schedulers = model.configure_optimizers()

    assert len(optimizers) == 1
    assert isinstance(optimizers[0], torch.optim.AdamW)
    assert len(schedulers) == 1
    assert schedulers[0]["interval"] == "step"
    scheduler = schedulers[0]["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
    assert scheduler._milestones == [40]


def test_forward_shapes_for_stride_eight_model() -> None:
    model = _tiny_system()
    image = torch.randn(2, 1, 16, 16)

    output = model(image)

    assert output["recon"].shape == image.shape
    assert output["foreground_logits"].shape == (2, 1, 2, 2)
    assert output["foreground_presence"].shape == (2, 1, 2, 2)
    assert output["p_fg"].shape == (2, 1, 16, 16)
    assert output["expected_flow"].shape == (2, 2, 16, 16)
    assert output["foreground_latents"].shape == (2, LATENT_DIM, 2, 2)
    assert output["foreground_latent_mu"].shape == (2, LATENT_DIM, 2, 2)
    assert output["foreground_latent_logvar"].shape == (2, LATENT_DIM, 2, 2)
    assert output["background_latents"].shape == (2, LATENT_DIM, 2, 2)
    assert output["background_latent_mu"].shape == (2, LATENT_DIM, 2, 2)
    assert output["background_latent_logvar"].shape == (2, LATENT_DIM, 2, 2)
    assert output["foreground_features"].shape == (2, LATENT_DIM, 2, 2)
    assert output["background_features"].shape == (2, LATENT_DIM, 2, 2)
    assert output["foreground_features"] is output["foreground_latents"]
    assert output["background_features"] is output["background_latents"]
    assert output["foreground_kl"].ndim == 0
    assert output["background_kl"].ndim == 0
    assert output["flow_l2"].ndim == 0
    assert output["foreground_kl_total"].ndim == 0
    assert output["background_kl_total"].ndim == 0
    assert output["flow_l2_total"].ndim == 0
    assert len(output["layer_transports"]) == 4
    assert [transport.lookup.fine_hw for transport in output["layer_transports"]] == [
        (2, 2),
        (4, 4),
        (8, 8),
        (16, 16),
    ]
    assert [transport.lookup.coarse_hw for transport in output["layer_transports"]] == [
        (2, 2),
        (2, 2),
        (4, 4),
        (8, 8),
    ]
    assert output["dense_assignment"] is None


def test_background_pool_stride_one_preserves_structure_and_outputs() -> None:
    torch.manual_seed(7)
    default = _tiny_system(dual_stream=True).eval()
    explicit = _tiny_system(dual_stream=True, background_latent_pool_stride=1).eval()
    explicit.load_state_dict(default.state_dict())
    image = torch.randn(2, 1, 16, 16)

    with torch.no_grad():
        default_output = default(image)
        explicit_output = explicit(image)

    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert explicit.background_latent_pool_stride == 1
    for key in (
        "recon",
        "foreground_latents",
        "background_latents",
        "foreground_kl",
        "background_kl",
    ):
        torch.testing.assert_close(default_output[key], explicit_output[key])


def test_background_posterior_receives_float32_pool_upsample_bottleneck() -> None:
    model = _tiny_system(background_latent_pool_stride=2).eval()
    image = torch.randn(1, 1, 32, 32)
    captured: dict[str, torch.Tensor] = {}

    foreground_handle = model.foreground_latent_head.register_forward_pre_hook(
        lambda _module, args: captured.update(foreground=cast(torch.Tensor, args[0]))
    )
    background_handle = model.background_latent_head.register_forward_pre_hook(
        lambda _module, args: captured.update(background=cast(torch.Tensor, args[0]))
    )
    with torch.no_grad():
        output = model(image)
        encoder_features = cast(PyramidFlowEncoderOutput, model.encoder(image)).features
    foreground_handle.remove()
    background_handle.remove()

    pooled_background = torch.nn.functional.avg_pool2d(
        encoder_features.float(), kernel_size=2, stride=2
    )
    expected_background = torch.nn.functional.interpolate(
        pooled_background,
        size=encoder_features.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).to(dtype=encoder_features.dtype)
    torch.testing.assert_close(captured["foreground"], encoder_features)
    torch.testing.assert_close(captured["background"], expected_background)
    assert output["foreground_latents"].shape[-2:] == (4, 4)
    assert output["background_latent_mu"].shape == (1, LATENT_DIM, 4, 4)
    assert output["background_latent_logvar"].shape == (1, LATENT_DIM, 4, 4)
    assert output["background_latents"].shape == (1, LATENT_DIM, 4, 4)

    reduced_precision = torch.arange(64, dtype=torch.bfloat16).reshape(1, 1, 8, 8)
    actual = model._pool_background_encoder_features(reduced_precision)
    reduced_precision_pooled = torch.nn.functional.avg_pool2d(
        reduced_precision.float(), kernel_size=2, stride=2
    )
    expected = torch.nn.functional.interpolate(
        reduced_precision_pooled,
        size=reduced_precision.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).to(dtype=reduced_precision.dtype)
    torch.testing.assert_close(actual, expected)


def test_stride_four_background_posterior_is_sampled_after_pixel_upsampling() -> None:
    model = _tiny_system(dual_stream=True, background_latent_pool_stride=4)
    background_dim = model.background_latent_dim
    with torch.no_grad():
        output_projection = cast(torch.nn.Conv2d, model.background_latent_head[-1])
        output_projection.weight.zero_()
        assert output_projection.bias is not None
        output_projection.bias[:background_dim].fill_(0.5)
        output_projection.bias[background_dim:].fill_(-0.25)

    torch.manual_seed(17)
    output = model(torch.randn(2, 1, 32, 32))

    assert output["foreground_latent_mu"].shape == (2, LATENT_DIM, 4, 4)
    assert output["background_latent_mu"].shape == (2, background_dim, 32, 32)
    assert output["background_latent_logvar"].shape == (2, background_dim, 32, 32)
    assert output["background_latents"].shape == (2, background_dim, 32, 32)
    assert not torch.allclose(
        output["background_latents"][..., 0, 0],
        output["background_latents"][..., 0, 1],
    )
    expected_element_kl = 0.5 * (0.5**2 + torch.exp(torch.tensor(-0.25)) - 0.75)
    torch.testing.assert_close(output["background_kl"], expected_element_kl)
    torch.testing.assert_close(
        output["background_kl_total"],
        expected_element_kl * background_dim * 32 * 32,
    )

    model.eval()
    eval_output = model(torch.randn(2, 1, 32, 32))
    torch.testing.assert_close(
        eval_output["background_latents"], eval_output["background_latent_mu"]
    )


def test_encoder_returns_deterministic_bottleneck_features() -> None:
    encoder = _tiny_encoder()
    image = torch.randn(2, 1, 16, 16)

    output = encoder(image)

    assert isinstance(output, PyramidFlowEncoderOutput)
    assert output.features.shape == (2, 16, 2, 2)
    assert not hasattr(encoder, "latent_mu_head")
    assert not hasattr(encoder, "latent_logvar_head")
    assert not hasattr(encoder, "up_stages")
    assert not hasattr(encoder, "flow_heads")


def test_encoder_default_latent_dim_matches_bottleneck_width() -> None:
    encoder = PyramidFlowEncoder2d(
        in_channels=1,
        channels=(4, 8, 12, 16),
        strides=(2, 2, 2),
        down_blocks=(1, 1, 1, 1),
        norm=("GROUP", {"num_groups": 2}),
    )

    assert encoder.dim == 16
    assert encoder.bottleneck_channels == 16


def test_decoder_predicts_one_transport_per_unrolled_layer() -> None:
    decoder = _tiny_decoder(stage_blocks=(2, 3, 1, 2))
    foreground, background, foreground_presence = _decoder_stream_inputs()

    recon, transports = decoder(
        foreground,
        background,
        foreground_presence,
    )

    assert recon.shape == (2, 1, 16, 16)
    assert len(transports) == 8
    assert decoder.stage_blocks == (2, 3, 1, 2)
    assert len(decoder.layers) == 8
    assert decoder.layer_stage_indices == (0, 0, 1, 1, 1, 2, 3, 3)
    assert decoder.layer_strides == (1, 1, 2, 1, 1, 2, 2, 1)
    assert decoder.layer_pixel_strides == (8, 8, 4, 4, 4, 2, 1, 1)
    assert [transport.lookup.coarse_hw for transport in transports] == [
        (2, 2),
        (2, 2),
        (2, 2),
        (4, 4),
        (4, 4),
        (4, 4),
        (8, 8),
        (16, 16),
    ]
    assert [transport.lookup.fine_hw for transport in transports] == [
        (2, 2),
        (2, 2),
        (4, 4),
        (4, 4),
        (4, 4),
        (8, 8),
        (16, 16),
        (16, 16),
    ]
    assert not torch.allclose(transports[0].logits, transports[1].logits)


@pytest.mark.parametrize(
    ("stream_name", "shape", "expected_message"),
    [
        ("foreground", (2, LATENT_DIM, 2), "shape"),
        ("background", (2, LATENT_DIM + 1, 2, 2), "channels"),
        ("foreground_presence", (2, 2, 2, 2), "channels"),
    ],
)
def test_decoder_validates_stream_shapes_at_its_public_boundary(
    stream_name: str,
    shape: tuple[int, ...],
    expected_message: str,
) -> None:
    decoder = _tiny_decoder()
    streams = dict(
        zip(
            ("foreground", "background", "foreground_presence"),
            _decoder_stream_inputs(),
            strict=True,
        )
    )
    streams[stream_name] = torch.randn(shape)

    with pytest.raises(ValueError, match=expected_message):
        decoder(
            streams["foreground"],
            streams["background"],
            streams["foreground_presence"],
        )


@pytest.mark.parametrize(
    ("stream_name", "shape", "expected_message"),
    [
        ("background", (1, LATENT_DIM, 2, 2), "batch sizes"),
        ("foreground_presence", (2, 1, 3, 2), "spatial shapes"),
    ],
)
def test_decoder_validates_matching_stream_grids_at_its_public_boundary(
    stream_name: str,
    shape: tuple[int, ...],
    expected_message: str,
) -> None:
    decoder = _tiny_decoder()
    streams = dict(
        zip(
            ("foreground", "background", "foreground_presence"),
            _decoder_stream_inputs(),
            strict=True,
        )
    )
    streams[stream_name] = torch.randn(shape)

    with pytest.raises(ValueError, match=expected_message):
        decoder(
            streams["foreground"],
            streams["background"],
            streams["foreground_presence"],
        )


def test_decoder_blocks_always_use_layer_norm_and_gelu() -> None:
    decoder = _tiny_decoder(stage_blocks=(2, 1, 1, 1))

    for layer in decoder.layers:
        layer = cast(pfs.PyramidFlowDecoderLayer2d, layer)
        assert isinstance(layer.block.norm, pfs.ChannelLayerNorm2d)
        assert layer.block.norm.affine
        assert not layer.block.norm.use_bias
        assert isinstance(layer.block.act, torch.nn.GELU)
        assert isinstance(layer.block.ffn_norm, pfs.ChannelLayerNorm2d)
        assert layer.block.ffn_norm.affine
        assert not layer.block.ffn_norm.use_bias
        assert isinstance(layer.block.ffn, PointwiseMLP2d)
        assert any(isinstance(module, torch.nn.GELU) for module in layer.block.ffn)
        assert all(
            module.bias is None
            for module in layer.block.ffn
            if isinstance(module, torch.nn.Conv2d)
        )
        assert not any(
            isinstance(module, torch.nn.PReLU) for module in layer.block.modules()
        )


def test_disabled_dual_stream_has_no_background_path_parameters() -> None:
    decoder = _tiny_decoder()

    assert not decoder.dual_stream
    assert not hasattr(decoder, "background_projection")


@pytest.mark.parametrize("value_modulation", [False, True])
def test_transport_residual_block_is_zero_preserving(
    value_modulation: bool,
) -> None:
    torch.manual_seed(11)
    block = pfs.TransportResidualBlock2d(
        in_channels=2,
        out_channels=3,
        ffn_expansion=2,
        value_modulation=value_modulation,
    )
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.uniform_(-2.0, 2.0)
    transport = _gather_test_transport(
        batch_size=2,
        fine_hw=(4, 4),
        coarse_hw=(2, 2),
    )
    gate = torch.rand(2, 1, 4, 4)

    actual = block(torch.zeros(2, 2, 2, 2), transport=transport, gate=gate)

    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("transport_predictor", ["attention", "conv"])
@pytest.mark.parametrize("value_modulation", [False, True])
@pytest.mark.parametrize("dual_stream", [False, True])
def test_decoder_zero_latents_produce_only_spatially_constant_output_bias(
    transport_predictor: Literal["attention", "conv"],
    value_modulation: bool,
    dual_stream: bool,
) -> None:
    torch.manual_seed(19)
    decoder = _tiny_decoder(
        stage_blocks=(1, 1, 1, 1),
        transport_predictor=transport_predictor,
        dual_stream=dual_stream,
        value_modulation=value_modulation,
    ).eval()
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.uniform_(-0.5, 0.5)
        foreground, background, presence = _decoder_stream_inputs(
            background_channels=decoder.channels[-1]
            if dual_stream
            else decoder.in_channels,
            background_grid_hw=(16, 16) if dual_stream else (2, 2),
        )
        _, arbitrary_transports = decoder(foreground, background, presence)
        zero_foreground = torch.zeros_like(foreground)
        zero_background = torch.zeros_like(background)
        recomputed, _ = decoder(zero_foreground, zero_background, presence)
        fixed, _ = decoder(
            zero_foreground,
            zero_background,
            1.0 - presence,
            layer_transports=arbitrary_transports,
        )

    expected = decoder.output(
        torch.zeros_like(recomputed).expand(-1, decoder.channels[-1], -1, -1)
    )
    torch.testing.assert_close(recomputed, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(fixed, expected, atol=0.0, rtol=0.0)


def test_pixel_feature_unit_normalization_is_finite_and_preserves_zeros() -> None:
    features = torch.tensor(
        [
            [
                [[0.0, 3.0, 1e-30]],
                [[0.0, 4.0, -1e-30]],
                [[0.0, 0.0, 1e-30]],
            ]
        ]
    )

    normalized = PyramidFlowDecoder2d._unit_normalize_pixel_features(features)

    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(
        normalized[:, :, :, 0], torch.zeros_like(normalized[:, :, :, 0])
    )
    torch.testing.assert_close(
        normalized[:, :, :, 1].square().sum(dim=1),
        torch.ones_like(normalized[:, 0, :, 1]),
    )
    assert normalized[:, :, :, 2].norm() < 1e-20


def test_variance_preserving_pixel_blend_has_unit_norm_for_orthogonal_streams() -> None:
    foreground = torch.tensor([[[[1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0]]]])
    background = torch.tensor([[[[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]]]])
    presence = torch.tensor([[[[0.0, 0.37, 1.0]]]])

    mixed = PyramidFlowDecoder2d._variance_preserving_pixel_blend(
        foreground=foreground,
        background=background,
        foreground_presence=presence,
    )

    torch.testing.assert_close(
        mixed.square().sum(dim=1), torch.ones_like(presence[:, 0])
    )
    torch.testing.assert_close(mixed[..., 0], background[..., 0])
    torch.testing.assert_close(mixed[..., 2], foreground[..., 2])


def test_decoder_uses_unbounded_pointwise_mlp_image_head() -> None:
    decoder = _tiny_decoder(dual_stream=True)

    assert isinstance(decoder.output, PointwiseMLP2d)
    convolutions = [
        module for module in decoder.output if isinstance(module, torch.nn.Conv2d)
    ]
    assert len(convolutions) == 2
    assert convolutions[0].in_channels == decoder.channels[-1]
    assert convolutions[-1].out_channels == decoder.out_channels
    assert not any(
        isinstance(module, (torch.nn.Sigmoid, torch.nn.Tanh))
        for module in decoder.output
    )


@pytest.mark.parametrize("presence_value", [0.0, 0.37, 1.0])
def test_dual_stream_matches_variance_preserving_pixel_blend_reference(
    presence_value: float,
) -> None:
    torch.manual_seed(23)
    decoder = _tiny_decoder(
        stage_blocks=(1, 1, 1, 1),
        dual_stream=True,
    ).eval()
    foreground, background, _presence = _decoder_stream_inputs(
        batch_size=1,
        background_channels=decoder.channels[-1],
        background_grid_hw=(16, 16),
    )
    presence = torch.full((1, 1, 2, 2), presence_value)
    captured: dict[str, torch.Tensor] = {}

    def capture_output_input(
        _module: torch.nn.Module, args: tuple[object, ...]
    ) -> None:
        captured["features"] = cast(torch.Tensor, args[0]).detach()

    def capture_first_layer_input(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        captured["first_layer"] = cast(torch.Tensor, kwargs["x"]).detach()

    output_hook = decoder.output.register_forward_pre_hook(capture_output_input)
    first_layer_hook = decoder.layers[0].register_forward_pre_hook(
        capture_first_layer_input, with_kwargs=True
    )
    with torch.no_grad():
        recon, transports = decoder(foreground, background, presence)
    output_hook.remove()
    first_layer_hook.remove()

    foreground_expected = decoder.latent_blend_norm(foreground)
    presence_expected = presence.to(dtype=foreground_expected.dtype)
    first_layer_expected = presence_expected * foreground_expected
    foreground_expected = first_layer_expected
    with torch.no_grad():
        for layer, transport in zip(decoder.layers, transports, strict=True):
            foreground_expected, presence_expected, _ = cast(
                pfs.PyramidFlowDecoderLayer2d, layer
            )(
                x=foreground_expected,
                foreground_presence=presence_expected,
                transport=transport,
            )
        background_expected = background
        foreground_expected = decoder._unit_normalize_pixel_features(
            foreground_expected
        )
        background_expected = decoder._unit_normalize_pixel_features(
            background_expected
        )
        expected = decoder._variance_preserving_pixel_blend(
            foreground=foreground_expected,
            background=background_expected,
            foreground_presence=presence_expected,
        )

    torch.testing.assert_close(captured["first_layer"], first_layer_expected)
    torch.testing.assert_close(captured["features"], expected)
    torch.testing.assert_close(recon, decoder.output(expected))
    torch.testing.assert_close(
        foreground_expected.float().norm(dim=1),
        (presence_expected[:, 0] > 0.0).float(),
    )
    torch.testing.assert_close(
        background_expected.float().norm(dim=1),
        torch.ones_like(background_expected[:, 0]).float(),
    )


@pytest.mark.parametrize("transport_predictor", ["attention", "conv"])
def test_dual_stream_background_cannot_change_transports(
    transport_predictor: Literal["attention", "conv"],
) -> None:
    torch.manual_seed(29)
    decoder = _tiny_decoder(
        stage_blocks=(1, 1, 1, 1),
        transport_predictor=transport_predictor,
        dual_stream=True,
    ).eval()
    foreground, background, presence = _decoder_stream_inputs(
        batch_size=1,
        background_channels=decoder.channels[-1],
        background_grid_hw=(16, 16),
    )

    with torch.no_grad():
        recon, transports = decoder(foreground, background, presence)
        changed_recon, changed_transports = decoder(
            foreground, background + 10.0 * torch.randn_like(background), presence
        )

    assert not torch.allclose(recon, changed_recon)
    for transport, changed_transport in zip(
        transports, changed_transports, strict=True
    ):
        torch.testing.assert_close(transport.logits, changed_transport.logits)
        torch.testing.assert_close(transport.probs, changed_transport.probs)
        torch.testing.assert_close(
            transport.expected_offset, changed_transport.expected_offset
        )


def test_dual_stream_reconstruction_gradients_reach_both_paths_and_alpha() -> None:
    torch.manual_seed(31)
    decoder = _tiny_decoder(
        stage_blocks=(1, 1, 1, 1),
        transport_predictor="conv",
        dual_stream=True,
    )
    foreground = torch.randn(2, LATENT_DIM, 2, 2, requires_grad=True)
    background = torch.randn(2, decoder.channels[-1], 16, 16, requires_grad=True)
    presence = torch.rand(2, 1, 2, 2, requires_grad=True)

    recon, _transports = decoder(foreground, background, presence)
    recon.square().mean().backward()

    assert foreground.grad is not None and foreground.grad.abs().sum() > 0
    assert background.grad is not None and background.grad.abs().sum() > 0
    assert presence.grad is not None and presence.grad.abs().sum() > 0
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in decoder.output.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for layer in decoder.layers
        for parameter in cast(
            pfs.PyramidFlowDecoderLayer2d, layer
        ).transport_scorer.parameters()
    )


def test_dual_stream_decoder_requires_pixel_resolution_background_features() -> None:
    decoder = _tiny_decoder(dual_stream=True)
    foreground, _background, presence = _decoder_stream_inputs(batch_size=1)
    coarse_background = torch.randn(1, decoder.channels[-1], 1, 1)

    with pytest.raises(ValueError, match="decoder output grid"):
        decoder(foreground, coarse_background, presence)


def test_decoder_rejects_pre_blend_channel_projection() -> None:
    with pytest.raises(ValueError, match=r"in_channels must equal channels\[0\]"):
        PyramidFlowDecoder2d(
            in_channels=2,
            out_channels=1,
            channels=(4,),
            strides=(),
            feature_stride=1,
            stage_blocks=(1,),
        )


def test_system_rejects_decoder_width_different_from_shared_latent_dim() -> None:
    decoder = PyramidFlowDecoder2d(
        in_channels=2,
        out_channels=1,
        channels=(2,),
        strides=(),
        feature_stride=1,
        stage_blocks=(1,),
    )

    with pytest.raises(ValueError, match="decoder in_channels must match latent_dim"):
        PyramidFlowSystem(
            encoder=PyramidFlowEncoder2d(
                in_channels=1,
                channels=(2,),
                strides=(),
                norm=("GROUP", {"num_groups": 1}),
            ),
            decoder=decoder,
            foreground_latent_hidden_dim=4,
            background_latent_hidden_dim=1,
            patch_size=1,
            latent_dim=3,
            latent_head_norm=("GROUP", {"num_groups": 1}),
            log_images_every_n_epochs=0,
        )


@pytest.mark.parametrize(
    ("foreground_hidden_dim", "background_hidden_dim", "expected_message"),
    [
        (0, BACKGROUND_LATENT_HIDDEN_DIM, "foreground_latent_hidden_dim"),
        (FOREGROUND_LATENT_HIDDEN_DIM, -1, "background_latent_hidden_dim"),
    ],
)
def test_system_rejects_nonpositive_latent_head_hidden_dimensions(
    foreground_hidden_dim: int,
    background_hidden_dim: int,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        _tiny_system(
            foreground_latent_hidden_dim=foreground_hidden_dim,
            background_latent_hidden_dim=background_hidden_dim,
        )


@pytest.mark.parametrize("stride", [0, -1])
def test_system_rejects_nonpositive_background_latent_pool_stride(
    stride: int,
) -> None:
    with pytest.raises(ValueError, match="background_latent_pool_stride"):
        _tiny_system(dual_stream=True, background_latent_pool_stride=stride)


def test_background_pool_upsample_bottleneck_supports_single_stream_decoder() -> None:
    model = _tiny_system(background_latent_pool_stride=2).eval()
    image = torch.randn(1, 1, 32, 32)

    output = model(image)

    assert output["recon"].shape == image.shape
    assert output["background_latents"].shape == (1, LATENT_DIM, 4, 4)


def test_background_pool_upsample_requires_divisible_encoder_grid() -> None:
    model = _tiny_system(dual_stream=True, background_latent_pool_stride=4).eval()

    with pytest.raises(ValueError, match="encoder feature spatial shape"):
        model(torch.randn(1, 1, 24, 24))


def test_background_pool_upsample_model_padding_uses_effective_pixel_stride() -> None:
    model = _tiny_system(dual_stream=True, background_latent_pool_stride=4).eval()
    image = torch.randn(1, 1, 18, 19)

    with pytest.warns(UserWarning, match="required input divisibility 32"):
        output, original_hw, padded_hw = model._forward_with_model_padding(image)

    assert model._reconstruction_input_divisibility == 32
    assert model._segmentation_input_divisibility == 32
    assert model._input_divisibility_stride == 32
    assert original_hw == (18, 19)
    assert padded_hw == (32, 32)
    assert output["recon"].shape[-2:] == original_hw
    assert output["p_fg"].shape[-2:] == original_hw
    assert output["expected_flow"].shape[-2:] == original_hw


def test_decoder_normalizes_latent_streams_without_affine_before_first_blend() -> None:
    decoder = PyramidFlowDecoder2d(
        in_channels=2,
        out_channels=1,
        channels=(2,),
        strides=(),
        feature_stride=1,
        stage_blocks=(1,),
    )
    foreground = torch.tensor([[[[2.0, 4.0]], [[6.0, 10.0]]]])
    background = torch.tensor([[[[20.0, 40.0]], [[60.0, 100.0]]]])
    foreground_presence = torch.full((1, 1, 1, 2), 0.25)
    transport = _center_one_hot_transport(fine_hw=(1, 2), coarse_hw=(1, 2))
    captured: dict[str, torch.Tensor] = {}

    def capture_first_layer(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        captured["x"] = cast(torch.Tensor, kwargs["x"]).detach()
        captured["foreground_presence"] = cast(
            torch.Tensor, kwargs["foreground_presence"]
        ).detach()

    handle = decoder.layers[0].register_forward_pre_hook(
        capture_first_layer,
        with_kwargs=True,
    )
    decoder(
        foreground,
        background,
        foreground_presence,
        layer_transports=(transport,),
    )
    handle.remove()

    assert isinstance(decoder.latent_blend_norm, pfs.ChannelLayerNorm2d)
    assert not decoder.latent_blend_norm.affine
    assert not hasattr(decoder, "input_proj")
    assert "weight" not in dict(decoder.latent_blend_norm.named_parameters())
    expected_foreground = decoder.latent_blend_norm(foreground)
    expected_background = decoder.latent_blend_norm(background)
    expected_visible = (
        foreground_presence * expected_foreground
        + (1.0 - foreground_presence) * expected_background
    )
    torch.testing.assert_close(captured["x"], expected_visible)
    torch.testing.assert_close(captured["foreground_presence"], foreground_presence)
    torch.testing.assert_close(
        expected_foreground.mean(dim=1),
        torch.zeros_like(expected_foreground.mean(dim=1)),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        expected_background.mean(dim=1),
        torch.zeros_like(expected_background.mean(dim=1)),
        atol=1e-5,
        rtol=0.0,
    )


def test_decoder_can_disable_latent_blend_normalization() -> None:
    decoder = PyramidFlowDecoder2d(
        in_channels=2,
        out_channels=1,
        channels=(2,),
        strides=(),
        feature_stride=1,
        stage_blocks=(1,),
        normalize_latent_blend=False,
    )
    foreground = torch.tensor([[[[2.0]], [[6.0]]]])
    background = torch.tensor([[[[20.0]], [[60.0]]]])
    foreground_presence = torch.full((1, 1, 1, 1), 0.25)
    transport = _center_one_hot_transport(fine_hw=(1, 1), coarse_hw=(1, 1))
    captured: dict[str, torch.Tensor] = {}

    def capture_first_layer(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        captured["x"] = cast(torch.Tensor, kwargs["x"]).detach()

    handle = decoder.layers[0].register_forward_pre_hook(
        capture_first_layer,
        with_kwargs=True,
    )
    decoder(
        foreground,
        background,
        foreground_presence,
        layer_transports=(transport,),
    )
    handle.remove()

    assert decoder.normalize_latent_blend is False
    assert isinstance(decoder.latent_blend_norm, torch.nn.Identity)
    assert not hasattr(decoder, "input_proj")
    expected_visible = (
        foreground_presence * foreground + (1.0 - foreground_presence) * background
    )
    torch.testing.assert_close(captured["x"], expected_visible)


def test_no_recurrent_transport_api_remains() -> None:
    model = _tiny_system(stage_blocks=(2, 1, 1, 1))

    assert not hasattr(model.decoder, "stages")
    assert not hasattr(model.decoder, "_recurrent_transport")
    assert not hasattr(model.decoder, "_forward_impl")
    assert not hasattr(model.decoder, "input_proj")
    assert not hasattr(model, "foreground_head")
    assert not hasattr(model, "background_head")
    assert not hasattr(model, "_rollout_layer_transports")
    assert not hasattr(model.decoder, "forward_with_layer_transports")
    assert not hasattr(model.decoder, "forward_with_fixed_layer_transports")
    assert not hasattr(model.decoder, "forward_with_stage_transports")
    assert not hasattr(model.decoder, "forward_with_fixed_stage_transports")


@pytest.mark.parametrize("dual_stream", [False, True])
def test_fixed_layer_transports_validate_unrolled_layer_metadata(
    dual_stream: bool,
) -> None:
    decoder = _tiny_decoder(stage_blocks=(2, 1, 1, 1), dual_stream=dual_stream)
    foreground, background, foreground_presence = _decoder_stream_inputs(
        batch_size=1,
        background_channels=decoder.channels[-1]
        if dual_stream
        else decoder.in_channels,
        background_grid_hw=(16, 16) if dual_stream else (2, 2),
    )
    _recon, transports = decoder(
        foreground,
        background,
        foreground_presence,
    )

    with pytest.raises(ValueError, match="decoder layers"):
        decoder(
            foreground,
            background,
            foreground_presence,
            layer_transports=transports[:-1],
        )

    bad_stride = transports[1]._replace(pixel_stride=transports[1].pixel_stride + 1)
    with pytest.raises(ValueError, match="pixel_stride"):
        decoder(
            foreground,
            background,
            foreground_presence,
            layer_transports=(transports[0], bad_stride, *transports[2:]),
        )


@pytest.mark.parametrize("dual_stream", [False, True])
def test_zero_alpha_decoder_output_ignores_foreground_and_transport(
    dual_stream: bool,
) -> None:
    decoder = PyramidFlowDecoder2d(
        in_channels=2,
        out_channels=1,
        channels=(2,),
        strides=(),
        feature_stride=1,
        stage_blocks=(1,),
        dual_stream=dual_stream,
        value_modulation=True,
    )
    projection = cast(Any, decoder.layers[0]).block.value_modulation_proj
    assert projection is not None
    with torch.no_grad():
        projection.weight.copy_(torch.linspace(-1e3, 1e3, steps=18).view(2, 9, 1, 1))
    foreground_1 = torch.randn(1, 2, 2, 3)
    foreground_2 = torch.randn(1, 2, 2, 3)
    background = torch.randn(1, 2, 2, 3)
    foreground_presence = torch.zeros(1, 1, 2, 3)
    center_transport = _center_one_hot_transport(fine_hw=(2, 3), coarse_hw=(2, 3))
    random_transport = _gather_test_transport(
        batch_size=1,
        fine_hw=(2, 3),
        coarse_hw=(2, 3),
    )

    recon_1, _ = decoder(
        foreground_1,
        background,
        foreground_presence,
        layer_transports=(center_transport,),
    )
    recon_2, _ = decoder(
        foreground_2,
        background,
        foreground_presence,
        layer_transports=(random_transport,),
    )

    torch.testing.assert_close(recon_1, recon_2)


@pytest.mark.parametrize("dual_stream", [False, True])
def test_one_alpha_decoder_output_ignores_background(dual_stream: bool) -> None:
    decoder = PyramidFlowDecoder2d(
        in_channels=2,
        out_channels=1,
        channels=(2,),
        strides=(),
        feature_stride=1,
        stage_blocks=(1,),
        dual_stream=dual_stream,
    )
    foreground = torch.randn(1, 2, 2, 3)
    background_1 = torch.randn(1, 2, 2, 3)
    background_2 = torch.randn(1, 2, 2, 3)
    foreground_presence = torch.ones(1, 1, 2, 3)
    transport = _gather_test_transport(
        batch_size=1,
        fine_hw=(2, 3),
        coarse_hw=(2, 3),
    )

    recon_1, _ = decoder(
        foreground,
        background_1,
        foreground_presence,
        layer_transports=(transport,),
    )
    recon_2, _ = decoder(
        foreground,
        background_2,
        foreground_presence,
        layer_transports=(transport,),
    )

    torch.testing.assert_close(recon_1, recon_2)


def test_decoder_layer_routes_alpha_with_same_transport_before_residual_gate() -> None:
    layer = pfs.PyramidFlowDecoderLayer2d(
        in_channels=2,
        out_channels=2,
        stride=1,
        pixel_stride=1,
        attention_channels=2,
        use_edge_bias=False,
        ffn_expansion=1,
    )
    x = torch.randn(1, 2, 2, 3)
    foreground_presence = torch.rand(1, 1, 2, 3)
    transport = _gather_test_transport(
        batch_size=1,
        fine_hw=(2, 3),
        coarse_hw=(2, 3),
    )

    next_x, next_presence, _transport = layer(
        x=x,
        foreground_presence=foreground_presence,
        transport=transport,
    )
    expected_presence = PyramidFlowSystem._gather_scalar_map(
        foreground_presence,
        transport=transport,
    )
    expected_skip = layer.block.skip(x)

    assert next_x.shape == x.shape
    torch.testing.assert_close(next_presence, expected_presence)
    assert not torch.allclose(next_x, expected_skip)


def test_decoder_layer_gates_transport_scorer_source_and_query_by_presence() -> None:
    layer = pfs.PyramidFlowDecoderLayer2d(
        in_channels=2,
        out_channels=2,
        stride=2,
        pixel_stride=1,
        attention_channels=2,
        use_edge_bias=False,
        ffn_expansion=1,
    )
    x = torch.randn(1, 2, 2, 3)
    foreground_presence = torch.tensor([[[[0.0, 0.25, 1.0], [0.5, 0.75, 0.0]]]])
    captured: dict[str, torch.Tensor] = {}
    original_forward = layer.transport_scorer.forward

    def capture_forward(
        _module: torch.nn.Module,
        *,
        source: torch.Tensor,
        query: torch.Tensor,
        lookup: pfs.FineToCoarseLookup,
        pixel_stride: int,
    ) -> PyramidTransport:
        captured["source"] = source.detach()
        captured["query"] = query.detach()
        return original_forward(
            source=source, query=query, lookup=lookup, pixel_stride=pixel_stride
        )

    cast(Any, layer.transport_scorer).forward = types.MethodType(
        capture_forward, layer.transport_scorer
    )

    layer(x=x, foreground_presence=foreground_presence, transport=None)

    expected_source = x * foreground_presence
    expected_query = torch.nn.functional.interpolate(
        x, size=(4, 6), mode="nearest"
    ) * torch.nn.functional.interpolate(
        foreground_presence, size=(4, 6), mode="nearest"
    )
    torch.testing.assert_close(captured["source"], expected_source)
    torch.testing.assert_close(captured["query"], expected_query)


def test_zero_presence_transport_scoring_ignores_visible_feature_noise() -> None:
    layer = pfs.PyramidFlowDecoderLayer2d(
        in_channels=2,
        out_channels=2,
        stride=1,
        pixel_stride=1,
        attention_channels=2,
        use_edge_bias=False,
        ffn_expansion=1,
    )
    foreground_presence = torch.zeros(1, 1, 2, 3)
    x_1 = torch.randn(1, 2, 2, 3)
    x_2 = torch.randn(1, 2, 2, 3) * 100.0 + 50.0

    _next_x_1, _next_presence_1, transport_1 = layer(
        x=x_1,
        foreground_presence=foreground_presence,
        transport=None,
    )
    _next_x_2, _next_presence_2, transport_2 = layer(
        x=x_2,
        foreground_presence=foreground_presence,
        transport=None,
    )

    torch.testing.assert_close(transport_1.logits, transport_2.logits)
    torch.testing.assert_close(transport_1.probs, transport_2.probs)


def test_fixed_transport_bypasses_pre_attention_gate_and_preserves_alpha_routing() -> (
    None
):
    layer = pfs.PyramidFlowDecoderLayer2d(
        in_channels=2,
        out_channels=2,
        stride=1,
        pixel_stride=1,
        attention_channels=2,
        use_edge_bias=False,
        ffn_expansion=1,
    )
    x = torch.randn(1, 2, 2, 3)
    foreground_presence = torch.tensor([[[[0.0, 0.5, 1.0], [1.0, 0.25, 0.0]]]])
    transport = _gather_test_transport(
        batch_size=1,
        fine_hw=(2, 3),
        coarse_hw=(2, 3),
    )

    def fail_forward(
        _module: torch.nn.Module,
        *,
        source: torch.Tensor,
        query: torch.Tensor,
        lookup: pfs.FineToCoarseLookup,
        pixel_stride: int,
    ) -> PyramidTransport:
        raise AssertionError("fixed transports must bypass transport scoring")

    cast(Any, layer.transport_scorer).forward = types.MethodType(
        fail_forward, layer.transport_scorer
    )
    next_x, next_presence, returned_transport = layer(
        x=x,
        foreground_presence=foreground_presence,
        transport=transport,
    )
    expected_presence = PyramidFlowSystem._gather_scalar_map(
        foreground_presence,
        transport=transport,
    )
    expected_x = layer.block(x, transport=transport, gate=expected_presence)

    assert returned_transport is transport
    torch.testing.assert_close(next_presence, expected_presence)
    torch.testing.assert_close(next_x, expected_x)


def test_changing_transport_changes_features_and_alpha_routing_together() -> None:
    layer = pfs.PyramidFlowDecoderLayer2d(
        in_channels=2,
        out_channels=2,
        stride=1,
        pixel_stride=1,
        attention_channels=2,
        use_edge_bias=False,
        ffn_expansion=1,
    )
    x = torch.randn(1, 2, 2, 3)
    foreground_presence = torch.rand(1, 1, 2, 3)
    center_transport = _center_one_hot_transport(fine_hw=(2, 3), coarse_hw=(2, 3))
    random_transport = _gather_test_transport(
        batch_size=1,
        fine_hw=(2, 3),
        coarse_hw=(2, 3),
    )

    x_center, alpha_center, _ = layer(
        x=x,
        foreground_presence=foreground_presence,
        transport=center_transport,
    )
    x_random, alpha_random, _ = layer(
        x=x,
        foreground_presence=foreground_presence,
        transport=random_transport,
    )

    assert not torch.allclose(x_center, x_random)
    assert not torch.allclose(alpha_center, alpha_random)


def test_zero_alpha_gates_transport_update_before_residual_connection() -> None:
    layer = pfs.PyramidFlowDecoderLayer2d(
        in_channels=2,
        out_channels=3,
        stride=2,
        pixel_stride=1,
        attention_channels=2,
        use_edge_bias=False,
        ffn_expansion=1,
    )
    x = torch.randn(1, 2, 2, 2)
    foreground_presence = torch.zeros(1, 1, 2, 2)
    transport = _center_one_hot_transport(fine_hw=(4, 4), coarse_hw=(2, 2))

    next_x, next_presence, _transport = layer(
        x=x,
        foreground_presence=foreground_presence,
        transport=transport,
    )
    residual = torch.nn.functional.interpolate(x, size=(4, 4), mode="nearest")
    expected_residual = layer.block.skip(residual)
    expected = expected_residual + layer.block.ffn(
        layer.block.ffn_norm(expected_residual)
    )

    torch.testing.assert_close(next_presence, torch.zeros_like(next_presence))
    torch.testing.assert_close(next_x, expected)


@pytest.mark.parametrize("scale", [1, 2])
def test_spatially_varying_convolution_matches_reference_and_gradients(
    scale: int,
) -> None:
    fine_hw = (2 * scale, 3 * scale)
    transport = _gather_test_transport(batch_size=2, fine_hw=fine_hw, coarse_hw=(2, 3))
    actual_coarse = torch.linspace(-2.0, 2.0, steps=2 * 3 * 2 * 3).view(2, 3, 2, 3)
    actual_coarse.requires_grad_()
    expected_coarse = actual_coarse.detach().clone().requires_grad_()
    actual_logits = transport.logits.detach().clone().requires_grad_()
    expected_logits = transport.logits.detach().clone().requires_grad_()
    actual_transport = pfs._transport_from_logits(
        actual_logits,
        lookup=transport.lookup,
        pixel_stride=transport.pixel_stride,
    )
    expected_transport = pfs._transport_from_logits(
        expected_logits,
        lookup=transport.lookup,
        pixel_stride=transport.pixel_stride,
    )

    actual = pfs.TransportResidualBlock2d.spatially_varying_convolution(
        actual_coarse,
        transport=actual_transport,
    )
    expected = _manual_spatial_conv(
        expected_coarse,
        transport=expected_transport,
    )

    torch.testing.assert_close(actual, expected)
    output_grad = torch.linspace(-1.0, 1.0, steps=actual.numel()).view_as(actual)
    actual.backward(output_grad)
    expected.backward(output_grad)
    torch.testing.assert_close(actual_coarse.grad, expected_coarse.grad)
    torch.testing.assert_close(actual_logits.grad, expected_logits.grad)


@pytest.mark.parametrize("transport_predictor", ["attention", "conv"])
def test_decoder_value_modulation_is_optional_and_zero_initialized(
    transport_predictor: Literal["attention", "conv"],
) -> None:
    disabled = _tiny_decoder(transport_predictor=transport_predictor)
    enabled = _tiny_decoder(
        transport_predictor=transport_predictor,
        value_modulation=True,
    )

    assert not disabled.value_modulation
    assert all(
        cast(Any, layer).block.value_modulation_proj is None
        for layer in disabled.layers
    )
    assert enabled.value_modulation
    for layer in enabled.layers:
        layer_typed = cast(Any, layer)
        projection = layer_typed.block.value_modulation_proj
        assert layer_typed.value_modulation
        assert isinstance(projection, torch.nn.Conv2d)
        assert projection.in_channels == 9
        assert projection.out_channels == layer_typed.out_channels
        assert projection.bias is None
        torch.testing.assert_close(
            projection.weight, torch.zeros_like(projection.weight)
        )


def test_value_modulation_gain_is_identity_initialized_bounded_and_ignores_logits() -> (
    None
):
    block = pfs.TransportResidualBlock2d(
        in_channels=2,
        out_channels=3,
        ffn_expansion=1,
        value_modulation=True,
    )
    transport = _gather_test_transport(
        batch_size=2,
        fine_hw=(4, 4),
        coarse_hw=(2, 2),
    )
    initial_gain = block._value_modulation_gain(transport=transport)
    torch.testing.assert_close(initial_gain, torch.ones_like(initial_gain))

    projection = block.value_modulation_proj
    assert projection is not None
    with torch.no_grad():
        projection.weight.copy_(torch.linspace(-1e6, 1e6, steps=27).view(3, 9, 1, 1))
    shifted_logits = transport.logits + 1e30
    shifted_transport = transport._replace(logits=shifted_logits)
    gain = block._value_modulation_gain(transport=transport)
    shifted_gain = block._value_modulation_gain(transport=shifted_transport)

    assert torch.isfinite(gain).all()
    assert gain.amin() >= 0.0
    assert gain.amax() <= 2.0
    torch.testing.assert_close(gain, shifted_gain)


def test_transport_probabilities_modulate_routed_features() -> None:
    block = pfs.TransportResidualBlock2d(
        in_channels=2,
        out_channels=2,
        ffn_expansion=1,
        value_modulation=True,
    )
    with torch.no_grad():
        block.value_proj.weight.copy_(torch.eye(2).view(2, 2, 1, 1))
        projection = block.value_modulation_proj
        assert projection is not None
        projection.weight.zero_()
        projection.weight[0, 4] = 1.0
        projection.weight[1, 4] = -1.0
    x = torch.tensor([1.0, 2.0]).view(1, 2, 1, 1).expand(1, 2, 3, 3)
    center_transport = _center_one_hot_transport(fine_hw=(3, 3), coarse_hw=(3, 3))
    distributed_transport = _gather_test_transport(
        batch_size=1,
        fine_hw=(3, 3),
        coarse_hw=(3, 3),
    )

    values = block.value_proj(x)
    center_routed = block.spatially_varying_convolution(
        values, transport=center_transport
    )
    distributed_routed = block.spatially_varying_convolution(
        values, transport=distributed_transport
    )
    center_out = center_routed * block._value_modulation_gain(
        transport=center_transport
    )
    distributed_out = distributed_routed * block._value_modulation_gain(
        transport=distributed_transport
    )

    torch.testing.assert_close(center_routed, distributed_routed)
    assert not torch.allclose(center_out, distributed_out)


def test_transport_scorer_uses_3x3_query_projection_without_default_edge_bias() -> None:
    scorer = pfs.LocalTransportScorer2d(channels=2, attention_channels=3)

    assert scorer.query_proj.kernel_size == (3, 3)
    assert scorer.query_proj.padding == (1, 1)
    assert scorer.key_proj.kernel_size == (3, 3)
    assert scorer.key_proj.padding == (1, 1)
    assert not scorer.use_edge_bias
    assert "edge_bias" not in dict(scorer.named_parameters())


def test_transport_scorer_optional_edge_bias_is_zero_initialized() -> None:
    scorer = pfs.LocalTransportScorer2d(
        channels=2,
        attention_channels=2,
        use_edge_bias=True,
    )

    assert scorer.use_edge_bias
    assert scorer.edge_bias.shape == (1, 9, 1, 1)
    torch.testing.assert_close(scorer.edge_bias, torch.zeros_like(scorer.edge_bias))


def test_disabled_edge_bias_gives_uniform_center_probs_when_qk_is_zero() -> None:
    scorer = pfs.LocalTransportScorer2d(
        channels=2,
        attention_channels=2,
    )
    with torch.no_grad():
        scorer.query_proj.weight.zero_()
        assert scorer.query_proj.bias is not None
        scorer.query_proj.bias.zero_()
        scorer.key_proj.weight.zero_()
        assert scorer.key_proj.bias is not None
        scorer.key_proj.bias.zero_()
    source = torch.randn(1, 2, 3, 3)
    query = torch.randn(1, 2, 3, 3)
    lookup = fine_to_coarse_lookup(
        fine_h=3,
        fine_w=3,
        coarse_h=3,
        coarse_w=3,
        device=torch.device("cpu"),
    )

    transport = scorer(source=source, query=query, lookup=lookup, pixel_stride=1)

    torch.testing.assert_close(
        transport.probs[:, :, 1, 1],
        torch.full_like(transport.probs[:, :, 1, 1], 1.0 / 9.0),
    )


def test_enabled_edge_bias_changes_logits_before_routing_when_qk_is_zero() -> None:
    scorer = pfs.LocalTransportScorer2d(
        channels=2,
        attention_channels=2,
        use_edge_bias=True,
    )
    with torch.no_grad():
        scorer.query_proj.weight.zero_()
        assert scorer.query_proj.bias is not None
        scorer.query_proj.bias.zero_()
        scorer.key_proj.weight.zero_()
        assert scorer.key_proj.bias is not None
        scorer.key_proj.bias.zero_()
        scorer.edge_bias.copy_(torch.arange(9, dtype=torch.float32).view(1, 9, 1, 1))
    source = torch.randn(1, 2, 3, 3)
    query = torch.randn(1, 2, 3, 3)
    lookup = fine_to_coarse_lookup(
        fine_h=3,
        fine_w=3,
        coarse_h=3,
        coarse_w=3,
        device=torch.device("cpu"),
    )

    transport = scorer(source=source, query=query, lookup=lookup, pixel_stride=1)
    uniform_transport = pfs._transport_from_logits(
        torch.zeros_like(transport.logits),
        lookup=lookup,
        pixel_stride=1,
    )

    torch.testing.assert_close(
        transport.logits[:, :, 1, 1], scorer.edge_bias.view(1, 9)
    )
    assert not torch.allclose(transport.logits, uniform_transport.logits)
    assert not torch.allclose(transport.probs, uniform_transport.probs)


def test_conv_transport_predictor_creates_conv_scorers_on_decoder_layers() -> None:
    decoder = _tiny_decoder(transport_predictor="conv")

    assert decoder.transport_predictor == "conv"
    assert all(
        isinstance(cast(Any, layer).transport_scorer, pfs.ConvTransportScorer2d)
        for layer in decoder.layers
    )


def test_conv_transport_predictor_rejects_attention_edge_bias() -> None:
    with pytest.raises(ValueError, match="use_edge_bias"):
        PyramidFlowDecoder2d(
            in_channels=2,
            out_channels=1,
            channels=(2,),
            strides=(),
            feature_stride=1,
            stage_blocks=(1,),
            transport_predictor="conv",
            use_edge_bias=True,
        )


def test_conv_transport_scorer_masks_and_normalizes_local_edges() -> None:
    scorer = pfs.ConvTransportScorer2d(channels=2)
    lookup = fine_to_coarse_lookup(
        fine_h=4,
        fine_w=4,
        coarse_h=2,
        coarse_w=2,
        device=torch.device("cpu"),
    )
    source = torch.randn(1, 2, 2, 2)
    query = torch.randn(1, 2, 4, 4)

    transport = scorer(source=source, query=query, lookup=lookup, pixel_stride=1)

    assert transport.probs.shape == (1, 9, 4, 4)
    assert transport.logits.shape == (1, 9, 4, 4)
    torch.testing.assert_close(
        transport.probs.sum(dim=1),
        torch.ones(1, 4, 4),
        rtol=1e-5,
        atol=1e-6,
    )
    valid = lookup.valid_edges.to(device=transport.probs.device)
    probs_flat = rearrange(transport.probs, "b k h w -> b (h w) k")
    assert probs_flat.masked_select(~valid.unsqueeze(0)).abs().sum() == 0.0


def test_conv_transport_scorer_ignores_source_when_query_is_fixed() -> None:
    scorer = pfs.ConvTransportScorer2d(channels=2)
    lookup = fine_to_coarse_lookup(
        fine_h=3,
        fine_w=3,
        coarse_h=3,
        coarse_w=3,
        device=torch.device("cpu"),
    )
    query = torch.randn(1, 2, 3, 3)
    source_1 = torch.randn(1, 2, 3, 3)
    source_2 = torch.randn(1, 2, 3, 3) * 100.0 + 50.0

    transport_1 = scorer(source=source_1, query=query, lookup=lookup, pixel_stride=1)
    transport_2 = scorer(source=source_2, query=query, lookup=lookup, pixel_stride=1)

    torch.testing.assert_close(transport_1.logits, transport_2.logits)
    torch.testing.assert_close(transport_1.probs, transport_2.probs)


def test_conv_decoder_layer_passes_foreground_gated_query_to_scorer() -> None:
    layer = pfs.PyramidFlowDecoderLayer2d(
        in_channels=2,
        out_channels=2,
        stride=2,
        pixel_stride=1,
        attention_channels=2,
        use_edge_bias=False,
        ffn_expansion=1,
        transport_predictor="conv",
    )
    x = torch.randn(1, 2, 2, 3)
    foreground_presence = torch.tensor([[[[0.0, 0.25, 1.0], [0.5, 0.75, 0.0]]]])
    captured: dict[str, torch.Tensor] = {}
    original_forward = layer.transport_scorer.forward

    def capture_forward(
        _module: torch.nn.Module,
        *,
        source: torch.Tensor,
        query: torch.Tensor,
        lookup: pfs.FineToCoarseLookup,
        pixel_stride: int,
    ) -> PyramidTransport:
        captured["source"] = source.detach()
        captured["query"] = query.detach()
        return original_forward(
            source=source, query=query, lookup=lookup, pixel_stride=pixel_stride
        )

    cast(Any, layer.transport_scorer).forward = types.MethodType(
        capture_forward, layer.transport_scorer
    )

    layer(x=x, foreground_presence=foreground_presence, transport=None)

    expected_source = x * foreground_presence
    expected_query = torch.nn.functional.interpolate(
        x, size=(4, 6), mode="nearest"
    ) * torch.nn.functional.interpolate(
        foreground_presence, size=(4, 6), mode="nearest"
    )
    torch.testing.assert_close(captured["source"], expected_source)
    torch.testing.assert_close(captured["query"], expected_query)


def test_layer_transports_are_positive_row_normalized_and_mask_borders() -> None:
    model = _tiny_system()
    output = model(torch.randn(1, 1, 16, 16))

    for transport in output["layer_transports"]:
        probs = transport.probs
        row_sums = probs.sum(dim=1)
        torch.testing.assert_close(
            row_sums,
            torch.ones_like(row_sums),
            rtol=1e-5,
            atol=1e-6,
        )
        assert probs.amin() >= 0.0
        assert probs.amax() <= 1.0
        batch_size, _, fine_h, fine_w = probs.shape
        valid = transport.lookup.valid_edges.to(device=probs.device)
        probs_flat = probs.permute(0, 2, 3, 1).reshape(batch_size, fine_h * fine_w, 9)
        assert probs_flat.masked_select(~valid.unsqueeze(0)).abs().sum() == 0.0


def test_expected_flow_is_zero_for_same_resolution_center_transport() -> None:
    model = _one_stage_system()
    transport = _center_one_hot_transport(fine_hw=(3, 4), coarse_hw=(3, 4))

    flow = model._expected_flow(
        layer_transports=(transport,),
        output_hw=(3, 4),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    flow_l2, flow_l2_total = model._flow_l2_mean_and_total((transport,))

    torch.testing.assert_close(flow, torch.zeros_like(flow))
    torch.testing.assert_close(flow_l2, torch.zeros_like(flow_l2))
    torch.testing.assert_close(flow_l2_total, torch.zeros_like(flow_l2_total))


def test_flow_l2_penalizes_deterministic_one_pixel_step() -> None:
    model = _one_stage_system()
    transport = _one_hot_transport_from_edges(
        fine_hw=(1, 2),
        coarse_hw=(1, 2),
        edges={(0, 0): (0, 1)},
    )

    flow_l2, flow_l2_total = model._flow_l2_mean_and_total((transport,))

    torch.testing.assert_close(flow_l2, torch.tensor(0.5))
    torch.testing.assert_close(flow_l2_total, torch.tensor(1.0))


def test_flow_l2_penalizes_symmetric_diffuse_step_with_zero_expected_flow() -> None:
    model = _one_stage_system()
    lookup = fine_to_coarse_lookup(
        fine_h=1,
        fine_w=3,
        coarse_h=1,
        coarse_w=3,
        device=torch.device("cpu"),
    )
    center_index = _edge_index(lookup, dy=0, dx=0)
    left_index = _edge_index(lookup, dy=0, dx=-1)
    right_index = _edge_index(lookup, dy=0, dx=1)
    probs = torch.zeros(1, 9, 1, 3)
    probs[:, center_index] = 1.0
    probs[:, :, 0, 1] = 0.0
    probs[:, left_index, 0, 1] = 0.5
    probs[:, right_index, 0, 1] = 0.5
    transport = PyramidTransport(
        probs=probs,
        logits=probs,
        expected_offset=pfs._expected_local_offset(probs, lookup=lookup),
        lookup=lookup,
        pixel_stride=1,
    )

    flow = model._expected_flow(
        layer_transports=(transport,),
        output_hw=(1, 3),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    flow_l2, flow_l2_total = model._flow_l2_mean_and_total((transport,))

    torch.testing.assert_close(flow, torch.zeros_like(flow))
    torch.testing.assert_close(flow_l2, torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(flow_l2_total, torch.tensor(1.0))


def test_flow_l2_penalizes_backtracking_when_terminal_flow_cancels() -> None:
    model = _one_stage_system()
    first = _one_hot_transport_from_edges(
        fine_hw=(1, 3),
        coarse_hw=(1, 3),
        edges={(0, 2): (0, -1)},
    )
    second = _one_hot_transport_from_edges(
        fine_hw=(1, 3),
        coarse_hw=(1, 3),
        edges={(0, 1): (0, 1)},
    )

    flow = model._expected_flow(
        layer_transports=(first, second),
        output_hw=(1, 3),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    flow_l2, flow_l2_total = model._flow_l2_mean_and_total((first, second))

    torch.testing.assert_close(flow[:, :, 0, 1], torch.zeros(1, 2))
    torch.testing.assert_close(flow_l2, torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(flow_l2_total, torch.tensor(2.0))


def test_flow_l2_uses_exact_upsampling_subcell_geometry() -> None:
    model = _one_stage_system()
    transport = _center_one_hot_transport(
        fine_hw=(2, 2),
        coarse_hw=(1, 1),
        pixel_stride=4,
    )

    step_l2 = model._transport_step_squared_distance(transport)
    flow_l2, flow_l2_total = model._flow_l2_mean_and_total((transport,))

    torch.testing.assert_close(step_l2[:, 4], torch.full((1, 2, 2), 8.0))
    torch.testing.assert_close(flow_l2, torch.tensor(8.0))
    torch.testing.assert_close(flow_l2_total, torch.tensor(32.0))


def test_dense_assignment_matches_projected_pixel_foreground() -> None:
    model = _tiny_system()
    image = torch.randn(1, 1, 16, 16)

    output = model(image, return_dense_assignment=True)
    assignment = output["dense_assignment"]
    assert assignment is not None
    dense_p_fg = (
        assignment
        * output["foreground_presence"].float().flatten(start_dim=1).view(1, -1, 1, 1)
    ).sum(dim=1, keepdim=True)

    torch.testing.assert_close(
        assignment.sum(dim=1),
        torch.ones(1, 16, 16),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(dense_p_fg, output["p_fg"].float(), rtol=1e-5, atol=1e-5)


def test_foreground_sparsity_prefers_concentrated_mass() -> None:
    model = _tiny_system()
    concentrated = torch.tensor([[[[4.0, 0.0], [0.0, 0.0]]]])
    diffuse = torch.ones(1, 1, 2, 2)

    concentrated_loss = model.foreground_sparsity_loss(concentrated)
    diffuse_loss = model.foreground_sparsity_loss(diffuse)

    assert concentrated_loss < diffuse_loss


def test_kl_standard_normal_matches_closed_form_formula() -> None:
    mu = torch.tensor([[[[0.0]], [[1.0]]]])
    logvar = torch.tensor([[[[0.0]], [[-0.5]]]])

    kl = PyramidFlowSystem._kl_standard_normal(mu, logvar)
    expected = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar)

    torch.testing.assert_close(kl, expected)


def test_split_latent_kl_matches_closed_form_formula() -> None:
    foreground_mu = torch.tensor([[[[0.0]], [[1.0]]]])
    foreground_logvar = torch.tensor([[[[0.0]], [[-0.5]]]])
    background_mu = torch.tensor([[[[-1.0]], [[0.25]], [[0.5]]]])
    background_logvar = torch.tensor([[[[0.25]], [[0.0]], [[-1.0]]]])

    foreground_kl, foreground_kl_total = PyramidFlowSystem._latent_kl_mean_and_total(
        foreground_mu,
        foreground_logvar,
    )
    background_kl, background_kl_total = PyramidFlowSystem._latent_kl_mean_and_total(
        background_mu,
        background_logvar,
    )

    foreground_expected = PyramidFlowSystem._kl_standard_normal(
        foreground_mu,
        foreground_logvar,
    )
    background_expected = PyramidFlowSystem._kl_standard_normal(
        background_mu,
        background_logvar,
    )
    torch.testing.assert_close(foreground_kl, foreground_expected.mean())
    torch.testing.assert_close(
        foreground_kl_total,
        foreground_expected.sum(dim=(1, 2, 3)).mean(),
    )
    torch.testing.assert_close(background_kl, background_expected.mean())
    torch.testing.assert_close(
        background_kl_total,
        background_expected.sum(dim=(1, 2, 3)).mean(),
    )


def test_train_mode_samples_split_latents_and_eval_mode_uses_posterior_means() -> None:
    model = _tiny_system()
    image = torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        for head in (model.foreground_latent_head, model.background_latent_head):
            output_projection = cast(torch.nn.Conv2d, head[-1])
            output_projection.weight.zero_()
            assert output_projection.bias is not None
            output_projection.bias.zero_()

    model.train()
    train_output_1 = model(image)
    train_output_2 = model(image)

    assert (
        train_output_1["foreground_latents"].shape[-2:]
        == train_output_1["background_latents"].shape[-2:]
    )
    assert not torch.allclose(
        train_output_1["foreground_latents"],
        train_output_1["foreground_latent_mu"],
    )
    assert not torch.allclose(
        train_output_1["background_latents"],
        train_output_1["background_latent_mu"],
    )
    assert not torch.allclose(
        train_output_1["foreground_latents"],
        train_output_2["foreground_latents"],
    )
    assert not torch.allclose(
        train_output_1["background_latents"],
        train_output_2["background_latents"],
    )

    model.eval()
    eval_output_1 = model(image)
    eval_output_2 = model(image)

    torch.testing.assert_close(
        eval_output_1["foreground_latents"],
        eval_output_1["foreground_latent_mu"],
    )
    torch.testing.assert_close(
        eval_output_1["background_latents"],
        eval_output_1["background_latent_mu"],
    )
    torch.testing.assert_close(
        eval_output_1["foreground_latents"],
        eval_output_2["foreground_latents"],
    )
    torch.testing.assert_close(
        eval_output_1["background_latents"],
        eval_output_2["background_latents"],
    )
    torch.testing.assert_close(eval_output_1["recon"], eval_output_2["recon"])


def test_latent_heads_are_pointwise_mlps_with_branch_specific_hidden_widths() -> None:
    model = _tiny_system()

    for head, hidden_dim in (
        (model.foreground_latent_head, FOREGROUND_LATENT_HIDDEN_DIM),
        (model.background_latent_head, BACKGROUND_LATENT_HIDDEN_DIM),
    ):
        assert isinstance(head, PointwiseMLP2d)
        input_projection = cast(torch.nn.Conv2d, head[0])
        output_projection = cast(torch.nn.Conv2d, head[3])
        assert input_projection.in_channels == model.encoder_dim
        assert input_projection.out_channels == hidden_dim
        assert input_projection.kernel_size == (1, 1)
        assert isinstance(head[1], torch.nn.GroupNorm)
        assert isinstance(head[2], torch.nn.GELU)
        assert output_projection.in_channels == hidden_dim
        assert output_projection.out_channels == 2 * LATENT_DIM
        assert output_projection.kernel_size == (1, 1)
        assert not any(
            isinstance(module, (torch.nn.AvgPool2d, torch.nn.Upsample))
            for module in head.modules()
        )


def test_latent_heads_split_mean_before_log_variance() -> None:
    model = _tiny_system()
    image = torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        for head in (model.foreground_latent_head, model.background_latent_head):
            output_projection = cast(torch.nn.Conv2d, head[-1])
            output_projection.weight.zero_()
            assert output_projection.bias is not None
            output_projection.bias[:LATENT_DIM].fill_(1.25)
            output_projection.bias[LATENT_DIM:].fill_(-0.75)

    model.eval()
    output = model(image)

    for latent_mu, latent_logvar in (
        (output["foreground_latent_mu"], output["foreground_latent_logvar"]),
        (output["background_latent_mu"], output["background_latent_logvar"]),
    ):
        torch.testing.assert_close(
            latent_mu,
            torch.full_like(latent_mu, 1.25),
        )
        torch.testing.assert_close(
            latent_logvar,
            torch.full_like(latent_logvar, -0.75),
        )


def test_latent_heads_receive_the_same_encoder_feature_grid() -> None:
    model = _tiny_system()
    image = torch.randn(1, 1, 16, 16)
    captured: dict[str, torch.Tensor] = {}

    def capture_head_input(
        name: str,
        _module: torch.nn.Module,
        args: tuple[object, ...],
    ) -> None:
        captured[name] = cast(torch.Tensor, args[0]).detach()

    foreground_handle = model.foreground_latent_head.register_forward_pre_hook(
        lambda module, args: capture_head_input("foreground", module, args)
    )
    background_handle = model.background_latent_head.register_forward_pre_hook(
        lambda module, args: capture_head_input("background", module, args)
    )
    model.eval()
    _output = model(image)
    foreground_handle.remove()
    background_handle.remove()

    encoder_output = cast(PyramidFlowEncoderOutput, model.encoder(image))
    torch.testing.assert_close(captured["foreground"], encoder_output.features)
    torch.testing.assert_close(captured["background"], encoder_output.features)


def test_reconstruction_loss_gradients_reach_latent_head_paths() -> None:
    model = _tiny_system()
    image = torch.randn(1, 1, 16, 16)
    output = model(image)
    loss = model._reconstruction_loss(output["recon"], image)

    loss.backward()

    encoder_has_grad = any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.encoder.parameters()
    )
    foreground_posterior_has_grad = any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.foreground_latent_head.parameters()
    )
    background_posterior_has_grad = any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.background_latent_head.parameters()
    )
    presence_head_has_grad = any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.foreground_presence_head.parameters()
    )
    transport_scorer_has_grad = any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for layer in model.decoder.layers
        for parameter in cast(Any, layer).transport_scorer.parameters()
    )
    decoder_block_has_grad = any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for layer in model.decoder.layers
        for parameter in layer.parameters()
    )
    assert torch.isfinite(loss)
    assert encoder_has_grad
    assert foreground_posterior_has_grad
    assert background_posterior_has_grad
    assert presence_head_has_grad
    assert transport_scorer_has_grad
    assert decoder_block_has_grad


def test_reconstruction_and_flow_loss_reach_conv_transport_predictor() -> None:
    model = _tiny_system(transport_predictor="conv")
    image = torch.randn(1, 1, 16, 16)
    output = model(image)
    loss = model._reconstruction_loss(output["recon"], image) + 0.1 * output["flow_l2"]

    loss.backward()

    conv_block_has_grad = any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for layer in model.decoder.layers
        for parameter in cast(Any, layer).transport_scorer.block.parameters()
    )
    logits_head_has_grad = any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for layer in model.decoder.layers
        for parameter in cast(Any, layer).transport_scorer.logits_head.parameters()
    )
    assert torch.isfinite(loss)
    assert conv_block_has_grad
    assert logits_head_has_grad


def test_reconstruction_loss_gradients_reach_value_modulation_projections() -> None:
    model = _tiny_system(transport_predictor="conv", value_modulation=True)
    image = torch.randn(1, 1, 16, 16)
    output = model(image)
    loss = model._reconstruction_loss(output["recon"], image)

    loss.backward()

    projections = [
        cast(Any, layer).block.value_modulation_proj for layer in model.decoder.layers
    ]
    assert torch.isfinite(loss)
    assert all(projection is not None for projection in projections)
    assert all(
        projection.weight.grad is not None
        and torch.isfinite(projection.weight.grad).all()
        and projection.weight.grad.abs().sum() > 0
        for projection in projections
    )


def test_reconstruction_kl_flow_and_sparsity_are_differentiable() -> None:
    model = _tiny_system()
    image = torch.randn(1, 1, 16, 16)
    output = model(image)
    terms = model._loss_terms(image=image, output=output)
    loss = (
        terms["recon"]
        + 0.1 * terms["foreground_kl"]
        + 0.1 * terms["background_kl"]
        + 0.1 * terms["flow_l2"]
        + 0.1 * terms["foreground_sparsity"]
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert terms["foreground_kl"].dtype == torch.float32
    assert terms["background_kl"].dtype == torch.float32
    assert terms["flow_l2"].dtype == torch.float32


def test_cyclic_transport_shuffle_preserves_valid_transport_structure() -> None:
    transports = (
        _gather_test_transport(
            batch_size=3,
            fine_hw=(4, 4),
            coarse_hw=(2, 2),
            pixel_stride=2,
        ),
        _gather_test_transport(
            batch_size=3,
            fine_hw=(4, 4),
            coarse_hw=(4, 4),
        ),
    )
    original_tensors = tuple(
        (
            transport.probs.clone(),
            transport.logits.clone(),
            transport.expected_offset.clone(),
        )
        for transport in transports
    )

    shuffled = PyramidFlowSystem._cyclically_shuffle_layer_transports(transports)

    for transport, shuffled_transport, original in zip(
        transports, shuffled, original_tensors, strict=True
    ):
        assert shuffled_transport.lookup is transport.lookup
        assert shuffled_transport.pixel_stride == transport.pixel_stride
        torch.testing.assert_close(
            shuffled_transport.probs, torch.roll(transport.probs, shifts=1, dims=0)
        )
        torch.testing.assert_close(
            shuffled_transport.logits, torch.roll(transport.logits, shifts=1, dims=0)
        )
        torch.testing.assert_close(
            shuffled_transport.expected_offset,
            torch.roll(transport.expected_offset, shifts=1, dims=0),
        )
        valid = transport.lookup.valid_edges.T.reshape(1, 9, *transport.lookup.fine_hw)
        assert torch.count_nonzero(shuffled_transport.probs.masked_select(~valid)) == 0
        torch.testing.assert_close(
            shuffled_transport.probs.sum(dim=1),
            torch.ones_like(shuffled_transport.probs[:, 0]),
        )
        assert torch.isfinite(shuffled_transport.probs).all()
        assert torch.isfinite(shuffled_transport.logits).all()
        assert torch.isfinite(shuffled_transport.expected_offset).all()
        torch.testing.assert_close(transport.probs, original[0])
        torch.testing.assert_close(transport.logits, original[1])
        torch.testing.assert_close(transport.expected_offset, original[2])


@pytest.mark.parametrize(
    ("dual_stream", "background_latent_pool_stride"),
    [(False, 1), (True, 1), (True, 2)],
)
def test_causal_decoder_ablation_metrics_match_explicit_interventions(
    dual_stream: bool,
    background_latent_pool_stride: int,
) -> None:
    torch.manual_seed(17)
    model = _tiny_system(
        stage_blocks=(1, 1, 1, 1),
        dual_stream=dual_stream,
        background_latent_pool_stride=background_latent_pool_stride,
    ).eval()
    image = torch.randn(2, 1, 16, 16)
    with torch.no_grad():
        output = model(image)
    original_presence = output["foreground_presence"].clone()
    supplied_transports: list[tuple[PyramidTransport, ...] | None] = []

    def capture_decoder_call(module, args, kwargs):
        del module, args
        supplied_transports.append(kwargs.get("layer_transports"))

    hook = model.decoder.register_forward_pre_hook(
        capture_decoder_call, with_kwargs=True
    )
    metrics = model.decoder_ablation_metrics(image=image, output=output)
    hook.remove()

    assert supplied_transports[:3] == [None, None, None]
    assert supplied_transports[3] is not None
    assert all(transports is not None for transports in supplied_transports[3:])
    torch.testing.assert_close(output["foreground_presence"], original_presence)

    foreground = output["foreground_features"].detach()
    background = output["background_features"].detach()
    presence = output["foreground_presence"].detach().to(dtype=background.dtype)
    shuffled_presence = torch.roll(presence, shifts=1, dims=0)
    shuffled_transports = model._cyclically_shuffle_layer_transports(
        output["layer_transports"]
    )
    with torch.no_grad():
        interventions = {
            "zero_foreground_recomputed_transport": model.decoder(
                torch.zeros_like(foreground), background, presence
            )[0],
            "zero_background_recomputed_transport": model.decoder(
                foreground, torch.zeros_like(background), presence
            )[0],
            "shuffled_presence_recomputed_transport": model.decoder(
                foreground, background, shuffled_presence
            )[0],
            "shuffled_transport": model.decoder(
                foreground,
                background,
                presence,
                layer_transports=shuffled_transports,
            )[0],
        }
    target = image.float()
    recon = output["recon"].detach().float()
    for name, intervention_recon in interventions.items():
        expected_mse = (intervention_recon.float() - target).square().mean()
        expected_delta_mse = (intervention_recon.float() - recon).square().mean()
        torch.testing.assert_close(metrics[f"{name}_mse"], expected_mse)
        torch.testing.assert_close(metrics[f"{name}_delta_mse"], expected_delta_mse)
    torch.testing.assert_close(
        metrics["zero_latents_presence_fixed_transport_mse"],
        metrics["zero_latents_zero_presence_fixed_transport_mse"],
    )
    assert metrics["zero_latents_presence_fixed_transport_spatial_std"] == 0
    assert metrics["zero_latents_zero_presence_fixed_transport_spatial_std"] == 0
    assert metrics["zero_latents_fixed_vs_uniform_transport_delta_mse"] == 0
    assert all(not metric.requires_grad for metric in metrics.values())


def test_decoder_ablation_diagnostics_require_two_samples() -> None:
    with pytest.raises(ValueError, match="must be at least 2"):
        _tiny_system(
            log_decoder_ablation_every_n_epochs=1,
            log_decoder_ablation_max_batch_size=1,
        )

    model = _tiny_system().eval()
    image = torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        output = model(image)
    with pytest.raises(ValueError, match="require at least 2 samples"):
        model.decoder_ablation_metrics(image=image, output=output)


def test_tiny_training_step_returns_finite_loss_and_logs_terms() -> None:
    model = _tiny_system(
        dual_stream=True,
        background_latent_pool_stride=2,
        log_decoder_ablation_every_n_epochs=1,
        log_decoder_ablation_max_batch_size=2,
    )
    logged: dict[str, torch.Tensor] = {}

    def capture_log(self, name, value, *args, **kwargs):
        del self, args, kwargs
        logged[name] = torch.as_tensor(value).detach()

    model.log = types.MethodType(capture_log, model)
    batch = _single_frame_batch(torch.randn(2, 1, 16, 16))

    loss = model.training_step(batch, batch_idx=0)

    assert torch.isfinite(loss)
    assert "loss/train_recon" in logged
    assert "loss/train_foreground_kl" in logged
    assert "loss/train_background_kl" in logged
    assert "loss/train_flow_l2" in logged
    assert "loss/train_foreground_sparsity" in logged
    assert "stats/foreground_kl_total_mean_train" in logged
    assert "stats/background_kl_total_mean_train" in logged
    assert "stats/flow_l2_total_mean_train" in logged
    assert "stats/foreground_kl_weight_train" in logged
    assert "stats/background_kl_weight_train" in logged
    assert "stats/flow_l2_weight_train" in logged
    assert "stats/foreground_sparsity_weight_train" in logged
    assert "stats/foreground_presence_mean_train" in logged
    diagnostic_names = {
        "recon_mse",
        "zero_foreground_recomputed_transport_mse",
        "zero_foreground_recomputed_transport_delta_mse",
        "zero_background_recomputed_transport_mse",
        "zero_background_recomputed_transport_delta_mse",
        "shuffled_presence_recomputed_transport_mse",
        "shuffled_presence_recomputed_transport_delta_mse",
        "shuffled_transport_mse",
        "shuffled_transport_delta_mse",
        "zero_foreground_fixed_transport_mse",
        "zero_foreground_fixed_transport_delta_mse",
        "zero_presence_fixed_transport_mse",
        "zero_presence_fixed_transport_delta_mse",
        "foreground_values_full_presence_fixed_transport_mse",
        "foreground_values_full_presence_fixed_transport_delta_mse",
        "zero_latents_presence_fixed_transport_mse",
        "zero_latents_zero_presence_fixed_transport_mse",
        "zero_latents_presence_fixed_transport_spatial_std",
        "zero_latents_zero_presence_fixed_transport_spatial_std",
        "zero_latents_fixed_vs_uniform_transport_delta_mse",
        "zero_presence_fixed_vs_uniform_transport_delta_mse",
    }
    assert {f"diagnostics/{name}_train" for name in diagnostic_names}.issubset(logged)
    assert "diagnostics/no_fg_mixture_mse_train" not in logged


def test_weighted_loss_includes_flow_term() -> None:
    model = _tiny_system()
    model.recon_loss_weight = LossWeightSchedule(0.0, 0.0, 0)
    model.foreground_kl_loss_weight = LossWeightSchedule(0.0, 0.0, 0)
    model.background_kl_loss_weight = LossWeightSchedule(0.0, 0.0, 0)
    model.flow_l2_loss_weight = LossWeightSchedule(2.0, 2.0, 0)
    model.foreground_sparsity_loss_weight = LossWeightSchedule(0.0, 0.0, 0)
    terms = {
        "recon": torch.tensor(1.0),
        "foreground_kl": torch.tensor(2.0),
        "background_kl": torch.tensor(3.0),
        "flow_l2": torch.tensor(4.0),
        "foreground_sparsity": torch.tensor(6.0),
    }

    loss = model._weighted_loss(terms)

    torch.testing.assert_close(loss, torch.tensor(8.0))


def test_flow_induced_instance_labels_propagates_center_components() -> None:
    model = _tiny_system(stage_blocks=(1, 1, 1, 1))
    center_presence = torch.tensor([[[[1.0, 0.0, 1.0]]]])
    transports = (
        _center_one_hot_transport(fine_hw=(1, 3), coarse_hw=(1, 3)),
        _center_one_hot_transport(fine_hw=(1, 3), coarse_hw=(1, 3)),
        _center_one_hot_transport(fine_hw=(1, 3), coarse_hw=(1, 3)),
        _center_one_hot_transport(fine_hw=(2, 6), coarse_hw=(1, 3)),
    )

    segmentation = model.flow_induced_instance_labels(
        center_presence=center_presence,
        layer_transports=transports,
        center_threshold=0.5,
        pixel_mass_threshold=0.5,
        min_object_area=None,
        max_object_area=None,
    )

    expected = torch.tensor(
        [[[1, 1, 0, 0, 2, 2], [1, 1, 0, 0, 2, 2]]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(segmentation.pred_labels, expected)


def test_flow_induced_component_chunks_match_single_component_reference() -> None:
    model = _tiny_system(stage_blocks=(1, 1, 1, 1))
    center_presence = torch.zeros(2, 1, 5, 7)
    center_presence[0, 0, 0, 0] = 0.6
    center_presence[0, 0, 0, 3] = 0.7
    center_presence[0, 0, 2, 1] = 0.8
    center_presence[0, 0, 4, 3] = 0.9
    center_presence[0, 0, 4, 6] = 1.0
    center_presence[1, 0, 0, 1] = 0.9
    center_presence[1, 0, 2, 3] = 0.8
    center_presence[1, 0, 4, 5] = 0.7
    transports = (
        _gather_test_transport(
            batch_size=2,
            fine_hw=(5, 7),
            coarse_hw=(5, 7),
        ),
    )

    reference = model.flow_induced_instance_labels(
        center_presence=center_presence,
        layer_transports=transports,
        center_threshold=0.5,
        pixel_mass_threshold=0.05,
        component_chunk_size=1,
    )
    chunked = model.flow_induced_instance_labels(
        center_presence=center_presence,
        layer_transports=transports,
        center_threshold=0.5,
        pixel_mass_threshold=0.05,
        component_chunk_size=16,
    )

    torch.testing.assert_close(chunked.center_components, reference.center_components)
    torch.testing.assert_close(
        chunked.pred_labels_unfiltered, reference.pred_labels_unfiltered
    )
    torch.testing.assert_close(chunked.pred_labels, reference.pred_labels)
    torch.testing.assert_close(chunked.winning_mass, reference.winning_mass)


def test_flow_induced_instance_labels_preserves_first_component_on_mass_tie() -> None:
    model = _tiny_system(stage_blocks=(1, 1, 1, 1))
    center_presence = torch.tensor([[[[1.0, 0.0, 1.0]]]])
    lookup = fine_to_coarse_lookup(
        fine_h=1,
        fine_w=3,
        coarse_h=1,
        coarse_w=3,
        device=torch.device("cpu"),
    )
    center_index = _edge_index(lookup, dy=0, dx=0)
    left_index = _edge_index(lookup, dy=0, dx=-1)
    right_index = _edge_index(lookup, dy=0, dx=1)
    probs = torch.zeros(1, 9, 1, 3)
    probs[:, center_index] = 1.0
    probs[:, center_index, 0, 1] = 0.0
    probs[:, left_index, 0, 1] = 0.5
    probs[:, right_index, 0, 1] = 0.5
    transport = PyramidTransport(
        probs=probs,
        logits=probs,
        expected_offset=pfs._expected_local_offset(probs, lookup=lookup),
        lookup=lookup,
        pixel_stride=1,
    )

    segmentation = model.flow_induced_instance_labels(
        center_presence=center_presence,
        layer_transports=(transport,),
        center_threshold=0.5,
        pixel_mass_threshold=0.5,
        component_chunk_size=1,
        min_object_area=None,
    )

    assert torch.equal(
        segmentation.pred_labels,
        torch.tensor([[[1, 1, 2]]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        segmentation.winning_mass,
        torch.tensor([[[1.0, 0.5, 1.0]]]),
    )


def test_flow_induced_instance_labels_handles_empty_centers_and_output_crop() -> None:
    model = _tiny_system(stage_blocks=(1, 1, 1, 1))
    center_presence = torch.zeros(1, 1, 1, 3)
    transport = _center_one_hot_transport(
        fine_hw=(2, 6),
        coarse_hw=(1, 3),
    )

    segmentation = model.flow_induced_instance_labels(
        center_presence=center_presence,
        layer_transports=(transport,),
        center_threshold=0.5,
        pixel_mass_threshold=0.5,
        output_hw=(1, 5),
    )

    assert segmentation.center_components.shape == (1, 1, 3)
    assert segmentation.pred_labels.shape == (1, 1, 5)
    assert segmentation.winning_mass.shape == (1, 1, 5)
    assert segmentation.pred_labels.dtype == torch.int32
    assert segmentation.winning_mass.dtype == torch.float32
    assert torch.count_nonzero(segmentation.pred_labels) == 0
    assert torch.count_nonzero(segmentation.winning_mass) == 0


def test_filter_label_tensor_by_area_compacts_retained_ids() -> None:
    labels = torch.tensor([[1, 1, 2, 3, 3, 3]], dtype=torch.int32)

    minimum_filtered = pfs._filter_label_tensor_by_area(
        labels,
        min_object_area=2,
        max_object_area=None,
    )
    maximum_filtered = pfs._filter_label_tensor_by_area(
        labels,
        min_object_area=None,
        max_object_area=2,
    )

    assert torch.equal(
        minimum_filtered,
        torch.tensor([[1, 1, 0, 2, 2, 2]], dtype=torch.int32),
    )
    assert torch.equal(
        maximum_filtered,
        torch.tensor([[1, 1, 2, 0, 0, 0]], dtype=torch.int32),
    )


def test_segmentation_config_rejects_nonpositive_component_chunk_size() -> None:
    with pytest.raises(ValueError, match="component_chunk_size must be positive"):
        _tiny_system(
            segmentation_test_config=PyramidFlowSegmentationTestConfig(
                component_chunk_size=0
            )
        )


def test_forward_with_model_padding_warns_and_crops_outputs() -> None:
    model = _tiny_system()
    image = torch.randn(1, 1, 18, 19)

    with pytest.warns(UserWarning, match="zero-padding"):
        output, original_hw, padded_hw = model._forward_with_model_padding(image)

    assert original_hw == (18, 19)
    assert padded_hw == (24, 24)
    assert output["recon"].shape[-2:] == (18, 19)
    assert output["p_fg"].shape[-2:] == (18, 19)
    assert output["expected_flow"].shape[-2:] == (18, 19)


def test_predict_step_returns_labels_and_metadata_without_ground_truth() -> None:
    model = _tiny_system(
        segmentation_test_config=PyramidFlowSegmentationTestConfig(
            prediction_output_scale_factor=2
        )
    )
    image = torch.randn(2, 1, 16, 16)
    batch = _single_frame_batch(image)
    batch["sequence_ids"] = ["first.ome.zarr", "second.ome.zarr"]
    batch["frame_numbers"] = torch.tensor([[3], [7]], dtype=torch.long)
    batch["filename_padding_width"] = torch.tensor([1, 1], dtype=torch.long)

    with torch.no_grad():
        prediction = model.predict_step(batch, batch_idx=0)

    assert prediction["sequence_ids"] == batch["sequence_ids"]
    assert torch.equal(prediction["frame_numbers"], torch.tensor([3, 7]))
    assert torch.equal(prediction["filename_padding_width"], torch.tensor([1, 1]))
    assert prediction["instance_labels"].shape == (2, 32, 32)
    assert prediction["instance_labels"].dtype == torch.int32
    assert torch.equal(
        prediction["instance_labels"][:, ::2, ::2],
        prediction["instance_labels"][:, 1::2, 1::2],
    )


@pytest.mark.parametrize("scale_factor", [1, 2, 0.5])
def test_prediction_label_scaling_preserves_ids_and_dtype(scale_factor: float) -> None:
    labels = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.int32)
    if scale_factor < 1:
        labels = labels.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)

    scaled = pfs._scale_label_images(labels, scale_factor=scale_factor)

    assert scaled.dtype == torch.int32
    expected = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.int32)
    if scale_factor > 1:
        expected = expected.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    assert torch.equal(scaled, expected)


def test_segmentation_config_rejects_non_integral_prediction_scale() -> None:
    with pytest.raises(ValueError, match="integer or the reciprocal"):
        _tiny_system(
            segmentation_test_config=PyramidFlowSegmentationTestConfig(
                prediction_output_scale_factor=1.5
            )
        )
