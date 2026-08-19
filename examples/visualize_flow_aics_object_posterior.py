# %%
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from torch import Tensor
from torch.nn import functional as F

from routing_pyramids._vendor.betterplots import set_style
from routing_pyramids.data.temporal_video_dataset import OMEZarrVideoDataset
from routing_pyramids.data.transforms import get_normalization, get_transforms
from routing_pyramids.pyramid_flow_system import (
    PyramidFlowDecoder2d,
    PyramidFlowEncoder2d,
    PyramidFlowSystem,
    PyramidTransport,
)
from routing_pyramids.visualization import pca_colorize

DATA_DIR = Path("data/aics")
CHECKPOINT = Path(
    "outputs/aics/pyramid_flow_vae/"
    "dim_256_8x8_64-fg_1e-2-bg_5e-2-flow_5e-3-entropy_0-sparsity_5e-1/"
    "checkpoints/last.ckpt"
)
OUTPUT_DIR = Path("outputs/aics_object_posterior")
STORE_NAME = "20200323_06_medium_mip.ome.zarr"
SAMPLE_INDEX: int | None = None
FRAME_NUMBER: int | None = 5
CROP_SIZE = 128
INPUT_SCALE_FACTOR = 0.5
FLOW_DISPLAY_MAX_MAGNITUDE: float | None = None
FORCE_CPU = False
set_style()
device = torch.device("cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda")


# %%
def build_system(*, checkpoint: Path, device: torch.device) -> PyramidFlowSystem:
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
        checkpoint,
        map_location="cpu",
        weights_only=False,
        encoder=encoder,
        decoder=decoder,
    )
    system.to(device)
    system.eval()
    return system


def select_sample_index(
    dataset: OMEZarrVideoDataset,
    *,
    sample_index: int | None,
    store_name: str,
    frame_number: int | None,
) -> int:
    if sample_index is not None:
        return int(sample_index)

    for candidate_index, (video_index, start) in enumerate(
        dataset.window_index.samples
    ):
        candidate_sequence_id = dataset.sequence_ids[video_index]
        candidate_frame_number = int(dataset.frame_numbers[video_index][start])
        if candidate_sequence_id != store_name:
            continue
        if frame_number is not None and candidate_frame_number != frame_number:
            continue
        return int(candidate_index)

    target = f"store_name={store_name}"
    if frame_number is not None:
        target = f"{target} frame_number={frame_number}"
    raise RuntimeError(f"No sample found for {target}.")


def normalize_for_display(image_2d: np.ndarray) -> np.ndarray:
    image_min = float(image_2d.min())
    image_max = float(image_2d.max())
    return (image_2d - image_min) / (image_max - image_min + 1e-8)


def flow_display_scale(flow_y: np.ndarray, flow_x: np.ndarray) -> float:
    if FLOW_DISPLAY_MAX_MAGNITUDE is not None:
        return float(FLOW_DISPLAY_MAX_MAGNITUDE)
    magnitude = np.sqrt(flow_y**2 + flow_x**2)
    return max(float(np.quantile(magnitude, 0.99)), 1e-8)


def flow_hsv_rgb(
    flow_y: np.ndarray, flow_x: np.ndarray, foreground_mass: np.ndarray, scale: float
) -> np.ndarray:
    """HSV-encode flow: hue=direction, saturation=magnitude, value=foreground mass."""
    magnitude = np.sqrt(flow_y**2 + flow_x**2)
    hue = np.mod(np.arctan2(flow_y, flow_x) / (2.0 * np.pi) + 1.0, 1.0)
    saturation = np.clip(magnitude / max(float(scale), 1e-8), 0.0, 1.0)
    value = np.clip(foreground_mass, 0.0, 1.0)
    hsv = np.stack((hue, saturation, value), axis=-1)
    return hsv_to_rgb(hsv).astype(np.float32)


def crop_to_original(
    tensor: Tensor, *, original_hw: tuple[int, int], padded_hw: tuple[int, int]
) -> Tensor:
    if original_hw == padded_hw:
        return tensor
    return tensor[..., : original_hw[0], : original_hw[1]]


def layer_flow_and_mass_maps(
    system: PyramidFlowSystem,
    *,
    center_presence: Tensor,
    layer_transports: tuple[PyramidTransport, ...],
    original_hw: tuple[int, int],
    padded_hw: tuple[int, int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Return full-crop local flow and propagated mass after every layer."""
    mass = center_presence
    flow_y_maps: list[np.ndarray] = []
    flow_x_maps: list[np.ndarray] = []
    mass_maps: list[np.ndarray] = []

    with torch.no_grad():
        for transport in layer_transports:
            mass = system._gather_scalar_map(mass, transport=transport)
            fine_h, fine_w = transport.lookup.fine_hw
            coarse_h, coarse_w = transport.lookup.coarse_hw
            coarse_coords = system._pixel_center_coordinates_2d(
                grid_hw=transport.lookup.coarse_hw,
                pixel_stride_y=transport.pixel_stride * (fine_h // coarse_h),
                pixel_stride_x=transport.pixel_stride * (fine_w // coarse_w),
                device=center_presence.device,
                dtype=center_presence.dtype,
            )
            gathered_coarse_coords = system._gather_coordinate_map(
                coarse_coords, transport=transport
            )
            fine_coords = system._pixel_center_coordinates_2d(
                grid_hw=transport.lookup.fine_hw,
                pixel_stride_y=transport.pixel_stride,
                pixel_stride_x=transport.pixel_stride,
                device=center_presence.device,
                dtype=center_presence.dtype,
            )
            local_flow = gathered_coarse_coords - fine_coords
            flow_at_output_resolution = F.interpolate(
                local_flow.float(), size=padded_hw, mode="nearest"
            )
            mass_at_output_resolution = F.interpolate(
                mass.float(), size=padded_hw, mode="nearest"
            )
            flow_at_output_resolution = crop_to_original(
                flow_at_output_resolution,
                original_hw=original_hw,
                padded_hw=padded_hw,
            )
            mass_at_output_resolution = crop_to_original(
                mass_at_output_resolution,
                original_hw=original_hw,
                padded_hw=padded_hw,
            )
            flow_y_maps.append(flow_at_output_resolution[0, 0].detach().cpu().numpy())
            flow_x_maps.append(flow_at_output_resolution[0, 1].detach().cpu().numpy())
            mass_maps.append(mass_at_output_resolution[0, 0].detach().cpu().numpy())

    return flow_y_maps, flow_x_maps, mass_maps


def decoder_layer_feature_rgb_maps(
    system: PyramidFlowSystem,
    *,
    foreground: Tensor,
    background: Tensor,
    foreground_presence: Tensor,
    layer_transports: tuple[PyramidTransport, ...],
    original_hw: tuple[int, int],
    padded_hw: tuple[int, int],
) -> list[np.ndarray]:
    """Replay the decoder and PCA-colorize its feature map after every layer."""
    decoder = system.decoder
    foreground_out = decoder.latent_blend_norm(foreground)
    background_out = (
        background if decoder.dual_stream else decoder.latent_blend_norm(background)
    )
    presence_out = foreground_presence.to(dtype=foreground_out.dtype)
    features = (
        presence_out * foreground_out
        if decoder.dual_stream
        else presence_out * foreground_out + (1.0 - presence_out) * background_out
    )
    feature_rgb_maps: list[np.ndarray] = []

    with torch.no_grad():
        for layer, transport in zip(decoder.layers, layer_transports, strict=True):
            features, presence_out, _ = cast(Any, layer)(
                x=features,
                foreground_presence=presence_out,
                transport=transport,
            )
            feature_rgb = pca_colorize(features)
            feature_rgb = F.interpolate(feature_rgb, size=padded_hw, mode="nearest")
            feature_rgb = crop_to_original(
                feature_rgb, original_hw=original_hw, padded_hw=padded_hw
            )
            feature_rgb_maps.append(
                feature_rgb[0].permute(1, 2, 0).detach().cpu().numpy()
            )

    return feature_rgb_maps


# %%
torch.set_float32_matmul_precision("high")
system = build_system(checkpoint=CHECKPOINT, device=device)
dataset = OMEZarrVideoDataset(
    root_dir=str(DATA_DIR),
    store_names=(STORE_NAME,),
    normalization=get_normalization(
        clip_quantile_low=0.001,
        clip_quantile_high=0.999,
        norm_quantile_low=0.50,
        norm_quantile_high=0.99,
    ),
    augmentations=get_transforms(is_train=False, crop_size=CROP_SIZE),
    sequence_length=1,
    temporal_source_length=1,
    scale_factor=INPUT_SCALE_FACTOR,
)
selected_sample_index = select_sample_index(
    dataset,
    sample_index=SAMPLE_INDEX,
    store_name=STORE_NAME,
    frame_number=FRAME_NUMBER,
)
sample = dataset[selected_sample_index]

image = sample["video"][0, 0].numpy()
image_t = sample["video"][0].unsqueeze(0).to(device)

with torch.no_grad():
    output, original_hw, padded_hw = system._forward_with_model_padding(image_t)

foreground_presence = output["foreground_presence"]
layer_transports = output["layer_transports"]
layer_flow_y, layer_flow_x, layer_mass = layer_flow_and_mass_maps(
    system,
    center_presence=foreground_presence,
    layer_transports=layer_transports,
    original_hw=original_hw,
    padded_hw=padded_hw,
)
layer_feature_rgb = decoder_layer_feature_rgb_maps(
    system,
    foreground=output["foreground_latents"],
    background=output["background_latents"],
    foreground_presence=foreground_presence,
    layer_transports=layer_transports,
    original_hw=original_hw,
    padded_hw=padded_hw,
)
recon = crop_to_original(output["recon"], original_hw=original_hw, padded_hw=padded_hw)
total_flow = crop_to_original(
    output["expected_flow"], original_hw=original_hw, padded_hw=padded_hw
)
total_mass = crop_to_original(
    output["p_fg"], original_hw=original_hw, padded_hw=padded_hw
)
recon_image = recon[0, 0].detach().cpu().float().numpy()
total_flow_y = total_flow[0, 0].detach().cpu().float().numpy()
total_flow_x = total_flow[0, 1].detach().cpu().float().numpy()
total_mass_image = total_mass[0, 0].detach().cpu().float().numpy()
flow_scale = flow_display_scale(
    np.stack([*layer_flow_y, total_flow_y]),
    np.stack([*layer_flow_x, total_flow_x]),
)

print(f"device={device}")
print(
    f"store_name={sample['sequence_id']} sample_index={selected_sample_index} "
    f"frame_number={int(sample['frame_numbers'][0])}"
)
print(
    f"crop_hw={original_hw} foreground_latents=posterior_mean "
    "background_latents=posterior_mean"
)
print(f"flow_display_max_magnitude={flow_scale:.4f}")

# %%
num_layers = len(layer_transports)
num_columns = num_layers + 2
fig, axes = plt.subplots(
    2,
    num_columns,
    figsize=(2.8 * num_columns, 6.0),
    constrained_layout=True,
    squeeze=False,
)

for layer_index, _transport in enumerate(layer_transports):
    column_index = layer_index
    axes[0, column_index].imshow(
        flow_hsv_rgb(
            layer_flow_y[layer_index],
            layer_flow_x[layer_index],
            layer_mass[layer_index],
            scale=flow_scale,
        )
    )
    axes[0, column_index].set_title(f"Layer {layer_index + 1}")
    axes[1, column_index].imshow(layer_feature_rgb[layer_index])

total_column = num_columns - 2
axes[0, total_column].imshow(
    flow_hsv_rgb(
        total_flow_y,
        total_flow_x,
        total_mass_image,
        scale=flow_scale,
    )
)
axes[0, total_column].set_title("Total flow")
axes[1, total_column].imshow(total_mass_image, cmap="magma", vmin=0.0, vmax=1.0)
axes[1, total_column].set_title("Total propagated mass")

axes[0, -1].imshow(normalize_for_display(recon_image), cmap="gray", vmin=0.0, vmax=1.0)
axes[0, -1].set_title("Reconstruction")
axes[1, -1].imshow(normalize_for_display(image), cmap="gray", vmin=0.0, vmax=1.0)
axes[1, -1].set_title("Target")

for ax in axes.flat:
    if ax.axison:
        ax.axis("off")

fig.text(0.001, 0.72, "Flow", rotation=90, ha="center", va="center")
fig.text(0.001, 0.28, "PCA", rotation=90, ha="center", va="center")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(
    OUTPUT_DIR / "pyramid_flow_aics_object_posterior.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
