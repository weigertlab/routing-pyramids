# %%
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from cmap import Colormap as CmapColormap
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap as MatplotlibColormap
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure, SubFigure
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from torch import Tensor
from torch.utils.data import DataLoader
from umap import UMAP

from routing_pyramids._vendor.betterplots import set_style
from routing_pyramids._vendor.betterplots.boxstripplot import boxstripplot
from routing_pyramids.data.temporal_datamodule import BBBC013VideoDataModule
from routing_pyramids.pyramid_flow_system import (
    PyramidFlowDecoder2d,
    PyramidFlowEncoder2d,
    PyramidFlowSystem,
)

DATA_DIR = Path("data/BBBC013_v1_images_converted")
PLATE_MAP = Path("data/BBBC013_platemap_long_nM.csv")
CHECKPOINT = Path(
    "outputs/bbbc013/pyramid_flow_vae/"
    "dim_256_8x8_64-fg_1e-2-bg_5e-2-flow_5e-3-entropy_0-sparsity_5e-1/"
    "checkpoints/last.ckpt"
)
OUTPUT_DIR = Path("outputs/bbbc013_object_latents")
TRAIN_COLUMNS = tuple(range(2, 12))
VALIDATION_COLUMNS = (1, 12)
BATCH_SIZE = 8
NUM_WORKERS = 0
CROP_SIZE = 256
CENTER_THRESHOLD = 0.5
PIXEL_MASS_THRESHOLD = 0.1
COMPONENT_CHUNK_SIZE = 16
MIN_OBJECT_AREA_MODEL_PIXELS: int | None = 50
MAX_OBJECT_AREA_MODEL_PIXELS: int | None = None
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
GMM_NUM_COMPONENTS = 2
GMM_SAMPLES_PER_COMPONENT = 4
GMM_REG_COVAR = 1e-6
GMM_DENSITY_GRID_SIZE = 200
GMM_DENSITY_MASSES = (0.50, 0.80, 0.95)
GMM_PUBLICATION_CONTOUR_MASS = 0.95
GMM_DENSITY_COLORMAPS = ("gray", "bop_purple")
GMM_COMPONENT_COLOR_POSITIONS = (0.0, 1.0)
CHANNEL_DISPLAY_COLORMAPS = ("green", "magenta")
NEUTRAL_COLORMAP = CmapColormap("gray").to_mpl()
CONTROL_ROLES = ("negative_control", "positive_control")
GENERATED_OBJECT_CANVAS_GRID_SIZE = 16
GENERATED_OBJECT_PATCH_SIZE = 32
GENERATED_DISPLAY_QUANTILES = (0.01, 0.99)
PUBLICATION_FIGURE_PATH = OUTPUT_DIR / "bbbc013_gmm_latents.pdf"
PUBLICATION_IMAGE_FIGURE_PATH = OUTPUT_DIR / "bbbc013_gmm_images.pdf"
RANDOM_STATE = 42
FORCE_CPU = False
FIGURE_WIDTH = 4.8
CONCENTRATION_DISPLAY_UNITS = {
    "LY294002": ("µM", 1000.0),
    "Wortmannin": ("nM", 1.0),
}

device = torch.device("cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda")
print(f"Using device: {device}")
set_style(
    usetex=False,
    serif=True,
    font_size=8,
    legend_font_size=6,
    label_size=8,
    tick_size=7,
)


# %%
@dataclass(frozen=True)
class ObjectLatent:
    """One segmented object's posterior-mean latent and metadata."""

    condition: str
    object_id: int
    pixel_area: int
    center_grid_row: float
    center_grid_col: float
    latent: NDArray[np.float32]


@dataclass(frozen=True)
class WellCondition:
    """Compound metadata for one BBBC013 well."""

    compound: str
    concentration_nm: float
    role: str


@dataclass(frozen=True)
class CompoundColorScale:
    """Colormap and concentration normalization for one compound."""

    colormap: MatplotlibColormap
    norm: SymLogNorm
    concentrations: tuple[float, ...]
    unit: str
    nanomolar_per_unit: float


def load_plate_map(path: Path) -> dict[str, WellCondition]:
    """Load BBBC013 well metadata keyed by image filename."""
    plate_map: dict[str, WellCondition] = {}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        expected_fields = {
            "row_name",
            "column_name",
            "compound",
            "concentration",
            "role",
        }
        if set(reader.fieldnames or ()) != expected_fields:
            raise ValueError(
                f"Expected plate-map fields {sorted(expected_fields)}, "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            condition = f"{row['row_name']}{int(row['column_name'])}.tif"
            if condition in plate_map:
                raise ValueError(f"Duplicate plate-map entry for {condition}")
            plate_map[condition] = WellCondition(
                compound=row["compound"],
                concentration_nm=float(row["concentration"]),
                role=row["role"],
            )
    return plate_map


def object_conditions(
    objects: list[ObjectLatent], plate_map: dict[str, WellCondition]
) -> list[WellCondition]:
    """Look up plate-map metadata for every segmented object."""
    missing = sorted(
        {obj.condition for obj in objects if obj.condition not in plate_map}
    )
    if missing:
        raise KeyError(f"Conditions absent from plate map: {missing}")
    return [plate_map[obj.condition] for obj in objects]


def build_system(*, checkpoint: Path, device: torch.device) -> PyramidFlowSystem:
    """Load the architecture and weights used by the BBBC013 training script."""
    latent_dim = 64
    encoder_channels = (32, 64, 128, 256)
    encoder_strides = (2, 2, 2)
    encoder = PyramidFlowEncoder2d(
        in_channels=2,
        channels=encoder_channels,
        strides=encoder_strides,
        down_blocks=(2, 2, 2, 2),
        norm="GROUP",
    )
    decoder = PyramidFlowDecoder2d(
        in_channels=latent_dim,
        out_channels=2,
        channels=(latent_dim, *tuple(reversed(encoder_channels[:-1]))),
        strides=tuple(reversed(encoder_strides)),
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


def build_evaluation_data(
    *,
    columns: tuple[int, ...],
    device: torch.device,
) -> BBBC013VideoDataModule:
    """Build a data module that applies deterministic evaluation transforms."""
    selected_columns = set(columns)
    other_columns = tuple(
        column for column in range(1, 13) if column not in selected_columns
    )
    data = BBBC013VideoDataModule(
        data_dir=str(DATA_DIR),
        train_columns=other_columns,
        eval_columns=columns,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        crop_size=CROP_SIZE,
        train_repeat_factor=1,
        clip_quantile_low=0.05,
        clip_quantile_high=0.999,
        norm_quantile_low=0.50,
        norm_quantile_high=0.99,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    data.setup("test")
    return data


def retained_component_ids(
    labels: Tensor, *, min_area: int | None, max_area: int | None
) -> list[tuple[int, int]]:
    """Return original component IDs and areas passing the area bounds."""
    counts = torch.bincount(labels.to(dtype=torch.long).reshape(-1))
    retained: list[tuple[int, int]] = []
    for component_id in range(1, int(counts.numel())):
        area = int(counts[component_id].item())
        if area == 0:
            continue
        if min_area is not None and area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        retained.append((component_id, area))
    return retained


def pool_object_latents(
    *,
    condition: str,
    center_components: Tensor,
    pred_labels: Tensor,
    center_presence: Tensor,
    latent_mu: Tensor,
    min_area: int | None,
    max_area: int | None,
) -> list[ObjectLatent]:
    """Presence-weight posterior means within each retained center component."""
    objects: list[ObjectLatent] = []
    for object_id, (component_id, pixel_area) in enumerate(
        retained_component_ids(pred_labels, min_area=min_area, max_area=max_area),
        start=1,
    ):
        component_mask = center_components == component_id
        weights = center_presence[component_mask].float()
        if weights.numel() == 0:
            raise RuntimeError(
                f"Condition {condition} has pixels assigned to center component "
                f"{component_id}, but that component is absent from the center grid"
            )
        weights = weights / weights.sum()
        component_latents = latent_mu[:, component_mask].float()
        pooled = (component_latents * weights.unsqueeze(0)).sum(dim=1)
        grid_locations = component_mask.nonzero(as_tuple=False).float()
        center = (grid_locations * weights.unsqueeze(1)).sum(dim=0)
        objects.append(
            ObjectLatent(
                condition=condition,
                object_id=object_id,
                pixel_area=pixel_area,
                center_grid_row=float(center[0].item()),
                center_grid_col=float(center[1].item()),
                latent=pooled.detach().cpu().numpy().astype(np.float32),
            )
        )
    return objects


def extract_object_latents(
    *,
    system: PyramidFlowSystem,
    loader: DataLoader,
    split: str,
    device: torch.device,
    center_threshold: float,
    pixel_mass_threshold: float,
    component_chunk_size: int,
    min_object_area: int | None,
    max_object_area: int | None,
) -> tuple[list[ObjectLatent], dict[str, NDArray[np.float32]]]:
    """Collect object latents and registered model-input images for one split."""
    objects: list[ObjectLatent] = []
    source_images: dict[str, NDArray[np.float32]] = {}
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            images = batch["video"][:, 0].to(device, non_blocking=True)
            conditions = batch.get("sequence_ids")
            if conditions is None:
                raise RuntimeError("BBBC013 evaluation batch has no sequence_ids")
            output, original_hw, _padded_hw = system._forward_with_model_padding(images)
            segmentation = system.flow_induced_instance_labels(
                center_presence=output["foreground_presence"],
                layer_transports=output["layer_transports"],
                center_threshold=center_threshold,
                pixel_mass_threshold=pixel_mass_threshold,
                component_chunk_size=component_chunk_size,
                output_hw=original_hw,
            )
            for sample_index, sequence_id in enumerate(conditions):
                condition = f"{sequence_id}.tif"
                source_images[condition] = (
                    images[sample_index].detach().cpu().float().numpy()
                )
                sample_objects = pool_object_latents(
                    condition=condition,
                    center_components=segmentation.center_components[sample_index],
                    pred_labels=segmentation.pred_labels_unfiltered[sample_index],
                    center_presence=output["foreground_presence"][sample_index, 0],
                    latent_mu=output["foreground_latent_mu"][sample_index],
                    min_area=min_object_area,
                    max_area=max_object_area,
                )
                objects.extend(sample_objects)
                print(f"{sequence_id}.tif: {len(sample_objects)} objects")
            print(f"Finished {split} batch {batch_index + 1}/{len(loader)}")
    if len(objects) < 2:
        raise RuntimeError(
            f"PCA and UMAP require at least two {split} objects, "
            f"but found {len(objects)}"
        )
    return objects, source_images


def object_latent_matrix(objects: list[ObjectLatent]) -> NDArray[np.float32]:
    """Stack object records into an object-by-latent-code matrix."""
    matrix = np.stack([obj.latent for obj in objects]).astype(np.float32)
    if matrix.ndim != 2:
        raise RuntimeError(f"Expected a 2D latent matrix, got shape {matrix.shape}")
    return matrix


def fit_pca_embedding(
    matrix: NDArray[np.float32],
) -> tuple[NDArray[np.float32], PCA]:
    """Fit a two-dimensional PCA embedding."""
    if matrix.shape[1] < 2:
        raise RuntimeError("PCA requires at least two latent dimensions")
    pca = PCA(n_components=2)
    return pca.fit_transform(matrix).astype(np.float32), pca


def fit_umap_embedding(
    matrix: NDArray[np.float32],
    *,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> tuple[NDArray[np.float32], int]:
    """Fit a two-dimensional umap-learn UMAP embedding."""
    if n_neighbors < 2:
        raise ValueError(f"UMAP n_neighbors must be at least 2, got {n_neighbors}")
    if min_dist < 0.0:
        raise ValueError(f"UMAP min_dist must be nonnegative, got {min_dist}")
    effective_n_neighbors = min(n_neighbors, matrix.shape[0] - 1)
    umap_embedding = UMAP(
        n_components=2,
        n_neighbors=effective_n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    ).fit_transform(matrix)
    return umap_embedding.astype(np.float32), effective_n_neighbors


def build_compound_color_scales(
    conditions: list[WellCondition],
) -> dict[str, CompoundColorScale]:
    """Build a distinct zero-inclusive log concentration scale per compound."""
    compound_names = sorted(
        {
            condition.compound
            for condition in conditions
            if condition.concentration_nm > 0.0
        }
    )
    base_colormap_names = ("BOP_Blue", "BOP_Orange")
    if len(compound_names) > len(base_colormap_names):
        raise ValueError(
            f"At most {len(base_colormap_names)} compounds are supported, "
            f"got {len(compound_names)}"
        )
    scales: dict[str, CompoundColorScale] = {}
    for compound, base_name in zip(compound_names, base_colormap_names, strict=False):
        unit, nanomolar_per_unit = CONCENTRATION_DISPLAY_UNITS[compound]
        concentrations = tuple(
            sorted(
                {
                    condition.concentration_nm / nanomolar_per_unit
                    for condition in conditions
                    if condition.compound == compound
                    and condition.concentration_nm > 0.0
                }
            )
        )
        scales[compound] = CompoundColorScale(
            colormap=CmapColormap(base_name).to_mpl(),
            norm=SymLogNorm(
                linthresh=concentrations[0] / 10.0,
                vmin=0.0,
                vmax=concentrations[-1],
                base=10.0,
            ),
            concentrations=concentrations,
            unit=unit,
            nanomolar_per_unit=nanomolar_per_unit,
        )
    return scales


def scatter_compound_concentrations(
    ax: Axes,
    *,
    embedding: NDArray[np.float32],
    conditions: list[WellCondition],
    scales: dict[str, CompoundColorScale],
    point_size: float = 16.0,
    alpha: float = 1.0,
    rasterized: bool = False,
) -> None:
    """Scatter an embedding by compound-specific concentration color."""
    frame = pd.DataFrame(
        {
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "compound": [condition.compound for condition in conditions],
            "concentration_nm": [
                condition.concentration_nm for condition in conditions
            ],
        }
    )
    zero_concentration = frame["concentration_nm"] == 0.0
    sns.scatterplot(
        data=frame[zero_concentration],
        x="x",
        y="y",
        ax=ax,
        color=NEUTRAL_COLORMAP(0.0),
        s=point_size,
        alpha=alpha,
        linewidth=0,
        rasterized=rasterized,
        legend=False,
    )
    for compound, scale in scales.items():
        selected = (frame["compound"] == compound) & ~zero_concentration
        compound_frame = frame[selected].copy()
        compound_frame["concentration"] = (
            compound_frame["concentration_nm"] / scale.nanomolar_per_unit
        )
        sns.scatterplot(
            data=compound_frame,
            x="x",
            y="y",
            hue="concentration",
            hue_norm=scale.norm,
            palette=scale.colormap,
            ax=ax,
            s=point_size,
            alpha=alpha,
            linewidth=0,
            rasterized=rasterized,
            legend=False,
        )
    assigned = zero_concentration | frame["compound"].isin(scales)
    if not bool(assigned.all()):
        unassigned = sorted(frame.loc[~assigned, "compound"].unique().tolist())
        raise ValueError(f"Positive concentrations have no color scale: {unassigned}")


def plot_embedding_diagnostics(
    *,
    train_umap_embedding: NDArray[np.float32],
    validation_pca_embedding: NDArray[np.float32],
    validation_pca: PCA,
    train_conditions: list[WellCondition],
    validation_conditions: list[WellCondition],
) -> plt.Figure:
    """Plot the split-specific embeddings used by the publication figure."""
    compound_scales = build_compound_color_scales(
        [*train_conditions, *validation_conditions]
    )
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.8), layout="constrained")
    scatter_compound_concentrations(
        axes[0],
        embedding=train_umap_embedding,
        conditions=train_conditions,
        scales=compound_scales,
    )
    axes[0].set(xlabel="UMAP 1", ylabel="UMAP 2", title="Training")

    scatter_compound_concentrations(
        axes[1],
        embedding=validation_pca_embedding,
        conditions=validation_conditions,
        scales=compound_scales,
    )
    explained = 100.0 * validation_pca.explained_variance_ratio_
    axes[1].set(
        xlabel=f"PC1 ({explained[0]:.1f}%)",
        ylabel=f"PC2 ({explained[1]:.1f}%)",
        title="Validation",
    )

    for ax in axes:
        ax.grid(alpha=0.2)

    for ax, (compound, scale) in zip(
        axes,
        compound_scales.items(),
        strict=True,
    ):
        colorbar = fig.colorbar(
            ScalarMappable(norm=scale.norm, cmap=scale.colormap),
            ax=ax,
            orientation="horizontal",
            fraction=0.06,
            pad=0.14,
            label=f"{compound} concentration ({scale.unit})",
        )
        ticks = (0.0, *scale.concentrations)
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"{concentration:g}" for concentration in ticks])
        plt.setp(
            colorbar.ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=8,
        )
    return fig


def fit_and_sample_gmm(
    matrix: NDArray[np.float32],
    *,
    num_components: int,
    samples_per_component: int,
    reg_covar: float,
    random_state: int,
) -> tuple[GaussianMixture, NDArray[np.float32]]:
    """Fit a full-covariance GMM and draw equally many samples per component."""
    if num_components != 2:
        raise ValueError(f"Expected exactly two GMM components, got {num_components}")
    if samples_per_component < 1:
        raise ValueError(
            f"samples_per_component must be positive, got {samples_per_component}"
        )
    gmm = GaussianMixture(
        n_components=num_components,
        covariance_type="full",
        reg_covar=reg_covar,
        random_state=random_state,
    ).fit(matrix)
    generator = np.random.default_rng(random_state)
    component_samples: list[NDArray[np.float32]] = []
    means = np.asarray(gmm.means_, dtype=np.float64)
    covariances = np.asarray(gmm.covariances_, dtype=np.float64)
    for mean, covariance in zip(means, covariances, strict=True):
        covariance_cholesky = np.linalg.cholesky(covariance)
        noise = generator.standard_normal((samples_per_component, matrix.shape[1]))
        samples = mean + noise @ covariance_cholesky.T
        component_samples.append(samples.astype(np.float32))
    return gmm, np.stack(component_samples)


def classify_control_wells(
    *,
    objects: list[ObjectLatent],
    well_conditions: list[WellCondition],
    matrix: NDArray[np.float32],
    gmm: GaussianMixture,
    component_roles: tuple[str, str],
) -> tuple[list[str], list[str]]:
    """Classify control wells from their mean object-level GMM responsibilities."""
    if not (len(objects) == len(well_conditions) == matrix.shape[0]):
        raise ValueError(
            "objects, well_conditions, and matrix rows must have equal lengths"
        )
    responsibilities = gmm.predict_proba(matrix)
    if responsibilities.shape[1] != len(CONTROL_ROLES):
        raise ValueError(
            f"Expected {len(CONTROL_ROLES)} GMM components for binary control "
            f"classification, got {responsibilities.shape[1]}"
        )

    responsibilities_by_well: dict[str, list[NDArray[np.float64]]] = {}
    role_by_well: dict[str, str] = {}
    for obj, condition, object_responsibilities in zip(
        objects, well_conditions, responsibilities, strict=True
    ):
        if condition.role not in CONTROL_ROLES:
            continue
        previous_role = role_by_well.setdefault(obj.condition, condition.role)
        if previous_role != condition.role:
            raise ValueError(
                f"Control well {obj.condition} has conflicting roles: "
                f"{previous_role!r} and {condition.role!r}"
            )
        responsibilities_by_well.setdefault(obj.condition, []).append(
            np.asarray(object_responsibilities, dtype=np.float64)
        )

    observed_roles = set(role_by_well.values())
    if observed_roles != set(CONTROL_ROLES):
        raise ValueError(
            f"Expected control roles {CONTROL_ROLES}, got {sorted(observed_roles)}"
        )
    well_names = sorted(responsibilities_by_well)
    mean_responsibilities = np.stack(
        [np.mean(responsibilities_by_well[name], axis=0) for name in well_names]
    )
    true_roles = [role_by_well[name] for name in well_names]

    predicted_components = mean_responsibilities.argmax(axis=1)
    predicted_roles = [component_roles[index] for index in predicted_components]
    return true_roles, predicted_roles


def infer_gmm_component_roles_from_controls(
    *,
    conditions: list[WellCondition],
    responsibilities: NDArray[np.float32],
) -> tuple[tuple[str, str], NDArray[np.float64]]:
    """Map validation-fitted GMM components to known control phenotypes."""
    if responsibilities.shape != (len(conditions), len(CONTROL_ROLES)):
        raise ValueError(
            "responsibilities must have one row per condition and one column per "
            f"control role, got {responsibilities.shape}"
        )
    mean_responsibilities = np.stack(
        [
            responsibilities[
                np.asarray(
                    [condition.role == role for condition in conditions], dtype=bool
                )
            ].mean(axis=0, dtype=np.float64)
            for role in CONTROL_ROLES
        ]
    )
    preferred_components = mean_responsibilities.argmax(axis=1)
    if len(set(preferred_components.tolist())) != len(CONTROL_ROLES):
        raise RuntimeError(
            "Both control phenotypes prefer the same GMM component: "
            f"{mean_responsibilities}"
        )
    positive_component = int(
        preferred_components[CONTROL_ROLES.index("positive_control")]
    )
    component_roles = (
        "positive_control" if positive_component == 0 else "negative_control",
        "positive_control" if positive_component == 1 else "negative_control",
    )
    return component_roles, mean_responsibilities


def decode_generated_object_patches(
    system: PyramidFlowSystem,
    *,
    component_samples: NDArray[np.float32],
    canvas_grid_size: int,
    patch_size: int,
    device: torch.device,
) -> NDArray[np.float32]:
    """Decode centered single-object latent samples on a zero background canvas."""
    if component_samples.ndim != 3:
        raise ValueError(
            "component_samples must have shape (components, samples, latent_dim), "
            f"got {component_samples.shape}"
        )
    num_components, samples_per_component, latent_dim = component_samples.shape
    if latent_dim != system.decoder.in_channels:
        raise ValueError(
            f"Expected latent dimension {system.decoder.in_channels}, got {latent_dim}"
        )
    flat_samples = torch.as_tensor(
        component_samples.reshape(-1, latent_dim), device=device
    )
    batch_size = int(flat_samples.shape[0])
    foreground = torch.zeros(
        (batch_size, latent_dim, canvas_grid_size, canvas_grid_size),
        device=device,
        dtype=flat_samples.dtype,
    )
    background = torch.zeros_like(foreground)
    presence = torch.zeros(
        (batch_size, 1, canvas_grid_size, canvas_grid_size),
        device=device,
        dtype=flat_samples.dtype,
    )
    grid_row = canvas_grid_size // 2
    grid_col = canvas_grid_size // 2
    foreground[:, :, grid_row, grid_col] = flat_samples
    presence[:, :, grid_row, grid_col] = 1.0
    with torch.inference_mode():
        reconstruction, _layer_transports = system.decoder(
            foreground, background, presence
        )
    center_row = grid_row * system.patch_size + system.patch_size // 2
    center_col = grid_col * system.patch_size + system.patch_size // 2
    row_start = center_row - patch_size // 2
    col_start = center_col - patch_size // 2
    patches = reconstruction[
        :,
        :,
        row_start : row_start + patch_size,
        col_start : col_start + patch_size,
    ]
    if tuple(patches.shape[-2:]) != (patch_size, patch_size):
        raise RuntimeError(
            f"Generated patch is clipped to {tuple(patches.shape[-2:])}; "
            "increase canvas_grid_size"
        )
    return (
        patches.reshape(
            num_components,
            samples_per_component,
            int(patches.shape[1]),
            patch_size,
            patch_size,
        )
        .detach()
        .cpu()
        .float()
        .numpy()
    )


def normalize_patch_channels(
    patches: NDArray[np.float32], *, quantiles: tuple[float, float]
) -> NDArray[np.float32]:
    """Quantile-normalize each channel jointly across a patch collection."""
    if patches.ndim != 5 or patches.shape[2] != 2:
        raise ValueError(
            "patches must have shape (components, samples, 2, height, width), "
            f"got {patches.shape}"
        )
    normalized = np.empty_like(patches)
    for channel in range(2):
        values = patches[:, :, channel]
        low, high = np.quantile(values, quantiles)
        if high <= low:
            raise RuntimeError(
                f"Channel {channel} has a degenerate display range [{low}, {high}]"
            )
        normalized[:, :, channel] = np.clip((values - low) / (high - low), 0.0, 1.0)
    return normalized


def two_channel_patch_composites(
    patches: NDArray[np.float32], *, quantiles: tuple[float, float]
) -> NDArray[np.float32]:
    """Map channel 0 to green and channel 1 to magenta with shared scales."""
    normalized = normalize_patch_channels(patches, quantiles=quantiles)
    channel_rgb = [
        CmapColormap(colormap_name).to_mpl()(normalized[:, :, channel])[..., :3]
        for channel, colormap_name in enumerate(CHANNEL_DISPLAY_COLORMAPS)
    ]
    return np.clip(channel_rgb[0] + channel_rgb[1], 0.0, 1.0).astype(np.float32)


def retrieve_nearest_object_patches(
    *,
    objects: list[ObjectLatent],
    matrix: NDArray[np.float32],
    component_means: NDArray[np.float64],
    source_images: dict[str, NDArray[np.float32]],
    samples_per_component: int,
    grid_patch_size: int,
    output_patch_size: int,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Crop observed objects nearest to each GMM mean in latent space."""
    if matrix.shape[0] != len(objects):
        raise ValueError("matrix must contain one row per object")
    half_size = output_patch_size // 2
    component_patches: list[NDArray[np.float32]] = []
    component_indices: list[NDArray[np.int64]] = []
    squared_distances = np.sum(
        (matrix[None].astype(np.float64) - component_means[:, None]) ** 2,
        axis=2,
    )
    for distances in squared_distances:
        patches: list[NDArray[np.float32]] = []
        indices: list[int] = []
        for object_index in np.argsort(distances):
            obj = objects[int(object_index)]
            image = source_images[obj.condition]
            center_row = round(
                obj.center_grid_row * grid_patch_size + grid_patch_size / 2
            )
            center_col = round(
                obj.center_grid_col * grid_patch_size + grid_patch_size / 2
            )
            row_start = center_row - half_size
            col_start = center_col - half_size
            patch = image[
                :,
                row_start : row_start + output_patch_size,
                col_start : col_start + output_patch_size,
            ]
            if patch.shape[1:] != (output_patch_size, output_patch_size):
                continue
            patches.append(patch)
            indices.append(int(object_index))
            if len(patches) == samples_per_component:
                break
        if len(patches) != samples_per_component:
            raise RuntimeError(
                f"Found only {len(patches)} complete observed patches, "
                f"expected {samples_per_component}"
            )
        component_patches.append(np.stack(patches))
        component_indices.append(np.asarray(indices, dtype=np.int64))
    return np.stack(component_patches), np.stack(component_indices)


def gaussian_density_grid(
    embedding: NDArray[np.float32],
    *,
    mean: NDArray[np.float64],
    covariance: NDArray[np.float64],
    grid_size: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    """Evaluate an analytical bivariate Gaussian over a PCA-space grid."""
    coordinate_min = embedding.min(axis=0)
    coordinate_max = embedding.max(axis=0)
    margin = 0.05 * (coordinate_max - coordinate_min)
    x = np.linspace(
        coordinate_min[0] - margin[0], coordinate_max[0] + margin[0], grid_size
    )
    y = np.linspace(
        coordinate_min[1] - margin[1], coordinate_max[1] + margin[1], grid_size
    )
    grid_x, grid_y = np.meshgrid(x, y)
    delta = np.stack((grid_x - mean[0], grid_y - mean[1]), axis=-1)
    covariance_precision = np.linalg.inv(covariance)
    mahalanobis_squared = np.einsum(
        "...i,ij,...j->...", delta, covariance_precision, delta
    )
    determinant = float(np.linalg.det(covariance))
    if determinant <= 0.0:
        raise RuntimeError(
            f"Projected GMM covariance must be positive definite, got det={determinant}"
        )
    peak_density = 1.0 / (2.0 * np.pi * np.sqrt(determinant))
    density = peak_density * np.exp(-0.5 * mahalanobis_squared)
    return grid_x, grid_y, density, peak_density


def square_axis_limits(
    *,
    embeddings: tuple[NDArray[np.float32], ...],
    component_means: NDArray[np.float64],
    component_covariances: NDArray[np.float64],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute equal-span limits containing the embedding and 95% GMM contours."""
    coordinates = np.concatenate(embeddings)
    contour_radius = np.sqrt(-2.0 * np.log(1.0 - GMM_PUBLICATION_CONTOUR_MASS))
    contour_extents = contour_radius * np.sqrt(
        np.diagonal(component_covariances, axis1=1, axis2=2)
    )
    coordinates = np.concatenate(
        (
            coordinates,
            component_means - contour_extents,
            component_means + contour_extents,
        )
    )
    coordinate_min = coordinates.min(axis=0)
    coordinate_max = coordinates.max(axis=0)
    center = (coordinate_min + coordinate_max) / 2.0
    half_span = 0.55 * float(np.max(coordinate_max - coordinate_min))
    return (
        (float(center[0] - half_span), float(center[0] + half_span)),
        (float(center[1] - half_span), float(center[1] + half_span)),
    )


def plot_umap_panel(
    ax: Axes,
    *,
    panel_label: str,
    embedding: NDArray[np.float32],
    conditions: list[WellCondition],
    concentration_scales: dict[str, CompoundColorScale],
) -> Axes:
    """Draw a concentration-colored UMAP panel."""
    scatter_compound_concentrations(
        ax,
        embedding=embedding,
        conditions=conditions,
        scales=concentration_scales,
        point_size=3,
        alpha=1.0,
        rasterized=True,
    )
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
    ax.set_box_aspect(1)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.set_title(panel_label, loc="left", pad=2, fontsize=9, fontweight="bold")
    return ax


def plot_pca_gmm_panel(
    ax: Axes,
    *,
    panel_label: str,
    embedding: NDArray[np.float32],
    density_grid_embedding: NDArray[np.float32],
    component_means: NDArray[np.float64],
    component_covariances: NDArray[np.float64],
    component_weights: NDArray[np.float64],
    component_colormaps: tuple[MatplotlibColormap, ...],
    explained_variance_ratio: NDArray[np.float64],
    conditions: list[WellCondition],
    concentration_scales: dict[str, CompoundColorScale],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    show_legend: bool,
) -> Axes:
    """Draw one text-light PCA panel with 95% GMM mass contours."""
    scatter_compound_concentrations(
        ax,
        embedding=embedding,
        conditions=conditions,
        scales=concentration_scales,
        point_size=7,
        alpha=1.0,
        rasterized=True,
    )
    for component_index, (colormap, color_position) in enumerate(
        zip(
            component_colormaps,
            GMM_COMPONENT_COLOR_POSITIONS,
            strict=True,
        )
    ):
        mean_xy = component_means[component_index]
        grid_x, grid_y, density, peak_density = gaussian_density_grid(
            density_grid_embedding,
            mean=mean_xy,
            covariance=component_covariances[component_index],
            grid_size=GMM_DENSITY_GRID_SIZE,
        )
        color = colormap(color_position)
        ax.contour(
            grid_x,
            grid_y,
            density,
            levels=[peak_density * (1.0 - GMM_PUBLICATION_CONTOUR_MASS)],
            colors=[color],
            linewidths=1.4,
            zorder=2,
        )
        ax.scatter(
            mean_xy[0],
            mean_xy[1],
            marker="*",
            s=75,
            color=color,
            edgecolor=NEUTRAL_COLORMAP(1.0),
            linewidth=0.6,
            label=rf"$k={component_index + 1}$ ({component_weights[component_index]:.2f})",
            zorder=3,
        )
    explained = 100.0 * explained_variance_ratio
    ax.set(
        xlabel=rf"PC1 ({explained[0]:.1f}\%)",
        ylabel=rf"PC2 ({explained[1]:.1f}\%)",
        xlim=xlim,
        ylim=ylim,
    )
    ax.set_box_aspect(1)
    ax.grid(alpha=0.15, linewidth=0.5)
    if show_legend:
        ax.legend(frameon=False, loc="upper right", handletextpad=0.2)
    ax.set_title(panel_label, loc="left", pad=2, fontsize=9, fontweight="bold")
    return ax


def add_concentration_colorbar(
    figure: Figure,
    *,
    axes: tuple[Axes, ...],
    compound: str,
    scale: CompoundColorScale,
) -> None:
    """Add one compact vertical compound concentration scale."""
    colorbar = figure.colorbar(
        ScalarMappable(norm=scale.norm, cmap=scale.colormap),
        ax=axes,
        orientation="vertical",
        shrink=0.85,
        aspect=18,
        pad=0.02,
        label=rf"{compound} ({scale.unit})",
    )
    all_ticks = np.asarray((0.0, *scale.concentrations))
    tick_indices = np.linspace(
        0,
        len(all_ticks) - 1,
        num=min(6, len(all_ticks)),
        dtype=int,
    )
    ticks = tuple(all_ticks[np.unique(tick_indices)])
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{concentration:g}" for concentration in ticks])
    colorbar.ax.tick_params(labelsize=4.5)


def experimental_compounds_by_plate_row(
    plate_map: dict[str, WellCondition],
) -> dict[str, str]:
    """Associate each plate row with its experimental compound."""
    compounds_by_row: dict[str, set[str]] = {}
    for well_name, condition in plate_map.items():
        if condition.role != "experiment":
            continue
        row = well_name.split(".", maxsplit=1)[0].rstrip("0123456789")
        compounds_by_row.setdefault(row, set()).add(condition.compound)
    ambiguous_rows = {
        row: compounds
        for row, compounds in compounds_by_row.items()
        if len(compounds) != 1
    }
    if ambiguous_rows:
        raise ValueError(f"Plate rows do not map to one compound: {ambiguous_rows}")
    return {row: next(iter(compounds)) for row, compounds in compounds_by_row.items()}


def build_validation_assignment_frame(
    *,
    objects: list[ObjectLatent],
    validation_conditions: list[WellCondition],
    validation_responsibilities: NDArray[np.float32],
    component_index: int,
    compounds_by_row: dict[str, str],
) -> pd.DataFrame:
    """Collect validation object responsibilities by phenotype and compound."""
    if not (
        len(objects)
        == len(validation_conditions)
        == validation_responsibilities.shape[0]
    ):
        raise ValueError("Validation objects, conditions, and responsibilities differ")
    plate_rows = [
        obj.condition.split(".", maxsplit=1)[0].rstrip("0123456789") for obj in objects
    ]
    return pd.DataFrame(
        {
            "phenotype": [
                "Positive" if condition.role == "positive_control" else "Negative"
                for condition in validation_conditions
            ],
            "compound": [compounds_by_row[row] for row in plate_rows],
            "responsibility": validation_responsibilities[:, component_index],
        }
    )


def build_training_dose_response_frame(
    *,
    train_conditions: list[WellCondition],
    train_responsibilities: NDArray[np.float32],
    component_index: int,
) -> pd.DataFrame:
    """Collect training object responsibilities at each experimental dose."""
    if train_responsibilities.shape != (
        len(train_conditions),
        GMM_NUM_COMPONENTS,
    ):
        raise ValueError(
            f"Training responsibilities have shape {train_responsibilities.shape}; "
            f"expected {(len(train_conditions), GMM_NUM_COMPONENTS)}"
        )
    frame = pd.DataFrame(
        {
            "compound": [condition.compound for condition in train_conditions],
            "concentration_nm": [
                condition.concentration_nm for condition in train_conditions
            ],
            "responsibility": train_responsibilities[:, component_index],
        }
    )
    frame = frame.loc[frame["concentration_nm"] > 0.0].copy()
    frame["concentration"] = [
        concentration_nm / CONCENTRATION_DISPLAY_UNITS[compound][1]
        for compound, concentration_nm in zip(
            frame["compound"], frame["concentration_nm"], strict=True
        )
    ]
    return pd.DataFrame(frame)


def plot_gmm_assignments_panel(
    ax: Axes,
    *,
    panel_label: str,
    assignment_frame: pd.DataFrame,
    component_index: int,
    compound_palette: dict[str, tuple[float, float, float, float]],
) -> None:
    """Draw validation responsibilities grouped by phenotype and compound."""
    compound_order = sorted(compound_palette)
    boxstripplot(
        data=assignment_frame,
        x="phenotype",
        y="responsibility",
        hue="compound",
        order=["Negative", "Positive"],
        hue_order=compound_order,
        palette=compound_palette,
        width=0.48,
        size=1.7,
        box_alpha=0.35,
        strip_alpha=0.55,
        linewidth=0.25,
        fliersize=0,
        ax=ax,
        strip_kwargs={"rasterized": True},
    )
    ax.set(
        xlabel="",
        ylabel=rf"$p(k={component_index + 1}\mid z)$",
        ylim=(-0.03, 1.03),
    )
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)
    ax.legend(frameon=False, title=None, ncol=2, loc="upper left")
    if panel_label:
        ax.text(
            -0.08,
            1.08,
            panel_label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )


def plot_separate_channel_patches(
    subfigure: SubFigure,
    *,
    panel_label: str,
    normalized_patches: NDArray[np.float32],
) -> None:
    """Arrange each component's channels as separate compact image axes."""
    num_components, samples_per_component, num_channels = normalized_patches.shape[:3]
    axes = subfigure.subplots(
        num_components * num_channels,
        samples_per_component,
        squeeze=False,
        gridspec_kw={"wspace": 0.02, "hspace": 0.02},
    )
    channel_colormaps = tuple(
        CmapColormap(colormap_name).to_mpl()
        for colormap_name in CHANNEL_DISPLAY_COLORMAPS
    )
    for component_index in range(num_components):
        for channel_index, colormap in enumerate(channel_colormaps):
            row = component_index * num_channels + channel_index
            for sample_index in range(samples_per_component):
                ax = axes[row, sample_index]
                ax.imshow(
                    normalized_patches[component_index, sample_index, channel_index],
                    cmap=colormap,
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="none",
                )
                ax.set_axis_off()
    axes[0, 0].set_title(
        panel_label,
        loc="left",
        pad=1,
        fontsize=9,
        fontweight="bold",
    )


def build_publication_figure(
    *,
    train_umap_embedding: NDArray[np.float32],
    validation_pca_embedding: NDArray[np.float32],
    pca: PCA,
    component_means: NDArray[np.float64],
    component_covariances: NDArray[np.float64],
    component_weights: NDArray[np.float64],
    component_colormaps: tuple[MatplotlibColormap, ...],
    train_conditions: list[WellCondition],
    validation_conditions: list[WellCondition],
    training_dose_response_frame: pd.DataFrame,
    assignment_component_index: int,
) -> Figure:
    """Build the compact three-panel latent-analysis figure."""
    xlim, ylim = square_axis_limits(
        embeddings=(validation_pca_embedding,),
        component_means=component_means,
        component_covariances=component_covariances,
    )
    density_grid_embedding = np.concatenate(
        (
            validation_pca_embedding,
            np.asarray(((xlim[0], ylim[0]), (xlim[1], ylim[1])), dtype=np.float32),
        )
    )
    concentration_scales = build_compound_color_scales(
        [*train_conditions, *validation_conditions]
    )
    compound_order = list(concentration_scales)
    compound_palette = {
        compound: scale.colormap(0.75)
        for compound, scale in concentration_scales.items()
    }
    dose_grid = sns.relplot(
        data=training_dose_response_frame,
        x="concentration",
        y="responsibility",
        hue="compound",
        col="compound",
        hue_order=compound_order,
        col_order=compound_order,
        palette=compound_palette,
        kind="line",
        estimator="mean",
        errorbar=("ci", 95),
        n_boot=1000,
        seed=RANDOM_STATE,
        marker="o",
        markersize=3.5,
        linewidth=1.2,
        legend=False,
        facet_kws={"sharex": False, "sharey": True},
        height=2.2,
        aspect=1.2,
    )
    figure = dose_grid.figure
    figure.set_layout_engine(None)
    figure.set_size_inches(FIGURE_WIDTH, 4.0)
    outer_grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.0, 1.0),
    )
    upper_grid = outer_grid[0].subgridspec(1, 2)
    dose_grid_spec = outer_grid[1].subgridspec(1, len(compound_order))
    dose_axes = np.asarray(dose_grid.axes).ravel()
    for ax, subplot_spec in zip(
        dose_axes,
        dose_grid_spec,  # type: ignore[bad-argument-type]
        strict=True,
    ):
        ax.set_subplotspec(subplot_spec)
    train_ax = figure.add_subplot(upper_grid[0])
    validation_ax = figure.add_subplot(upper_grid[1])
    plot_umap_panel(
        train_ax,
        panel_label="A",
        embedding=train_umap_embedding,
        conditions=train_conditions,
        concentration_scales=concentration_scales,
    )
    plot_pca_gmm_panel(
        validation_ax,
        panel_label="B",
        embedding=validation_pca_embedding,
        density_grid_embedding=density_grid_embedding,
        component_means=component_means,
        component_covariances=component_covariances,
        component_weights=component_weights,
        component_colormaps=component_colormaps,
        explained_variance_ratio=pca.explained_variance_ratio_,
        conditions=validation_conditions,
        concentration_scales=concentration_scales,
        xlim=xlim,
        ylim=ylim,
        show_legend=True,
    )
    for compound, scale in concentration_scales.items():
        add_concentration_colorbar(
            figure,
            axes=(train_ax, validation_ax),
            compound=compound,
            scale=scale,
        )
    for facet_index, (ax, compound) in enumerate(
        zip(dose_axes, compound_order, strict=True)
    ):
        unit = CONCENTRATION_DISPLAY_UNITS[compound][0]
        ax.set(
            xlabel=rf"Concentration ({unit})",
            ylabel=(
                rf"$p(k={assignment_component_index + 1}\mid z)$"
                if ax is dose_axes[0]
                else ""
            ),
            ylim=(-0.03, 1.03),
            xscale="log",
        )
        ax.set_title("")
        title = rf"$\mathbf{{C}}\quad$ {compound}" if facet_index == 0 else compound
        ax.set_title(title, loc="left" if facet_index == 0 else "center")
        ax.grid(alpha=0.15, linewidth=0.5)
    figure.set_layout_engine("constrained")
    return figure


def build_publication_image_figure(
    *,
    generated_patches: NDArray[np.float32],
    retrieved_patches: NDArray[np.float32],
) -> Figure:
    """Build the generated-versus-retrieved image figure."""
    figure = plt.figure(figsize=(FIGURE_WIDTH, 2.45), layout="constrained")
    image_subfigures = figure.subfigures(1, 2, squeeze=False, wspace=0.02)[0]
    plot_separate_channel_patches(
        image_subfigures[0],
        panel_label="A",
        normalized_patches=normalize_patch_channels(
            generated_patches,
            quantiles=GENERATED_DISPLAY_QUANTILES,
        ),
    )
    plot_separate_channel_patches(
        image_subfigures[1],
        panel_label="B",
        normalized_patches=normalize_patch_channels(
            retrieved_patches,
            quantiles=GENERATED_DISPLAY_QUANTILES,
        ),
    )
    return figure


# %% Extract object latents and registered source images.
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.set_float32_matmul_precision("high")
system = build_system(checkpoint=CHECKPOINT, device=device)
train_data = build_evaluation_data(columns=TRAIN_COLUMNS, device=device)
validation_data = build_evaluation_data(columns=VALIDATION_COLUMNS, device=device)
train_objects, _train_source_images = extract_object_latents(
    system=system,
    loader=train_data.test_dataloader(),
    split="training",
    device=device,
    center_threshold=CENTER_THRESHOLD,
    pixel_mass_threshold=PIXEL_MASS_THRESHOLD,
    component_chunk_size=COMPONENT_CHUNK_SIZE,
    min_object_area=MIN_OBJECT_AREA_MODEL_PIXELS,
    max_object_area=MAX_OBJECT_AREA_MODEL_PIXELS,
)
validation_objects, validation_source_images = extract_object_latents(
    system=system,
    loader=validation_data.test_dataloader(),
    split="validation",
    device=device,
    center_threshold=CENTER_THRESHOLD,
    pixel_mass_threshold=PIXEL_MASS_THRESHOLD,
    component_chunk_size=COMPONENT_CHUNK_SIZE,
    min_object_area=MIN_OBJECT_AREA_MODEL_PIXELS,
    max_object_area=MAX_OBJECT_AREA_MODEL_PIXELS,
)

print(f"device={device}")
print(f"train_columns={TRAIN_COLUMNS} validation_columns={VALIDATION_COLUMNS}")
print(
    f"center_threshold={CENTER_THRESHOLD:.3f} "
    f"pixel_mass_threshold={PIXEL_MASS_THRESHOLD:.3f}"
)
print(
    "object_area_filter_model_pixels "
    f"min={MIN_OBJECT_AREA_MODEL_PIXELS} max={MAX_OBJECT_AREA_MODEL_PIXELS}"
)
print(
    f"train_objects={len(train_objects)} "
    f"train_wells={len({obj.condition for obj in train_objects})}"
)
print(
    f"validation_objects={len(validation_objects)} "
    f"validation_wells={len({obj.condition for obj in validation_objects})}"
)


# %% Build metadata and latent matrices.
plate_map = load_plate_map(PLATE_MAP)
train_matrix = object_latent_matrix(train_objects)
validation_matrix = object_latent_matrix(validation_objects)
train_conditions = object_conditions(train_objects, plate_map)
validation_conditions = object_conditions(validation_objects, plate_map)
print(f"train_object_latent_matrix_shape={train_matrix.shape}")
print(f"validation_object_latent_matrix_shape={validation_matrix.shape}")


# %% Fit UMAP on training objects and PCA on validation objects.
train_umap_embedding, train_n_neighbors = fit_umap_embedding(
    train_matrix,
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    random_state=RANDOM_STATE,
)
validation_pca_embedding, validation_pca = fit_pca_embedding(validation_matrix)
print(
    "validation_pca_explained_variance_ratio="
    f"{validation_pca.explained_variance_ratio_}"
)
print(f"train_umap_n_neighbors={train_n_neighbors} umap_min_dist={UMAP_MIN_DIST:g}")


# %% Inline diagnostic: split-specific UMAP and PCA embeddings.
embedding_diagnostic_figure = plot_embedding_diagnostics(
    train_umap_embedding=train_umap_embedding,
    validation_pca_embedding=validation_pca_embedding,
    validation_pca=validation_pca,
    train_conditions=train_conditions,
    validation_conditions=validation_conditions,
)
plt.show()


# %% Fit the GMM on validation objects and assign objects from both splits.
gmm, gmm_component_samples = fit_and_sample_gmm(
    validation_matrix,
    num_components=GMM_NUM_COMPONENTS,
    samples_per_component=GMM_SAMPLES_PER_COMPONENT,
    reg_covar=GMM_REG_COVAR,
    random_state=RANDOM_STATE,
)
gmm_weights = np.asarray(gmm.weights_, dtype=np.float64)
gmm_means = np.asarray(gmm.means_, dtype=np.float64)
gmm_covariances = np.asarray(gmm.covariances_, dtype=np.float64)
validation_gmm_responsibilities = gmm.predict_proba(validation_matrix).astype(
    np.float32
)
train_gmm_responsibilities = gmm.predict_proba(train_matrix).astype(np.float32)
print(f"gmm_converged={gmm.converged_} iterations={gmm.n_iter_} weights={gmm_weights}")
gmm_component_roles, control_mean_responsibilities = (
    infer_gmm_component_roles_from_controls(
        conditions=validation_conditions,
        responsibilities=validation_gmm_responsibilities,
    )
)
control_true_roles, control_predicted_roles = classify_control_wells(
    objects=validation_objects,
    well_conditions=validation_conditions,
    matrix=validation_matrix,
    gmm=gmm,
    component_roles=gmm_component_roles,
)
print(
    f"gmm_component_roles={gmm_component_roles} "
    f"control_mean_responsibilities={control_mean_responsibilities}"
)
print(
    classification_report(
        control_true_roles,
        control_predicted_roles,
        labels=list(CONTROL_ROLES),
        zero_division=0,
    )
)
assignment_component_index = gmm_component_roles.index("positive_control")
compounds_by_plate_row = experimental_compounds_by_plate_row(plate_map)
validation_assignment_frame = build_validation_assignment_frame(
    objects=validation_objects,
    validation_conditions=validation_conditions,
    validation_responsibilities=validation_gmm_responsibilities,
    component_index=assignment_component_index,
    compounds_by_row=compounds_by_plate_row,
)
training_dose_response_frame = build_training_dose_response_frame(
    train_conditions=train_conditions,
    train_responsibilities=train_gmm_responsibilities,
    component_index=assignment_component_index,
)


# %% Inline diagnostic: validation GMM assignments by control phenotype.
assignment_diagnostic_scales = build_compound_color_scales(
    [*train_conditions, *validation_conditions]
)
assignment_diagnostic_palette = {
    compound: scale.colormap(0.75)
    for compound, scale in assignment_diagnostic_scales.items()
}
fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.0), layout="constrained")
plot_gmm_assignments_panel(
    ax,
    panel_label="",
    assignment_frame=validation_assignment_frame,
    component_index=assignment_component_index,
    compound_palette=assignment_diagnostic_palette,
)
plt.show()


# %% Inline diagnostic: validation PCA with GMM density contours.
gmm_density_colormaps = tuple(
    CmapColormap(colormap_name).to_mpl() for colormap_name in GMM_DENSITY_COLORMAPS
)
gmm_means_pca = validation_pca.transform(gmm_means)
gmm_pca_projection = np.asarray(validation_pca.components_, dtype=np.float64)
gmm_covariances_pca = np.stack(
    [
        gmm_pca_projection @ covariance @ gmm_pca_projection.T
        for covariance in gmm_covariances
    ]
)
fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4.2), layout="constrained")
ax.scatter(
    validation_pca_embedding[:, 0],
    validation_pca_embedding[:, 1],
    s=14,
    alpha=0.45,
    color=NEUTRAL_COLORMAP(0.0),
    zorder=1,
)
for component_index, (colormap, color_position) in enumerate(
    zip(gmm_density_colormaps, GMM_COMPONENT_COLOR_POSITIONS, strict=True)
):
    mean_xy = gmm_means_pca[component_index]
    component_color = colormap(color_position)
    grid_x, grid_y, density, peak_density = gaussian_density_grid(
        validation_pca_embedding,
        mean=mean_xy,
        covariance=gmm_covariances_pca[component_index],
        grid_size=GMM_DENSITY_GRID_SIZE,
    )
    density_levels = sorted(peak_density * (1.0 - mass) for mass in GMM_DENSITY_MASSES)
    ax.contour(
        grid_x,
        grid_y,
        density,
        levels=density_levels,
        colors=[component_color],
        linewidths=2.5,
        zorder=2,
    )
    ax.scatter(
        mean_xy[0],
        mean_xy[1],
        marker="*",
        s=350,
        color=component_color,
        edgecolor=NEUTRAL_COLORMAP(1.0),
        linewidth=1.5,
        label=(
            f"component {component_index + 1} "
            f"(weight={gmm_weights[component_index]:.3f})"
        ),
        zorder=3,
    )
validation_explained = 100.0 * validation_pca.explained_variance_ratio_
ax.set_xlabel(f"PC1 ({validation_explained[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({validation_explained[1]:.1f}%)")
ax.set_title("Validation")
ax.grid(alpha=0.2)
ax.legend()
plt.show()


# %% Decode GMM samples and retrieve their nearest observed neighbors.
generated_object_patches = decode_generated_object_patches(
    system,
    component_samples=gmm_component_samples,
    canvas_grid_size=GENERATED_OBJECT_CANVAS_GRID_SIZE,
    patch_size=GENERATED_OBJECT_PATCH_SIZE,
    device=device,
)
generated_object_composites = two_channel_patch_composites(
    generated_object_patches,
    quantiles=GENERATED_DISPLAY_QUANTILES,
)
retrieved_object_patches, retrieved_object_indices = retrieve_nearest_object_patches(
    objects=validation_objects,
    matrix=validation_matrix,
    component_means=gmm_means,
    source_images=validation_source_images,
    samples_per_component=GMM_SAMPLES_PER_COMPONENT,
    grid_patch_size=system.patch_size,
    output_patch_size=GENERATED_OBJECT_PATCH_SIZE,
)
print(f"generated_object_patches_shape={generated_object_patches.shape}")
print(f"retrieved_object_indices={retrieved_object_indices.tolist()}")


# %% Inline diagnostic: generated two-channel composites.
fig, axes = plt.subplots(
    GMM_NUM_COMPONENTS,
    GMM_SAMPLES_PER_COMPONENT,
    figsize=(FIGURE_WIDTH, 1.6),
    layout="constrained",
    squeeze=False,
)
for component_index in range(GMM_NUM_COMPONENTS):
    for sample_index in range(GMM_SAMPLES_PER_COMPONENT):
        ax = axes[component_index, sample_index]
        ax.imshow(generated_object_composites[component_index, sample_index])
        ax.axis("off")
        if component_index == 0:
            ax.set_title(f"sample {sample_index + 1}")
        if sample_index == 0:
            ax.text(
                -0.08,
                0.5,
                f"component {component_index + 1}\nweight={gmm_weights[component_index]:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="center",
            )
plt.show()


# %% Publication figure: saved as one PDF and shown inline in full.
publication_figure = build_publication_figure(
    train_umap_embedding=train_umap_embedding,
    validation_pca_embedding=validation_pca_embedding,
    pca=validation_pca,
    component_means=gmm_means_pca,
    component_covariances=gmm_covariances_pca,
    component_weights=gmm_weights,
    component_colormaps=gmm_density_colormaps,
    train_conditions=train_conditions,
    validation_conditions=validation_conditions,
    training_dose_response_frame=training_dose_response_frame,
    assignment_component_index=assignment_component_index,
)
publication_image_figure = build_publication_image_figure(
    generated_patches=generated_object_patches,
    retrieved_patches=retrieved_object_patches,
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
publication_figure.savefig(PUBLICATION_FIGURE_PATH)
print(f"saved_publication_figure={PUBLICATION_FIGURE_PATH.resolve()}")
plt.show()
publication_image_figure.savefig(PUBLICATION_IMAGE_FIGURE_PATH)
print(f"saved_publication_image_figure={PUBLICATION_IMAGE_FIGURE_PATH.resolve()}")
plt.show()

# %%
