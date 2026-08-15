"""Discovery of frame sequences for supported microscopy dataset layouts."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameSequence:
    sequence_id: str
    frame_files: tuple[Path, ...]


_BBBC013_IMAGE_RE = re.compile(r"^(?P<first_character>.)(?P<index>\d+)\.tif$")


def discover_bbbc013_images(
    root_dir: Path,
    first_characters: Sequence[str],
    columns: Sequence[int] | None = None,
) -> tuple[FrameSequence, ...]:
    """Discover selected BBBC013 wells from ``<first character><index>.tif``."""
    characters = tuple(first_characters)
    if not characters:
        raise ValueError("first_characters must contain at least one character")
    if any(
        not isinstance(character, str) or len(character) != 1
        for character in characters
    ):
        raise ValueError(
            "first_characters entries must each be exactly one character, "
            f"got {characters!r}"
        )
    if len(set(characters)) != len(characters):
        raise ValueError("first_characters must contain unique characters")
    selected_columns = None if columns is None else tuple(columns)
    if selected_columns is not None:
        if not selected_columns:
            raise ValueError("columns must contain at least one column")
        if any(
            not isinstance(column, int) or isinstance(column, bool) or column < 1
            for column in selected_columns
        ):
            raise ValueError(
                f"columns entries must be positive integers, got {selected_columns!r}"
            )
        if len(set(selected_columns)) != len(selected_columns):
            raise ValueError("columns must contain unique columns")
    if not root_dir.is_dir():
        raise FileNotFoundError(f"BBBC013 image directory {root_dir} does not exist")

    order = {character: index for index, character in enumerate(characters)}
    column_set = None if selected_columns is None else set(selected_columns)
    selected_files: list[tuple[int, int, Path]] = []
    discovered_characters: set[str] = set()
    discovered_columns: set[int] = set()
    for image_file in root_dir.glob("*.tif"):
        match = _BBBC013_IMAGE_RE.match(image_file.name)
        if match is None:
            raise ValueError(
                f"Unexpected BBBC013 filename {image_file.name!r}. "
                "Expected '<first character><integer>.tif'."
            )
        first_character = match.group("first_character")
        if first_character not in order:
            continue
        column = int(match.group("index"))
        if column_set is not None and column not in column_set:
            continue
        discovered_characters.add(first_character)
        discovered_columns.add(column)
        selected_files.append((order[first_character], column, image_file))

    missing_characters = sorted(set(characters).difference(discovered_characters))
    if missing_characters:
        raise RuntimeError(
            "No BBBC013 images found for first characters "
            f"{missing_characters} in {root_dir}"
        )
    if selected_columns is not None:
        missing_columns = sorted(set(selected_columns).difference(discovered_columns))
        if missing_columns:
            raise RuntimeError(
                f"No BBBC013 images found for columns {missing_columns} in {root_dir}"
            )

    selected_files.sort(key=lambda item: (item[0], item[1]))
    return tuple(
        FrameSequence(sequence_id=image_file.stem, frame_files=(image_file,))
        for _, _, image_file in selected_files
    )


def discover_ctc_split(root_dir: Path, split: str) -> tuple[FrameSequence, ...]:
    """Discover CTC videos from ``<root>/<split>/0[0-9]/t*.tif``."""
    split_dir = root_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory {split_dir} does not exist")

    sequences: list[FrameSequence] = []
    for video_dir in sorted(split_dir.glob("0[0-9]")):
        if not video_dir.is_dir():
            continue
        frame_files = tuple(sorted(video_dir.glob("t*.tif")))
        if frame_files:
            sequences.append(
                FrameSequence(sequence_id=video_dir.name, frame_files=frame_files)
            )

    return tuple(sequences)


_EPITHELIA_SPLITS: dict[str, tuple[str, ...]] = {
    "train": ("per02", "per03", "pro01"),
    "val": ("ds2/per01",),
    "test": ("ds2/per01",),
    "all": ("ds2/per01", "per02", "per03", "pro01"),
}


_HELA_KYOTO_SPLITS: dict[str, tuple[str, ...]] = {
    "train": ("20210904_TL2 - R05-C03", "20210904_TL2 - R05-C05"),
    "val": ("20210904_TL2 - R05-C07",),
    "test": ("20210904_TL2 - R05-C07",),
    "all": (
        "20210904_TL2 - R05-C03",
        "20210904_TL2 - R05-C05",
        "20210904_TL2 - R05-C07",
    ),
}
_HELA_KYOTO_ROI_RE = re.compile(r"^(?P<acquisition>.+)-F(?P<roi_index>\d+)\.tif$")


def discover_ome_zarr_stores(
    root_dir: Path,
    store_names: Sequence[str],
) -> tuple[FrameSequence, ...]:
    """Resolve explicitly named OME-Zarr stores under a dataset root."""
    names = tuple(store_names)
    if not names:
        raise ValueError("store_names must contain at least one OME-Zarr store")
    if len(set(names)) != len(names):
        raise ValueError(f"store_names must be unique, got {names!r}")
    if not root_dir.is_dir():
        raise FileNotFoundError(f"OME-Zarr root directory {root_dir} does not exist")

    stores: list[FrameSequence] = []
    for store_name in names:
        if Path(store_name).name != store_name or not store_name.endswith(".ome.zarr"):
            raise ValueError(
                "OME-Zarr store names must be directory names ending in "
                f"'.ome.zarr', got {store_name!r}"
            )
        store_dir = root_dir / store_name
        if not store_dir.is_dir():
            raise FileNotFoundError(f"OME-Zarr store {store_dir} does not exist")
        stores.append(FrameSequence(sequence_id=store_name, frame_files=(store_dir,)))
    return tuple(stores)
