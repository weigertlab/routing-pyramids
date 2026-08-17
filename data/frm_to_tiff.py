# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "tifffile",
# ]
# ///

import argparse
import struct
from pathlib import Path

import numpy as np
import tifffile

INPUT_DIR = Path("BBBC013_v1_images_frm")
OUTPUT_DIR = Path("BBBC013_v1_images_converted")
OVERWRITE = False


def read_channels(path: Path) -> np.ndarray:
    """Decode both compressed InCell 3000 planes as ``(channel, y, x)``."""
    data = path.read_bytes()
    pixels_offset, width, encoded_lines = struct.unpack_from("<HHH", data)
    channel_count = encoded_lines % 32
    if channel_count != 2:
        raise ValueError(
            f"{path}: expected two InCell 3000 planes, got {channel_count}"
        )
    height = (encoded_lines - channel_count) // channel_count
    pixel_count = channel_count * height * width
    pixels = np.empty(pixel_count, dtype=np.uint16)

    input_offset = pixels_offset
    output_offset = 0
    while output_offset < pixel_count:
        token = struct.unpack_from("<H", data, input_offset)[0]
        input_offset += 2
        if token <= 32768:
            pixels[output_offset] = token
            output_offset += 1
            continue

        run_length = token - 32768
        start_value = struct.unpack_from("<H", data, input_offset)[0]
        input_offset += 2
        packed_word_count = (run_length + 2) // 3
        packed_values = struct.unpack_from(
            f"<{packed_word_count}H",
            data,
            input_offset,
        )
        input_offset += 2 * packed_word_count
        if output_offset + run_length > pixel_count:
            raise ValueError(f"{path}: compressed pixel run exceeds image dimensions")
        for run_index in range(run_length):
            packed_value = packed_values[run_index // 3]
            if run_index % 3:
                packed_value >>= 5
            pixels[output_offset] = start_value + (packed_value & 31)
            output_offset += 1

    raw_channels = pixels.reshape(channel_count, height, width)
    return raw_channels[[1, 0]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frm_files = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".frm"
    )

    if not frm_files:
        raise RuntimeError(f"No FRM files found in {INPUT_DIR.resolve()}")

    for source in frm_files:
        well_name = source.stem.split("_")[-2]
        destination = OUTPUT_DIR / f"{well_name}.tif"

        if destination.exists() and not args.overwrite:
            print(f"Skipping existing file: {destination}")
            continue

        image = read_channels(source)

        tifffile.imwrite(
            destination,
            image,
            imagej=True,
            metadata={"axes": "CYX", "Labels": ["FKHR-GFP", "DNA"]},
            compression="zlib",
        )

        print(f"{source} -> {destination}")


if __name__ == "__main__":
    main()
