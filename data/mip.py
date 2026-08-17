# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "iohub>=0.3.9",
#     "zarr>=3.2.1",
# ]
# ///

from argparse import ArgumentParser
from pathlib import Path

from dask.diagnostics import ProgressBar
from iohub import open_ome_zarr


def convert_ome_zarr(old_store_path: Path, new_store_path: Path) -> None:
    with open_ome_zarr(old_store_path, mode="r", layout="fov") as old_dataset:
        new_axes: list = old_dataset.axes
        _ = new_axes.pop(2)
        with open_ome_zarr(
            new_store_path,
            layout="fov",
            mode="w",
            channel_names=old_dataset.channel_names[0:1],
            version="0.5",
            axes=new_axes,
        ) as new_dataset:
            print("Computing maximum intensity projection...")
            with ProgressBar():
                mip = old_dataset["0"].dask_array()[:, 0:1].max(axis=2).compute()
            print("Writing...")
            _ = new_dataset.create_image(
                "0",
                data=mip,
                chunks=(1, 1, *(s // 2 for s in mip.shape[-2:])),
                shards_ratio=(8, 1, 2, 2),
                transform=old_dataset.metadata.multiscales[0]
                .datasets[0]
                .coordinate_transformations,
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("old_store_path", type=Path, help="Old OME-Zarr store")
    parser.add_argument("new_store_path", type=Path, help="New OME-Zarr store")
    args = parser.parse_args()

    convert_ome_zarr(args.old_store_path, args.new_store_path)
