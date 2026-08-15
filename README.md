# Routing Pyramids

This package contains the generative routing-pyramid models and experiment
entry points for unsupervised instance segmentation described in the
[Routing Pyramids paper project](https://github.com/weigertlab/routing-pyramids).

## Install

Create the development environment with `uv sync`. Publication-analysis
dependencies are installed with `uv sync --group analysis`. Without uv, install
the analysis requirements and the Weigert Lab plotting package with
`pip install git+https://github.com/maweigert/betterplots`.

A wheel can be built with `uv build --no-sources` and installed with pip.

## Experiment data

The examples keep the source scripts' constant-based configuration. Set the
path constants near the top of a script before running it; the numerical paper
configuration remains fixed in the script.

| Experiment | Input | Scale | Training split | Prediction split | Original run |
|---|---|---:|---|---|---|
| AICS nuclear morphology | one-channel OME-Zarr timelapses | 0.5 | small + large colonies | medium colony | `dim_256_8x8_64-fg_1e-2-bg_5e-2-flow_5e-3-entropy_0-sparsity_5e-1` |
| Fluo-N2DL-HeLa | one-channel CTC TIFF sequences | 1.0 | CTC `test` | CTC `test` | `dim_256_8x8_64-fg_1e-2-bg_5e-2-flow_5e-3-entropy_0-sparsity_2e-1-200ep` |
| PhC-C2DL-PSC | one-channel CTC TIFF sequences | 2.0 | CTC `test` | CTC `train` | `dim_256_8x8_64-fg_1e-2-bg_5e-2-flow_5e-3-entropy_0-sparsity_5e-1` |
| BBBC013 | two-channel fluorescence TIFF pairs | 1.0 | columns 2–11 | columns 1 and 12 | `dim_256_8x8_64-fg_1e-2-bg_5e-2-flow_5e-3-entropy_0-sparsity_5e-1` |

Acquire AICS nuclear-morphology data from the [Allen Cell Explorer](https://www.allencell.org/), Fluo-N2DL-HeLa and PhC-C2DL-PSC from the [Cell Tracking Challenge](https://celltrackingchallenge.net/2d-datasets/), and BBBC013 from the [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC013).

Expected layouts are:

```text
AICS_ROOT/<colony>.ome.zarr/
CTC_ROOT/{train,test}/{01,02}/t<frame>.tif
BBBC013_ROOT/<row><column>.tif
BBBC013_PLATE_MAP.csv
```

The AICS scripts use the small and large colonies for training and the medium
colony for validation/prediction. The CTC scripts preserve the split assignments
shown above. BBBC013 TIFFs contain the two fluorescence channels in CYX order;
the analysis additionally requires the published plate-map CSV.

For example:

```bash
uv run python examples/train_pyramid_fluohela.py
uv run python examples/predict_pyramid_fluohela.py
```

The other experiment entry points follow the same design:

```bash
uv run python examples/train_pyramid_aics.py
uv run python examples/train_pyramid_psc.py
uv run python examples/train_pyramid_bbbc013.py
uv run python examples/predict_pyramid_aics.py
uv run python examples/predict_pyramid_psc.py
uv run python examples/visualize_pyramid_bbbc013_latents.py
uv run python examples/visualize_pyramid_flow_aics_object_posterior.py
```

Training writes ordinary Lightning checkpoints. Prediction and analysis scripts
load checkpoints produced by these training scripts using the same explicit
encoder and decoder construction. Segmentation prediction writes one label TIFF
per frame below CTC-style `<sequence>_RES` directories, with `res_track.txt`
alongside each sequence. The BBBC013 analysis writes
`bbbc013_gmm_latents.pdf` and `bbbc013_gmm_images.pdf`; the AICS analysis writes
its figure below its configured output directory.

The included scripts reproduce the model and preprocessing configurations, not
the paper's reported F1/PQ table or combined segmentation figure. Exact results
also depend on obtaining the original datasets and running on compatible
hardware and software.
