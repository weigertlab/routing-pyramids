# Datasets

This directory contains PEP-723 scripts for converting and preprocessing datasets.

| Dataset | Format | Scale | Training split | Test split | Preprocessing |
| --- | --- | ---: | --- | --- | --- |
| [AICS nuclear morphology] | OME-Zarr | 0.5 | small + large | medium | [max-intensity projection] |
| [Fluo-N2DL-HeLa] | TIFF | 1.0 | CTC `test` | CTC `test` | none |
| [PhC-C2DL-PSC] | TIFF | 2.0 | CTC `test` | CTC `train` | none |
| [BBBC013] | TIFF | 1.0 | columns 2–11 | columns 1 and 12 | [FRM to TIFF conversion] |

Place them in the expected layouts:

```text
AICS_ROOT/<timelapse>.ome.zarr/
CTC_ROOT/{train,test}/{01,02}/t<frame>.tif
BBBC013_ROOT/BBBC013_v1_images_converted/<row><column>.tif
```

We additionally provide a [plate map] converted from the [BBBC013] description.

[AICS nuclear morphology]: https://open.quiltdata.com/b/allencell/tree/aics/nuc-morph-dataset/hipsc_fov_nuclei_timelapse_dataset/hipsc_fov_nuclei_timelapse_data_used_for_analysis/baseline_colonies_fov_timelapse_dataset/
[Fluo-N2DL-HeLa]: https://celltrackingchallenge.net/2d-datasets/
[PhC-C2DL-PSC]: https://celltrackingchallenge.net/2d-datasets/
[BBBC013]: https://bbbc.broadinstitute.org/BBBC013

[max-intensity projection]: ./mip.py
[FRM to TIFF conversion]: ./frm_to_tiff.py
[plate map]: ./BBBC013_platemap_long_nM.csv
