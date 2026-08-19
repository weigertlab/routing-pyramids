# Generative Routing Pyramids

Official implementation of
_Unsupervised Learning of Cell Instances with Generative Routing Pyramids_
([arXiv](http://arxiv.org/abs/2608.16810)).

## Install

Install with pip:

```sh
pip install git+https://github.com/weigertlab/routing-pyramids.git
```

The visualization scripts require optional dependencies from the `[analysis]` extra.

Development installation requires [uv](https://docs.astral.sh/uv):

```sh
# in cloned repository
uv sync --extra analysis
```

## Examples

We provide example scripts for:

- [training](examples/train_fluohela.py)
- [prediction](examples/predict_psc.py)
- [visualizing pyramidal routing and feature maps](examples/visualize_flow_aics_object_posterior.py)
- [latent analysis and instance generation](examples/visualize_bbbc013_latents.py)

Also see the [dataset documentation](data/README.md).
