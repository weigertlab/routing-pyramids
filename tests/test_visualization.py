import torch

from routing_pyramids.visualization import pca_colorize


def test_pca_colorize_shape_dtype_and_range() -> None:
    torch.manual_seed(7)
    features = torch.randn(2, 5, 4, 3, dtype=torch.float64)
    rgb = pca_colorize(features)
    assert rgb.shape == (2, 3, 4, 3)
    assert rgb.dtype == torch.float32
    assert torch.isfinite(rgb).all()
    assert float(rgb.min()) >= 0.0
    assert float(rgb.max()) <= 1.0
