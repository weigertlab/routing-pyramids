import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "cuda: tests that require CUDA support",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    del config
    if torch.cuda.is_available():
        return

    skip_cuda = pytest.mark.skip(reason="CUDA required")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)


@pytest.fixture(scope="session")
def flex_device() -> torch.device:
    return torch.device("cuda")
