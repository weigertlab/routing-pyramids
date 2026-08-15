from pathlib import Path

import routing_pyramids


def test_public_exports() -> None:
    assert routing_pyramids.__all__ == [
        "LossWeightSchedule",
        "PyramidFlowDecoder2d",
        "PyramidFlowEncoder2d",
        "PyramidFlowSegmentationTestConfig",
        "PyramidFlowSystem",
        "PyramidTransport",
    ]


def test_project_sources_use_only_routing_pyramids_namespace() -> None:
    root = Path(__file__).parents[1]
    checked = [root / "src", root / "examples", root / "tests"]
    checked += [root / "README.md", root / "pyproject.toml"]
    forbidden = "cy" + "far"
    for path in checked:
        files = path.rglob("*.py") if path.is_dir() else (path,)
        for file in files:
            assert forbidden not in file.read_text(), file
