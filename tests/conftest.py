from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _gpu_available() -> bool:
    try:
        import torch
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def pytest_collection_modifyitems(config, items):
    if _gpu_available():
        return
    skip = pytest.mark.skip(reason="CUDA GPU with Triton not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def fixture_text():
    def load(name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")

    return load
