"""Shared pytest fixtures + path setup."""

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    """A tiny synthetic dataset generated once per test session."""
    from knee_mri.data import generate_dataset

    out = tmp_path_factory.mktemp("synthetic")
    generate_dataset(out=out, n_train=6, n_valid=4, size=32, slices=(4, 8), seed=0)
    return out
