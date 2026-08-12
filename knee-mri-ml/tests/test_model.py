import pytest
import torch

from knee_mri.models import KneeMRIModel, build_model
from knee_mri.config import Config


def _fake_exam(slice_counts, size=32):
    return {
        plane: torch.randn(s, 3, size, size)
        for plane, s in slice_counts.items()
    }


def test_forward_output_shape():
    model = KneeMRIModel(backbone="resnet18", num_tasks=3)
    out = model(_fake_exam({"axial": 5, "coronal": 7, "sagittal": 3}))
    assert out.shape == (3,)


def test_handles_variable_slice_counts():
    model = KneeMRIModel(backbone="alexnet", num_tasks=3)
    # Different slice counts across two forward passes must both work.
    out_a = model(_fake_exam({"axial": 2, "coronal": 2, "sagittal": 2}, size=64))
    out_b = model(_fake_exam({"axial": 9, "coronal": 4, "sagittal": 6}, size=64))
    assert out_a.shape == out_b.shape == (3,)


def test_missing_plane_raises():
    model = KneeMRIModel(planes=["axial", "coronal", "sagittal"], num_tasks=3)
    with pytest.raises(KeyError):
        model({"axial": torch.randn(3, 3, 32, 32)})


def test_build_from_config():
    cfg = Config(backbone="resnet18", tasks=["abnormal"], pool="avg")
    model = build_model(cfg)
    out = model(_fake_exam({p: 3 for p in cfg.planes}))
    assert out.shape == (1,)
