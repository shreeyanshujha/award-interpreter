import torch

from knee_mri.data import KneeMRIDataset


def test_item_shapes(synthetic_root):
    ds = KneeMRIDataset(synthetic_root, "train", image_size=32, train=False)
    assert len(ds) == 6

    item = ds[0]
    assert set(item["planes"]) == {"axial", "coronal", "sagittal"}
    for plane_tensor in item["planes"].values():
        s, c, h, w = plane_tensor.shape
        assert c == 3 and h == 32 and w == 32 and s >= 1
        assert plane_tensor.dtype == torch.float32
    assert item["labels"].shape == (3,)


def test_max_slices_caps_depth(synthetic_root):
    ds = KneeMRIDataset(synthetic_root, "train", image_size=32, max_slices=3, train=False)
    for tensor in ds[0]["planes"].values():
        assert tensor.shape[0] <= 3


def test_positive_fractions(synthetic_root):
    ds = KneeMRIDataset(synthetic_root, "train", train=False)
    fracs = ds.positive_fractions()
    assert set(fracs) == {"abnormal", "acl", "meniscus"}
    assert all(0.0 <= v <= 1.0 for v in fracs.values())
