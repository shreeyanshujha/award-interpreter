import csv

import numpy as np

from knee_mri.data.synthetic import PLANES, TASKS


def _read_csv(path):
    with open(path) as fh:
        return {row["id"]: int(row["label"]) for row in csv.DictReader(fh)}


def test_layout_and_files(synthetic_root):
    for split, n in (("train", 6), ("valid", 4)):
        for plane in PLANES:
            files = sorted((synthetic_root / split / plane).glob("*.npy"))
            assert len(files) == n
            vol = np.load(files[0])
            assert vol.ndim == 3
            assert vol.dtype == np.uint8
        for task in TASKS:
            labels = _read_csv(synthetic_root / f"{split}-{task}.csv")
            assert len(labels) == n
            assert set(labels.values()) <= {0, 1}


def test_labels_clinically_consistent(synthetic_root):
    # A torn ACL or meniscus implies the exam is abnormal.
    ab = _read_csv(synthetic_root / "train-abnormal.csv")
    acl = _read_csv(synthetic_root / "train-acl.csv")
    men = _read_csv(synthetic_root / "train-meniscus.csv")
    for eid in ab:
        if acl[eid] or men[eid]:
            assert ab[eid] == 1


def test_signal_is_present(synthetic_root):
    # Abnormal exams were made brighter — the injected signal should be detectable.
    ab = _read_csv(synthetic_root / "train-abnormal.csv")
    means = {"0": [], "1": []}
    for eid, label in ab.items():
        vol = np.load(synthetic_root / "train" / "axial" / f"{eid}.npy")
        means[str(label)].append(vol.mean())
    if means["0"] and means["1"]:
        assert np.mean(means["1"]) > np.mean(means["0"])
