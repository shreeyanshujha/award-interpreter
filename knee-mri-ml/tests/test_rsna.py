import csv

import pytest
import torch

from knee_mri.data import RSNA_TASKS, RSNAKneeDataset, generate_rsna_dataset


@pytest.fixture(scope="module")
def rsna_root(tmp_path_factory):
    out = tmp_path_factory.mktemp("rsna")
    generate_rsna_dataset(out=out, n_train=10, n_valid=4, gold_fraction=0.3,
                          size=32, slices=(6, 10), seed=0)
    return out


def test_generator_layout(rsna_root):
    with open(rsna_root / "train.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 10
    assert set(rows[0].keys()) == {"StudyInstanceUID", *RSNA_TASKS}

    # Unlabeled training studies must have a report.
    with open(rsna_root / "reports.csv") as fh:
        report_uids = {r["StudyInstanceUID"] for r in csv.DictReader(fh)}
    for row in rows:
        if any(row[t] == "" for t in RSNA_TASKS):
            assert row["StudyInstanceUID"] in report_uids

    # Validation studies are all gold-labeled.
    with open(rsna_root / "valid.csv") as fh:
        for row in csv.DictReader(fh):
            assert all(row[t] in {"0", "1"} for t in RSNA_TASKS)


def test_dataset_items(rsna_root):
    ds = RSNAKneeDataset(rsna_root, "train", image_size=32,
                         positions_per_plane=4, train=False)
    assert len(ds) == 10
    assert ds.num_gold + ds.num_report == 10
    assert ds.num_report > 0  # weak supervision actually exercised

    item = ds[0]
    assert item["labels"].shape == (12,)
    assert item["weights"].shape == (12,)
    for tensor in item["planes"].values():
        # triplets mode: (positions, 3, D, D)
        assert tensor.shape == (4, 3, 32, 32)
        assert tensor.dtype == torch.float32


def test_gold_vs_report_weights(rsna_root):
    ds = RSNAKneeDataset(rsna_root, "train", image_size=32, gold_weight=8.0,
                         report_weight=1.0, train=False)
    gold = set(ds.gold_indices())
    assert gold  # gold_fraction=0.3 over 10 studies should yield some
    for i in range(len(ds)):
        w = ds[i]["weights"]
        if i in gold:
            assert torch.all(w == 8.0)
        else:
            assert torch.all(w <= 1.0)  # report-derived, soft confidence


def test_slices_input_mode(rsna_root):
    ds = RSNAKneeDataset(rsna_root, "train", image_size=32,
                         input_mode="slices", train=False)
    tensor = next(iter(ds[0]["planes"].values()))
    s, c, h, w = tensor.shape
    assert c == 3 and h == w == 32 and 6 <= s <= 10
