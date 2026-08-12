"""Generate a synthetic knee-MRI dataset in the MRNet on-disk layout.

Real knee MRI data is licensed and cannot be shipped, so this module fabricates
volumes with a *learnable* signal: label-correlated intensity/texture cues are
injected into the pixels, so a model trained on the output reaches AUC > 0.5 and
the whole pipeline can be validated end to end.

Layout produced (identical to the Stanford MRNet release)::

    <out>/train/{axial,coronal,sagittal}/<id>.npy   # (num_slices, H, W) uint8
    <out>/valid/...
    <out>/{train,valid}-{abnormal,acl,meniscus}.csv # columns: id,label

The three tasks are nested the way they are clinically: an ACL or meniscus tear
implies the exam is abnormal. ``abnormal`` therefore acts as the umbrella label.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PLANES = ("axial", "coronal", "sagittal")
TASKS = ("abnormal", "acl", "meniscus")


def _sample_labels(rng: np.random.Generator) -> Dict[str, int]:
    """Sample a clinically-consistent label triplet for one exam."""
    abnormal = int(rng.random() < 0.5)
    if abnormal:
        acl = int(rng.random() < 0.4)
        meniscus = int(rng.random() < 0.4)
        # An abnormal exam should show *something*; guarantee at least one finding
        # part of the time so `abnormal` isn't perfectly predicted by acl|meniscus.
        if not acl and not meniscus and rng.random() < 0.5:
            if rng.random() < 0.5:
                acl = 1
            else:
                meniscus = 1
    else:
        acl = 0
        meniscus = 0
    return {"abnormal": abnormal, "acl": acl, "meniscus": meniscus}


def _make_volume(
    labels: Dict[str, int],
    rng: np.random.Generator,
    size: int,
    min_slices: int,
    max_slices: int,
) -> np.ndarray:
    """Build one (num_slices, size, size) uint8 volume carrying label signal."""
    num_slices = int(rng.integers(min_slices, max_slices + 1))
    vol = rng.normal(loc=110.0, scale=18.0, size=(num_slices, size, size))

    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size / 2.0

    # `abnormal`: raises overall brightness.
    if labels["abnormal"]:
        vol += 22.0

    # `acl`: a bright Gaussian blob near the intercondylar notch (image centre),
    # present on a subset of central slices.
    if labels["acl"]:
        blob = 60.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (size * 0.12) ** 2)))
        lo, hi = num_slices // 3, max(num_slices // 3 + 1, 2 * num_slices // 3)
        vol[lo:hi] += blob

    # `meniscus`: elevated high-frequency texture in the lower image region.
    if labels["meniscus"]:
        texture = rng.normal(0.0, 26.0, size=(num_slices, size, size))
        mask = (yy > size * 0.6).astype(np.float32)
        vol += texture * mask

    return np.clip(vol, 0, 255).astype(np.uint8)


def generate_dataset(
    out: str | Path,
    n_train: int = 40,
    n_valid: int = 10,
    size: int = 64,
    slices: Tuple[int, int] = (16, 32),
    seed: int = 0,
) -> Path:
    """Write a full synthetic dataset and return the output root path."""
    out = Path(out)
    rng = np.random.default_rng(seed)
    min_slices, max_slices = slices

    for split, n in (("train", n_train), ("valid", n_valid)):
        for plane in PLANES:
            (out / split / plane).mkdir(parents=True, exist_ok=True)

        rows: Dict[str, List[Tuple[str, int]]] = {t: [] for t in TASKS}
        for i in range(n):
            exam_id = f"{i:04d}"
            labels = _sample_labels(rng)
            for plane in PLANES:
                vol = _make_volume(labels, rng, size, min_slices, max_slices)
                np.save(out / split / plane / f"{exam_id}.npy", vol)
            for task in TASKS:
                rows[task].append((exam_id, labels[task]))

        for task in TASKS:
            with open(out / f"{split}-{task}.csv", "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["id", "label"])
                writer.writerows(rows[task])

    return out
