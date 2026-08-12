"""Synthetic data in the RSNA Knee Abnormality Detection layout.

Mirrors the competition's defining properties so the full pipeline can be
exercised without the licensed data:

- 12 binary findings per study,
- only a fraction of training studies carry gold labels; the rest get a
  templated English radiology report (the real reports span 12 languages),
- validation studies are all gold-labeled (matching how the competition
  scores on radiologist-annotated ground truth).

Each finding injects a bright Gaussian blob at a task-specific location (a
4×3 grid over the image), so every label has a distinct learnable signature.
Reports are noisy on purpose: a positive finding is occasionally left
unmentioned, mimicking the ~82% report/gold agreement in the real data.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .reports import RSNA_TASKS

PLANES = ("axial", "coronal", "sagittal")

# Per-task positive prevalence, loosely shaped like a knee-MRI case mix.
_PREVALENCE: Dict[str, float] = {
    "acl": 0.25, "mcl": 0.20, "medial_meniscus": 0.35, "lateral_meniscus": 0.25,
    "medial_oa": 0.30, "lateral_oa": 0.20, "pf_oa": 0.25, "effusion": 0.40,
    "synovitis": 0.20, "bakers_cyst": 0.25, "contusion": 0.20, "fracture": 0.10,
}

# (positive sentence, negative sentence) templates per task.
_REPORT_TEMPLATES: Dict[str, Tuple[str, str]] = {
    "acl": ("There is a tear of the anterior cruciate ligament.",
            "The anterior cruciate ligament is intact."),
    "mcl": ("Sprain of the medial collateral ligament.",
            "The medial collateral ligament is intact."),
    "medial_meniscus": ("Tear of the posterior horn of the medial meniscus.",
                        "The medial meniscus is normal."),
    "lateral_meniscus": ("Tear of the lateral meniscus.",
                         "The lateral meniscus is normal."),
    "medial_oa": ("Medial compartment osteoarthritis with medial joint space narrowing.",
                  "No medial compartment osteoarthritis."),
    "lateral_oa": ("Lateral compartment osteoarthritis.",
                   "No lateral compartment osteoarthritis."),
    "pf_oa": ("Chondromalacia patellae with patellar cartilage thinning.",
              "No patellofemoral osteoarthritis."),
    "effusion": ("Moderate joint effusion.", "No joint effusion."),
    "synovitis": ("Synovitis with synovial thickening.", "No synovitis."),
    "bakers_cyst": ("A Baker's cyst is present.", "No popliteal cyst."),
    "contusion": ("Bone marrow edema compatible with bone contusion.",
                  "No bone contusion."),
    "fracture": ("Nondisplaced fracture of the tibial plateau.", "No fracture."),
}


def _blob_centre(task_index: int, size: int) -> Tuple[float, float]:
    """Deterministic per-task blob location on a 4×3 grid."""
    row, col = divmod(task_index, 3)
    cy = (row + 0.5) / 4 * size
    cx = (col + 0.5) / 3 * size
    return cy, cx


def _make_volume(
    labels: Dict[str, int], rng: np.random.Generator, size: int,
    min_slices: int, max_slices: int,
) -> np.ndarray:
    num_slices = int(rng.integers(min_slices, max_slices + 1))
    vol = rng.normal(loc=110.0, scale=18.0, size=(num_slices, size, size))
    yy, xx = np.mgrid[0:size, 0:size]

    for i, task in enumerate(RSNA_TASKS):
        if not labels[task]:
            continue
        cy, cx = _blob_centre(i, size)
        # Span all slices so any sampled triplet window carries the signal.
        blob = 80.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (size * 0.13) ** 2)))
        vol += blob

    return np.clip(vol, 0, 255).astype(np.uint8)


def _make_report(labels: Dict[str, int], rng: np.random.Generator) -> str:
    sentences: List[str] = ["MRI of the knee."]
    for task in RSNA_TASKS:
        pos_s, neg_s = _REPORT_TEMPLATES[task]
        if labels[task]:
            # ~10% of positives go unmentioned → report/gold disagreement.
            if rng.random() >= 0.10:
                sentences.append(pos_s)
        else:
            # Negatives are mentioned only about half the time.
            if rng.random() < 0.5:
                sentences.append(neg_s)
    return " ".join(sentences)


def generate_rsna_dataset(
    out: str | Path,
    n_train: int = 40,
    n_valid: int = 16,
    gold_fraction: float = 0.2,
    size: int = 64,
    slices: Tuple[int, int] = (16, 32),
    seed: int = 0,
) -> Path:
    """Write an RSNA-layout synthetic dataset; returns the output root."""
    out = Path(out)
    rng = np.random.default_rng(seed)
    min_slices, max_slices = slices
    report_rows: List[Tuple[str, str]] = []

    for split, n, all_gold in (("train", n_train, False), ("valid", n_valid, True)):
        rows: List[Dict[str, str]] = []
        for i in range(n):
            uid = f"{split}-{i:04d}"
            labels = {t: int(rng.random() < _PREVALENCE[t]) for t in RSNA_TASKS}
            # The real dataset guarantees a (small) gold subset: force the first
            # study gold so a nonzero gold_fraction can never round down to zero.
            gold = all_gold or (i == 0 and gold_fraction > 0) or rng.random() < gold_fraction

            study_dir = out / split / uid
            study_dir.mkdir(parents=True, exist_ok=True)
            for plane in PLANES:
                np.save(study_dir / f"{plane}.npy",
                        _make_volume(labels, rng, size, min_slices, max_slices))

            row = {"StudyInstanceUID": uid}
            for t in RSNA_TASKS:
                row[t] = str(labels[t]) if gold else ""
            rows.append(row)
            if not gold:
                report_rows.append((uid, _make_report(labels, rng)))

        with open(out / f"{split}.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["StudyInstanceUID", *RSNA_TASKS])
            writer.writeheader()
            writer.writerows(rows)

    with open(out / "reports.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["StudyInstanceUID", "report"])
        writer.writerows(report_rows)

    return out
