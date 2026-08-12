"""Dataset adapter for the RSNA Knee Abnormality Detection task.

Twelve binary findings per study, scored by macro ROC-AUC. The defining
property of the real training data is *weak supervision*: only a small gold
subset (~58 of 4,407 studies) has image-level labels; the rest carry
free-text radiology reports. This adapter merges both:

- gold studies    → hard 0/1 targets, high confidence weight (``gold_weight``)
- report studies  → soft targets + per-finding weights mined from the report
                     text by :class:`~knee_mri.data.reports.ReportLabeler`

On-disk layout::

    <root>/<split>.csv            # StudyInstanceUID + one column per task
                                  # (cells empty for report-only studies)
    <root>/reports.csv            # StudyInstanceUID, report
    <root>/<split>/<uid>/         # either {axial,coronal,sagittal}.npy
                                  # or DICOM series subdirectories

Items add a ``weights`` tensor next to ``labels``; the training loop uses it
for confidence-weighted BCE.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .reports import RSNA_TASKS, ReportLabeler, soft_targets
from .transforms import volume_to_tensor, volume_to_triplets

PLANES = ("axial", "coronal", "sagittal")


class RSNAKneeDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        tasks: Sequence[str] = RSNA_TASKS,
        planes: Sequence[str] = PLANES,
        image_size: int = 224,
        input_mode: str = "triplets",
        positions_per_plane: int = 6,
        slice_gap: int = 2,
        max_slices: int = 0,
        train: bool = False,
        seed: int = 0,
        gold_weight: float = 8.0,
        report_weight: float = 1.0,
    ) -> None:
        self.root = Path(data_root)
        self.split = split
        self.tasks = list(tasks)
        self.planes = list(planes)
        self.image_size = image_size
        self.input_mode = input_mode
        self.positions_per_plane = positions_per_plane
        self.slice_gap = slice_gap
        self.max_slices = max_slices
        self.train = train
        self._base_seed = seed
        self.gold_weight = gold_weight
        self.report_weight = report_weight

        labels_csv = self.root / f"{split}.csv"
        if not labels_csv.is_file():
            raise FileNotFoundError(f"missing labels file: {labels_csv}")

        reports = self._read_reports(self.root / "reports.csv")
        labeler = ReportLabeler(self.tasks)

        self.ids: List[str] = []
        self._targets: Dict[str, List[float]] = {}
        self._weights: Dict[str, List[float]] = {}
        self._is_gold: Dict[str, bool] = {}
        self.num_gold = 0
        self.num_report = 0

        with open(labels_csv, newline="") as fh:
            reader = csv.DictReader(fh)
            missing = {"StudyInstanceUID", *self.tasks} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{labels_csv} missing columns: {sorted(missing)}")
            for row in reader:
                uid = row["StudyInstanceUID"]
                raw = [row[t].strip() for t in self.tasks]
                if all(v != "" for v in raw):
                    # Gold study: hard labels, high confidence.
                    targets = [float(int(v)) for v in raw]
                    weights = [self.gold_weight] * len(self.tasks)
                    self._is_gold[uid] = True
                    self.num_gold += 1
                elif uid in reports:
                    # Report-only study: mine soft labels from the text.
                    states = labeler.label(reports[uid])
                    targets, weights = soft_targets(
                        states, self.tasks, report_weight=self.report_weight
                    )
                    self._is_gold[uid] = False
                    self.num_report += 1
                else:
                    raise ValueError(
                        f"study {uid} has neither complete labels nor a report"
                    )
                self.ids.append(uid)
                self._targets[uid] = targets
                self._weights[uid] = weights

        for uid in self.ids:
            if not (self.root / split / uid).is_dir():
                raise FileNotFoundError(f"missing study directory: {self.root / split / uid}")

    # ------------------------------------------------------------------
    @staticmethod
    def _read_reports(path: Path) -> Dict[str, str]:
        if not path.is_file():
            return {}
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            return {row["StudyInstanceUID"]: row["report"] for row in reader}

    def _load_volumes(self, uid: str) -> Dict[str, np.ndarray]:
        study_dir = self.root / self.split / uid
        npy = {p: study_dir / f"{p}.npy" for p in self.planes}
        if all(f.is_file() for f in npy.values()):
            return {p: np.load(f) for p, f in npy.items()}
        # Fall back to DICOM series subdirectories.
        from .dicom import load_study_volumes

        volumes = load_study_volumes(study_dir)
        missing = set(self.planes) - set(volumes)
        if missing:
            raise ValueError(f"study {uid} lacks planes: {sorted(missing)}")
        return {p: volumes[p] for p in self.planes}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        uid = self.ids[index]
        rng = np.random.default_rng(self._base_seed + index) if self.train else None

        planes: Dict[str, torch.Tensor] = {}
        for plane, volume in self._load_volumes(uid).items():
            if self.input_mode == "triplets":
                planes[plane] = volume_to_triplets(
                    volume,
                    image_size=self.image_size,
                    positions=self.positions_per_plane,
                    gap=self.slice_gap,
                    train=self.train,
                    rng=rng,
                )
            else:
                planes[plane] = volume_to_tensor(
                    volume,
                    image_size=self.image_size,
                    max_slices=self.max_slices,
                    train=self.train,
                    rng=rng,
                )

        return {
            "planes": planes,
            "labels": torch.tensor(self._targets[uid], dtype=torch.float32),
            "weights": torch.tensor(self._weights[uid], dtype=torch.float32),
            "exam_id": uid,
        }

    def gold_indices(self) -> List[int]:
        """Indices of gold-labeled studies (for gold-only validation metrics)."""
        return [i for i, uid in enumerate(self.ids) if self._is_gold[uid]]
