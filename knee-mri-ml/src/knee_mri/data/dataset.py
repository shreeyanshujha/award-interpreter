"""MRNet-style dataset: one item = one exam across all requested planes.

Because the number of slices varies per exam, each item is returned on its own
(the DataLoader uses ``batch_size=1``). An item is:

    planes:  dict[str, Tensor(S_plane, 3, D, D)]   # one tensor per plane
    labels:  Tensor(num_tasks,)                    # float 0/1 per task
    exam_id: str
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import volume_to_tensor


def _read_labels(csv_path: Path) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "id" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must have 'id' and 'label' columns")
        for row in reader:
            labels[str(row["id"])] = int(row["label"])
    return labels


class KneeMRIDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        planes: Sequence[str] = ("axial", "coronal", "sagittal"),
        tasks: Sequence[str] = ("abnormal", "acl", "meniscus"),
        image_size: int = 224,
        max_slices: int = 0,
        train: bool = False,
        seed: int = 0,
    ) -> None:
        self.root = Path(data_root)
        self.split = split
        self.planes = list(planes)
        self.tasks = list(tasks)
        self.image_size = image_size
        self.max_slices = max_slices
        self.train = train
        self._base_seed = seed

        if not (self.root / split).is_dir():
            raise FileNotFoundError(f"missing split directory: {self.root / split}")

        # Load per-task label maps and confirm they cover the same exam ids.
        self._labels: Dict[str, Dict[str, int]] = {}
        for task in self.tasks:
            self._labels[task] = _read_labels(self.root / f"{split}-{task}.csv")

        first_task = self.tasks[0]
        self.ids: List[str] = sorted(self._labels[first_task].keys())
        for task in self.tasks[1:]:
            if set(self._labels[task]) != set(self.ids):
                raise ValueError(f"label id mismatch between {first_task} and {task}")

        # Confirm the volume files exist for every id/plane.
        for exam_id in self.ids:
            for plane in self.planes:
                path = self.root / split / plane / f"{exam_id}.npy"
                if not path.is_file():
                    raise FileNotFoundError(f"missing volume: {path}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        exam_id = self.ids[index]
        # Deterministic-yet-varied augmentation stream per item.
        rng = np.random.default_rng(self._base_seed + index) if self.train else None

        planes: Dict[str, torch.Tensor] = {}
        for plane in self.planes:
            volume = np.load(self.root / self.split / plane / f"{exam_id}.npy")
            planes[plane] = volume_to_tensor(
                volume,
                image_size=self.image_size,
                max_slices=self.max_slices,
                train=self.train,
                rng=rng,
            )

        labels = torch.tensor(
            [float(self._labels[task][exam_id]) for task in self.tasks],
            dtype=torch.float32,
        )
        return {"planes": planes, "labels": labels, "exam_id": exam_id}

    # Fraction of positives per task — handy for pos_weight in BCE loss.
    def positive_fractions(self) -> Dict[str, float]:
        n = len(self.ids)
        return {
            task: (sum(self._labels[task].values()) / n if n else 0.0)
            for task in self.tasks
        }
