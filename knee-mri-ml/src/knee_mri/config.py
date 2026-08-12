"""Typed configuration for the knee-MRI pipeline.

The :class:`Config` dataclass is the single source of truth. It can be loaded
from a YAML file and then selectively overridden with keyword arguments (used
to wire up command-line flags in the scripts).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    # data
    data_root: str = "data/synthetic"
    dataset_format: str = "mrnet"  # mrnet (3-task, per-plane .npy) or rsna (12-task, weak labels)
    planes: List[str] = field(default_factory=lambda: ["axial", "coronal", "sagittal"])
    tasks: List[str] = field(default_factory=lambda: ["abnormal", "acl", "meniscus"])
    image_size: int = 224
    max_slices: int = 0

    # rsna-specific: 2.5D input construction + weak-supervision weighting
    input_mode: str = "triplets"   # triplets (2.5D, bounded compute) or slices (all slices)
    positions_per_plane: int = 6   # triplet centres sampled per plane
    slice_gap: int = 2             # channel offset within a triplet [c-gap, c, c+gap]
    gold_weight: float = 8.0       # loss weight for gold-labeled studies
    report_weight: float = 1.0     # loss weight for report-derived labels

    # model
    backbone: str = "resnet18"
    pretrained: bool = False
    pool: str = "max"
    dropout: float = 0.5

    # optimization
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 1e-2
    optimizer: str = "adam"
    grad_clip: float = 0.0

    # runtime
    device: str = "auto"
    seed: int = 42
    num_workers: int = 0
    out_dir: str = "runs"
    log_every: int = 10

    def __post_init__(self) -> None:
        if self.dataset_format not in {"mrnet", "rsna"}:
            raise ValueError(f"dataset_format must be 'mrnet' or 'rsna', got {self.dataset_format!r}")
        if self.input_mode not in {"triplets", "slices"}:
            raise ValueError(f"input_mode must be 'triplets' or 'slices', got {self.input_mode!r}")
        if self.pool not in {"max", "avg"}:
            raise ValueError(f"pool must be 'max' or 'avg', got {self.pool!r}")
        if self.backbone not in {"resnet18", "alexnet"}:
            raise ValueError(f"unsupported backbone {self.backbone!r}")
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError(f"unsupported optimizer {self.optimizer!r}")
        if not self.planes:
            raise ValueError("at least one plane is required")
        if not self.tasks:
            raise ValueError("at least one task is required")

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "Config":
        """Load config from ``path``, applying non-None ``overrides`` on top."""
        data = {}
        if path is not None:
            with open(path, "r") as fh:
                data = yaml.safe_load(fh) or {}
        return cls.from_dict(data, **overrides)

    @classmethod
    def from_dict(cls, data: dict, **overrides) -> "Config":
        known = {f.name for f in fields(cls)}
        merged = {k: v for k, v in (data or {}).items() if k in known}
        # Only apply overrides that were actually provided (not None).
        for key, value in overrides.items():
            if value is not None and key in known:
                merged[key] = value
        unknown = set((data or {}).keys()) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**merged)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
