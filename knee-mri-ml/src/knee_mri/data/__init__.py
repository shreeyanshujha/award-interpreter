from .dataset import KneeMRIDataset
from .reports import RSNA_TASKS, ReportLabeler, State, soft_targets
from .rsna import RSNAKneeDataset
from .synthetic import generate_dataset
from .synthetic_rsna import generate_rsna_dataset
from .transforms import volume_to_tensor, volume_to_triplets

__all__ = [
    "KneeMRIDataset",
    "RSNAKneeDataset",
    "RSNA_TASKS",
    "ReportLabeler",
    "State",
    "soft_targets",
    "generate_dataset",
    "generate_rsna_dataset",
    "volume_to_tensor",
    "volume_to_triplets",
    "build_dataset",
]


def build_dataset(cfg, split: str, train: bool):
    """Construct the right dataset for ``cfg.dataset_format``."""
    if cfg.dataset_format == "rsna":
        return RSNAKneeDataset(
            cfg.data_root,
            split,
            tasks=cfg.tasks,
            planes=cfg.planes,
            image_size=cfg.image_size,
            input_mode=cfg.input_mode,
            positions_per_plane=cfg.positions_per_plane,
            slice_gap=cfg.slice_gap,
            max_slices=cfg.max_slices,
            train=train,
            seed=cfg.seed,
            gold_weight=cfg.gold_weight,
            report_weight=cfg.report_weight,
        )
    return KneeMRIDataset(
        cfg.data_root,
        split,
        cfg.planes,
        cfg.tasks,
        image_size=cfg.image_size,
        max_slices=cfg.max_slices,
        train=train,
        seed=cfg.seed,
    )
