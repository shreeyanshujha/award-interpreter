"""Validation / inference loop."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .metrics import compute_metrics


def _to_device(planes: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    # DataLoader with batch_size=1 adds a leading batch dim; drop it so each
    # plane is (S, 3, D, D) as the model expects.
    return {p: t.squeeze(0).to(device) for p, t in planes.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_names: List[str],
    criterion: torch.nn.Module | None = None,
) -> Dict[str, object]:
    """Run the model over ``loader`` and return metrics + probabilities."""
    model.eval()
    all_labels: List[List[float]] = []
    all_probs: List[List[float]] = []
    total_loss = 0.0
    n = 0

    for batch in loader:
        planes = _to_device(batch["planes"], device)
        labels = batch["labels"].squeeze(0).to(device)  # (num_tasks,)
        logits = model(planes)                           # (num_tasks,)
        if criterion is not None:
            total_loss += float(criterion(logits, labels))
        probs = torch.sigmoid(logits)
        all_labels.append(labels.cpu().tolist())
        all_probs.append(probs.cpu().tolist())
        n += 1

    metrics = compute_metrics(all_labels, all_probs, task_names)
    return {
        "metrics": metrics,
        "loss": total_loss / n if n else float("nan"),
        "labels": all_labels,
        "probs": all_probs,
    }
