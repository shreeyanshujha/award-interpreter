"""Per-task classification metrics."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC that degrades gracefully when a task has a single class present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


def compute_metrics(
    labels: Sequence[Sequence[float]],
    probs: Sequence[Sequence[float]],
    task_names: Sequence[str],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Return per-task AUC and accuracy plus a macro summary.

    ``labels`` and ``probs`` are ``(N, num_tasks)`` arrays.
    """
    y = np.asarray(labels, dtype=np.float32).reshape(len(labels), -1)
    p = np.asarray(probs, dtype=np.float32).reshape(len(probs), -1)
    if y.shape != p.shape:
        raise ValueError(f"labels/probs shape mismatch: {y.shape} vs {p.shape}")

    out: Dict[str, Dict[str, float]] = {}
    aucs: List[float] = []
    accs: List[float] = []
    for i, name in enumerate(task_names):
        auc = _safe_auc(y[:, i], p[:, i])
        acc = float(((p[:, i] >= threshold).astype(np.float32) == y[:, i]).mean())
        out[name] = {"auc": auc, "acc": acc}
        if not np.isnan(auc):
            aucs.append(auc)
        accs.append(acc)

    out["macro"] = {
        "auc": float(np.mean(aucs)) if aucs else float("nan"),
        "acc": float(np.mean(accs)) if accs else float("nan"),
    }
    return out


def format_metrics(metrics: Dict[str, Dict[str, float]]) -> str:
    parts = []
    for name, vals in metrics.items():
        parts.append(f"{name}(auc={vals['auc']:.3f}, acc={vals['acc']:.3f})")
    return "  ".join(parts)
