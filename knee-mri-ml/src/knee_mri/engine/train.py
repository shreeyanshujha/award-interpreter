"""Training loop for the knee-MRI model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

import torch.nn.functional as F

from ..config import Config
from ..data import KneeMRIDataset, build_dataset
from ..models import build_model
from ..utils import get_logger, seed_everything
from .evaluate import _to_device, evaluate


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
    )


def _pos_weight(dataset: KneeMRIDataset, device: torch.device) -> torch.Tensor:
    """Class-imbalance weighting for BCE: (neg / pos) per task, clamped."""
    fracs = dataset.positive_fractions()
    weights = []
    for task in dataset.tasks:
        p = fracs[task]
        weights.append((1 - p) / p if 0 < p < 1 else 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device).clamp(0.1, 10.0)


def train(cfg: Config, run_name: Optional[str] = None) -> Dict[str, object]:
    logger = get_logger()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info("device: %s", device)

    train_ds = build_dataset(cfg, "train", train=True)
    valid_ds = build_dataset(cfg, "valid", train=False)
    logger.info("train exams: %d | valid exams: %d", len(train_ds), len(valid_ds))
    if hasattr(train_ds, "num_gold"):
        logger.info("  gold-labeled: %d | report-labeled: %d",
                    train_ds.num_gold, train_ds.num_report)

    # batch_size is fixed at 1: slice counts vary, so exams can't be stacked.
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=cfg.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    model = build_model(cfg, num_tasks=len(cfg.tasks)).to(device)
    optimizer = _build_optimizer(model, cfg)
    if isinstance(train_ds, KneeMRIDataset):
        # Hard 0/1 labels: class-imbalance-corrected BCE.
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=_pos_weight(train_ds, device))
    else:
        # RSNA weak supervision: per-sample confidence weights carried by the
        # dataset (gold >> report-derived >> unmentioned); plain BCE for val.
        criterion = torch.nn.BCEWithLogitsLoss()

    out_dir = Path(cfg.out_dir) / (run_name or "latest")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)

    history: List[Dict[str, object]] = []
    best_auc = -1.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            planes = _to_device(batch["planes"], device)
            labels = batch["labels"].squeeze(0).to(device)

            optimizer.zero_grad()
            logits = model(planes)
            if "weights" in batch:
                weights = batch["weights"].squeeze(0).to(device)
                loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
            else:
                loss = criterion(logits, labels)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            running += loss.item()
            if step % cfg.log_every == 0:
                logger.info("epoch %d | step %d/%d | loss %.4f",
                            epoch, step, len(train_loader), running / step)

        val = evaluate(model, valid_loader, device, cfg.tasks, criterion)
        macro_auc = val["metrics"]["macro"]["auc"]
        logger.info(
            "epoch %d | train_loss %.4f | val_loss %.4f | val_macro_auc %.4f",
            epoch, running / max(1, len(train_loader)), val["loss"], macro_auc,
        )
        history.append({"epoch": epoch, "train_loss": running / max(1, len(train_loader)),
                        "val_loss": val["loss"], "metrics": val["metrics"]})

        torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "epoch": epoch},
                   out_dir / "last.pt")
        # NaN AUC (single-class val split) shouldn't count as an improvement.
        if macro_auc == macro_auc and macro_auc > best_auc:
            best_auc = macro_auc
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "epoch": epoch},
                       out_dir / "best.pt")
            logger.info("  new best macro AUC %.4f -> best.pt", best_auc)

    with open(out_dir / "history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    return {"best_auc": best_auc, "history": history, "out_dir": str(out_dir)}
