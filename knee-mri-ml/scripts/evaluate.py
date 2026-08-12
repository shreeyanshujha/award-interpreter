#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the validation split."""

import argparse

import _bootstrap  # noqa: F401

import torch
from torch.utils.data import DataLoader

from knee_mri.config import Config
from knee_mri.data import build_dataset
from knee_mri.engine import evaluate, resolve_device
from knee_mri.engine.metrics import format_metrics
from knee_mri.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    ap.add_argument("--data-root")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # The checkpoint's baked-in config is the source of truth for the model
    # architecture (backbone, planes, tasks, image_size, ...) — it must match the
    # saved weights. Fall back to --config only if the checkpoint lacks a config.
    # CLI flags may override runtime-only fields (data_root, device).
    base = ckpt.get("config")
    if base:
        cfg = Config.from_dict(base, data_root=args.data_root, device=args.device)
    elif args.config:
        cfg = Config.from_yaml(args.config, data_root=args.data_root, device=args.device)
    else:
        raise SystemExit("checkpoint has no embedded config; pass --config explicitly")

    device = resolve_device(cfg.device)
    ds = build_dataset(cfg, args.split, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = build_model(cfg, num_tasks=len(cfg.tasks)).to(device)
    model.load_state_dict(ckpt["model"])

    result = evaluate(model, loader, device, cfg.tasks)
    print(f"Split: {args.split} ({len(ds)} exams)")
    print(format_metrics(result["metrics"]))


if __name__ == "__main__":
    main()
