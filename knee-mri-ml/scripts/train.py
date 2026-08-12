#!/usr/bin/env python3
"""Train the knee-MRI model. CLI flags override the YAML config."""

import argparse

import _bootstrap  # noqa: F401

from knee_mri.config import Config
from knee_mri.engine import train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="path to a YAML config file")
    ap.add_argument("--run-name", default="latest", help="subdir under out_dir for this run")
    # Overrides (None => fall back to the config / defaults).
    ap.add_argument("--data-root")
    ap.add_argument("--dataset-format", choices=["mrnet", "rsna"], dest="dataset_format")
    ap.add_argument("--input-mode", choices=["triplets", "slices"], dest="input_mode")
    ap.add_argument("--backbone", choices=["resnet18", "alexnet"])
    ap.add_argument("--pool", choices=["max", "avg"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--image-size", type=int, dest="image_size")
    ap.add_argument("--max-slices", type=int, dest="max_slices")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--pretrained", action="store_true", default=None)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k not in {"config", "run_name"}}
    cfg = Config.from_yaml(args.config, **overrides)
    result = train(cfg, run_name=args.run_name)
    print(f"Done. Best macro AUC: {result['best_auc']:.4f}  ->  {result['out_dir']}")


if __name__ == "__main__":
    main()
