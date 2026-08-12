#!/usr/bin/env python3
"""Generate a synthetic knee-MRI dataset.

Two layouts:
  mrnet — 3 tasks (abnormal/acl/meniscus), per-plane .npy, fully labeled
  rsna  — 12 findings, per-study directories, mostly report-labeled
          (mimics the RSNA Knee Abnormality Detection weak-supervision setup)
"""

import argparse

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

from knee_mri.data import generate_dataset, generate_rsna_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", choices=["mrnet", "rsna"], default="mrnet")
    ap.add_argument("--out", default=None, help="output root (default: data/<format>)")
    ap.add_argument("--train", type=int, default=40, help="number of training exams")
    ap.add_argument("--valid", type=int, default=10, help="number of validation exams")
    ap.add_argument("--gold-fraction", type=float, default=0.2,
                    help="rsna only: fraction of training studies with gold labels")
    ap.add_argument("--size", type=int, default=64, help="in-plane resolution (H=W)")
    ap.add_argument("--min-slices", type=int, default=16)
    ap.add_argument("--max-slices", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.out or f"data/{args.format}"
    if args.format == "rsna":
        out = generate_rsna_dataset(
            out=out, n_train=args.train, n_valid=args.valid,
            gold_fraction=args.gold_fraction, size=args.size,
            slices=(args.min_slices, args.max_slices), seed=args.seed,
        )
    else:
        out = generate_dataset(
            out=out, n_train=args.train, n_valid=args.valid, size=args.size,
            slices=(args.min_slices, args.max_slices), seed=args.seed,
        )
    print(f"Wrote synthetic {args.format} dataset to {out.resolve()}")
    print(f"  train exams: {args.train} | valid exams: {args.valid}")


if __name__ == "__main__":
    main()
