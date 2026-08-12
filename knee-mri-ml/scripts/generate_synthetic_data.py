#!/usr/bin/env python3
"""Generate a synthetic knee-MRI dataset in the MRNet on-disk layout."""

import argparse

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

from knee_mri.data import generate_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic", help="output root directory")
    ap.add_argument("--train", type=int, default=40, help="number of training exams")
    ap.add_argument("--valid", type=int, default=10, help="number of validation exams")
    ap.add_argument("--size", type=int, default=64, help="in-plane resolution (H=W)")
    ap.add_argument("--min-slices", type=int, default=16)
    ap.add_argument("--max-slices", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = generate_dataset(
        out=args.out,
        n_train=args.train,
        n_valid=args.valid,
        size=args.size,
        slices=(args.min_slices, args.max_slices),
        seed=args.seed,
    )
    print(f"Wrote synthetic dataset to {out.resolve()}")
    print(f"  train exams: {args.train} | valid exams: {args.valid}")


if __name__ == "__main__":
    main()
