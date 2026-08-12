# Knee MRI ML

A deep-learning pipeline for classifying knee MRI exams from multi-plane
(axial / coronal / sagittal) volumes. The architecture follows the
**MRNet** design (Bien et al., *PLOS Medicine* 2018): a per-slice 2D CNN
backbone extracts features from every slice of a series, features are pooled
across slices, and the three planes are combined into per-exam predictions
for three binary tasks:

| Task          | Meaning                                  |
| ------------- | ---------------------------------------- |
| `abnormal`    | Any abnormality present in the exam      |
| `acl`         | Anterior cruciate ligament (ACL) tear    |
| `meniscus`    | Meniscal tear                            |

> **Data note.** Real knee MRI datasets (e.g. Stanford
> [MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/) or the
> [OAI](https://nda.nih.gov/oai)) are licensed and cannot be redistributed
> here. This repo ships a **synthetic data generator** that writes volumes in
> the exact on-disk layout the real datasets use, so the full pipeline —
> data loading, training, evaluation — runs end to end today. Point the
> config at the real data directory when you have access; no code changes
> needed.

## Layout

```
knee-mri-ml/
├── configs/default.yaml        # single source of truth for hyperparameters
├── src/knee_mri/
│   ├── config.py               # dataclass config + YAML loader
│   ├── data/
│   │   ├── dataset.py          # MRNet-style torch Dataset
│   │   ├── transforms.py       # per-slice augmentation / normalization
│   │   └── synthetic.py        # synthetic volume + label generator
│   ├── models/mrnet.py         # per-slice CNN + slice-pooling classifier
│   ├── engine/
│   │   ├── train.py            # training loop
│   │   ├── evaluate.py         # validation loop + metrics
│   │   └── metrics.py          # AUC / accuracy helpers
│   └── utils/{seed,logging}.py
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── train.py
│   └── evaluate.py
└── tests/                      # pytest smoke + unit tests
```

## On-disk data format

Mirrors the MRNet release. For each split (`train`, `valid`) and each plane:

```
<data_root>/
├── train/
│   ├── axial/0000.npy          # array shape (num_slices, H, W), uint8/float
│   ├── coronal/0000.npy
│   └── sagittal/0000.npy
├── valid/…
├── train-abnormal.csv          # columns: id,label  (label ∈ {0,1})
├── train-acl.csv
├── train-meniscus.csv
├── valid-abnormal.csv
├── valid-acl.csv
└── valid-meniscus.csv
```

Slice counts vary per exam; training therefore uses **batch size 1** (one
exam at a time), which is the standard MRNet approach.

## Quickstart

```bash
cd knee-mri-ml
python -m pip install -r requirements.txt

# 1. Generate a small synthetic dataset with a learnable signal
python scripts/generate_synthetic_data.py --out data/synthetic \
    --train 40 --valid 10 --seed 0

# 2. Train (CPU-friendly defaults; override anything from the CLI)
python scripts/train.py --config configs/default.yaml \
    --data-root data/synthetic --epochs 2 --backbone resnet18

# 3. Evaluate a checkpoint (architecture is read from the checkpoint itself)
python scripts/evaluate.py \
    --data-root data/synthetic --checkpoint runs/latest/best.pt
```

The synthetic labels are correlated with a mean-intensity signal injected
into the volumes, so validation AUC should climb above 0.5 within an epoch —
enough to prove the pipeline learns. On real data, expect the published
MRNet AUCs (abnormality ≈ 0.94, ACL ≈ 0.96, meniscus ≈ 0.85).

## Testing

```bash
pytest -q
```

The suite covers the synthetic generator, the dataset loader, the model's
forward pass (variable slice counts), the metrics, and a one-step training
smoke test.

## Roadmap

- [ ] Swap synthetic loader for a real MRNet/OAI adapter (config-only change).
- [ ] Per-plane model ensembling + logistic-regression fusion (as in the paper).
- [ ] Mixed-precision + GPU training path.
- [ ] Experiment tracking (TensorBoard / Weights & Biases) behind a flag.
- [ ] Grad-CAM slice-level interpretability for clinical review.
