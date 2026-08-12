# Knee MRI ML

A deep-learning pipeline for classifying knee MRI exams from multi-plane
(axial / coronal / sagittal) volumes, targeting the
**[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)**
Kaggle challenge (2026): predict **twelve binary findings** per study, scored
by **macro ROC-AUC**.

| # | Task               | Finding                                |
|---|--------------------|----------------------------------------|
| 1 | `acl`              | Anterior cruciate ligament abnormality |
| 2 | `mcl`              | Medial collateral ligament abnormality |
| 3 | `medial_meniscus`  | Medial meniscus tear                   |
| 4 | `lateral_meniscus` | Lateral meniscus tear                  |
| 5 | `medial_oa`        | Medial tibiofemoral osteoarthritis     |
| 6 | `lateral_oa`       | Lateral tibiofemoral osteoarthritis    |
| 7 | `pf_oa`            | Patellofemoral osteoarthritis          |
| 8 | `effusion`         | Joint effusion                         |
| 9 | `synovitis`        | Synovitis                              |
| 10| `bakers_cyst`      | Baker (popliteal) cyst                 |
| 11| `contusion`        | Bone contusion / marrow edema          |
| 12| `fracture`         | Fracture                               |

The architecture follows the **MRNet** design (Bien et al., *PLOS Medicine*
2018) — per-slice 2D CNN backbone → pooling across slices → multi-plane
fusion — extended for the RSNA setup:

- **DICOM ingestion** (`data/dicom.py`): plane inference from
  `ImageOrientationPatient`, primary-series selection per plane (preferring
  fluid-sensitive fat-suppressed sequences), slice sorting along the slice
  normal via `ImagePositionPatient`.
- **2.5D triplet inputs** (`input_mode: triplets`): instead of feeding every
  slice, sample N centre positions per plane and stack `[c−gap, c, c+gap]`
  as the three channels — bounded compute per exam (the competition caps
  offline inference at 9 hours and awards efficiency).
- **Weak supervision** (`data/reports.py`): only ~1% of real training
  studies carry gold image-level labels; the rest have free-text radiology
  reports (12 languages). A rule-based labeler maps each report to
  positive / explicit-negative / unmentioned per finding; these become
  *soft targets* with *confidence weights* (gold ≈ 8× a report label,
  unmentioned findings further down-weighted) for weighted-BCE training.

The original 3-task MRNet mode (`abnormal` / `acl` / `meniscus`) is still
available via `dataset_format: mrnet`.

> **Data note.** The competition DICOM data lives on Kaggle and is not
> redistributable; MRNet/OAI are licensed too. This repo ships **synthetic
> generators for both layouts** — volumes with learnable per-finding signals
> plus templated (deliberately noisy) radiology reports — so the entire
> pipeline runs end to end today. Point `data_root` at real data when you
> have access; no code changes needed.

## Layout

```
knee-mri-ml/
├── configs/
│   ├── default.yaml            # MRNet 3-task mode
│   └── rsna.yaml               # RSNA 12-finding mode (weak supervision, 2.5D)
├── src/knee_mri/
│   ├── config.py               # dataclass config + YAML loader
│   ├── data/
│   │   ├── dataset.py          # MRNet-style torch Dataset
│   │   ├── rsna.py             # RSNA Dataset: gold + report-derived labels
│   │   ├── reports.py          # report → finding-state labeler + soft targets
│   │   ├── dicom.py            # DICOM scan / series selection / sorted loading
│   │   ├── transforms.py       # per-slice pipeline + 2.5D triplet sampling
│   │   ├── synthetic.py        # synthetic MRNet-layout generator
│   │   └── synthetic_rsna.py   # synthetic RSNA-layout generator (+ reports)
│   ├── models/mrnet.py         # per-slice CNN + slice-pooling classifier
│   ├── engine/
│   │   ├── train.py            # training loop (confidence-weighted BCE)
│   │   ├── evaluate.py         # validation loop + metrics
│   │   └── metrics.py          # per-task AUC / accuracy, macro summary
│   └── utils/{seed,logging}.py
├── scripts/
│   ├── generate_synthetic_data.py   # --format mrnet|rsna
│   ├── train.py
│   └── evaluate.py
└── tests/                      # pytest smoke + unit tests
```

## On-disk data formats

**RSNA layout** (`dataset_format: rsna`):

```
<data_root>/
├── train.csv                   # StudyInstanceUID + one column per finding
│                               # (cells empty for report-only studies)
├── valid.csv                   # validation studies are fully gold-labeled
├── reports.csv                 # StudyInstanceUID, report  (free text)
├── train/<uid>/                # either axial.npy / coronal.npy / sagittal.npy
│                               # or DICOM series subdirectories (real data)
└── valid/<uid>/…
```

**MRNet layout** (`dataset_format: mrnet`) mirrors the Stanford release:
`train/<plane>/<id>.npy` volumes + `train-<task>.csv` label files.

Slice counts vary per exam; training therefore uses **batch size 1** (one
exam at a time), which is the standard MRNet approach.

## Quickstart (RSNA mode)

```bash
cd knee-mri-ml
python -m pip install -r requirements.txt

# 1. Generate a synthetic RSNA-layout dataset (12 findings, mostly
#    report-labeled, learnable per-finding signals)
python scripts/generate_synthetic_data.py --format rsna \
    --train 64 --valid 16 --seed 0

# 2. Train with weak supervision (CPU-friendly demo settings)
python scripts/train.py --config configs/rsna.yaml --data-root data/rsna \
    --backbone alexnet --image-size 64 --epochs 6 --lr 2e-4

# 3. Evaluate (architecture is read from the checkpoint itself)
python scripts/evaluate.py --data-root data/rsna \
    --checkpoint runs/latest/best.pt
```

On the synthetic demo above, validation macro AUC climbs from chance to
**≈ 0.64 within 6 CPU epochs** — proof the weak-supervision pipeline learns.
(For scale: the public-leaderboard baseline the approach mirrors scores
≈ 0.66 on the real competition data.)

For the original MRNet 3-task mode, use `--format mrnet` /
`configs/default.yaml`; on real MRNet data expect the published AUCs
(abnormality ≈ 0.94, ACL ≈ 0.96, meniscus ≈ 0.85).

## Testing

```bash
pytest -q
```

The suite covers the synthetic generator, the dataset loader, the model's
forward pass (variable slice counts), the metrics, and a one-step training
smoke test.

## Roadmap

- [x] RSNA 12-finding task set, macro ROC-AUC scoring
- [x] DICOM ingestion: plane inference, primary-series selection, slice sorting
- [x] 2.5D triplet inputs for bounded per-exam compute
- [x] Report-derived weak labels with confidence-weighted soft targets
- [ ] Kaggle inference notebook: DICOM → predictions CSV inside the 9 h offline budget
- [ ] Broader multilingual term tables (12 report languages) or a small
      multilingual text model as the report labeler
- [ ] Empirical-Bayes calibration of report labels against the gold subset
      (per-fold `positive > unmentioned > explicit-negative` ordering)
- [ ] EfficientNet-B0 backbone + ImageNet pretraining (competition-strength)
- [ ] Per-plane ensembling; five-fold CV ensemble
- [ ] Mixed-precision + GPU training path
- [ ] Experiment tracking (TensorBoard / W&B) behind a flag
- [ ] Grad-CAM slice-level interpretability for clinical review
