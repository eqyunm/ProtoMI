# ProtoMI

**Prototype-Level Mutual Information Maximization with Redundancy Reduction for Incomplete Multi-View Clustering**

A non-contrastive framework for cross-view alignment under arbitrary missing patterns.

---

## Overview

Incomplete Multi-View Clustering (IMVC) groups data when some views are entirely missing for certain samples. The dominant approach—contrastive learning (InfoNCE)—suffers from four structural problems in this setting: false negative contamination, paired-data dependency, task-alignment gap, and implicit uniformity bias.

ProtoMI replaces contrastive learning with two complementary objectives:

1. **Prototype-level Mutual Information (PMI)** — directly maximizes MI between cluster assignment distributions across views, without constructing any positive/negative pairs.
2. **Feature-level Redundancy Reduction (RR)** — prevents representation collapse without negative samples, via a Barlow-Twins-style cross-correlation penalty.

A third component, **Prototype-Guided Distillation (PD)**, propagates semantic knowledge to samples observed in only one view by mapping their latent representation to a pseudo-assignment in the missing view's prototype space.

---

## Method

### Architecture

```
Input x_i^(v)  (sample i, view v; may be missing)
        │
   ┌────┴────┐
   Encoder_v       (view-specific MLP, d_v → 1024 → 512 → 256 → 128)
        │
       h_i^(v)     (latent representation, d_z = 128)
        │
   ┌────┴────┐
   Clustering Head (shared, 128 → 256 → K, temperature-scaled softmax)
        │
       q_i^(v)     (soft assignment, K-dim simplex)
        │
   ┌────┴────────────────────────┐
   L_PMI  (cross-view MI at cluster level)
   L_H    (marginal entropy guard)
   L_RR   (Barlow Twins redundancy reduction)
   L_Var  (VICReg variance guard)
   L_PD   (prototype-guided distillation)
```

### Overall Objective

$$\mathcal{L} = \mathcal{L}_\text{PMI} + \alpha\,\mathcal{L}_H + \beta\,\mathcal{L}_\text{RR} + \mathcal{L}_\text{Var} + \delta(t)\,\mathcal{L}_\text{PD}$$

### Training Phases

| Phase | Epochs | Active Losses | Notes |
|-------|--------|---------------|-------|
| 1 — Warm-up | 30 | `L_recon + β·L_RR + L_Var` | Builds view-specific representations via AE reconstruction |
| 2 — Joint | 200 | `L_PMI + α·L_H + β·L_RR + L_Var + w(t)·L_recon + δ(t)·L_PD` | `L_recon` anchor decays from `recon_weight_joint` to 10% of its initial value, preventing encoder drift |
| 3 — Fine-tune | 20 | `L_PMI + α·L_H` (encoders frozen) | Starts from the joint checkpoint with highest assignment confidence (unsupervised criterion) |

**Schedules:**
- `L_PD`: activated at joint epoch 50, linear ramp-up `δ: 0.1 → 1.0` over 50 epochs
- Temperature: anneals `τ: 1.0 → 0.3` during joint phase; fixed at `τ = 0.5` during fine-tune
- `L_recon` anchor: `w(t) = recon_weight_joint × (0.1 + 0.9 × (1 − t))` where `t ∈ [0,1]` over joint

**Checkpoint selection for fine-tune:** The fine-tune phase restores the joint-phase checkpoint that maximises mean max-assignment probability (an unsupervised confidence proxy), not the checkpoint with highest ACC. This avoids any label access during the training procedure.

### PMI Loss vs. IIC

Unlike IIC (Ji et al., 2019), ProtoMI applies a stop-gradient to the marginal distributions when computing MI:

```
with torch.no_grad():
    P_a = P.sum(dim=1)   # treated as constant
    P_b = P.sum(dim=0)
```

This **decouples** cross-view consistency (handled by `L_PMI`) from marginal uniformity (handled by `α·L_H`), allowing independent tuning of each objective. The trade-off is a biased (but lower-variance) gradient compared to the full IIC gradient.

---

## Supported Datasets

All datasets live in `dataset/`. Add new entries to `DATASET_REGISTRY` in `data.py` to support additional benchmarks.

| Name (key) | File | Samples | Views | Clusters |
|------------|------|---------|-------|----------|
| `handwritten` | `handwritten.mat` | 2,000 | 6 | 10 |
| `bdgp` | `BDGP_fea.mat` | 2,500 | 3 | 5 |
| `cub` | `CUB.mat` | 600 | 2 | 10 |
| `caltech7` | `Caltech101-7.mat` | 1,474 | 6 | 7 |

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, NumPy, SciPy, scikit-learn.

---

## Usage

### Training

```bash
# Default: handwritten, ρ = 0.5, uniform missing pattern, 5 runs
python train.py

# Different dataset
python train.py --dataset bdgp
python train.py --dataset caltech7 --missing_rate 0.3

# Block missing pattern
python train.py --missing_rate 0.5 --missing_pattern block

# Quick smoke-test (1 run, short training)
python train.py --warmup_epochs 5 --joint_epochs 20 --finetune_epochs 5 --n_runs 1
```

Output format (joint phase prints per-component breakdown):
```
[warmup] ep  30/250 loss=8240086.4 | ACC=0.7510 NMI=0.7229 ARI=0.6271 [km]
[joint ] ep  50/250 loss=795.69    | ACC=0.7615 NMI=0.7010 ARI=0.6304 [head]
           pmi=-25.77  lh=-13.55  rr=194.36  var=0.90  recon=7667677(x9.1e-05)
[fine  ] ep 250/250 loss=-94.98    | ACC=0.7765 NMI=0.7226 ARI=0.6543 [head]
>> Final: ACC=0.7765  NMI=0.7205  ARI=0.6535  PUR=0.7765
...
Summary (5 runs, missing_rate=0.5):
  ACC: 0.xxxx ± 0.xxxx
  NMI: 0.xxxx ± 0.xxxx
  ARI: 0.xxxx ± 0.xxxx
  PUR: 0.xxxx ± 0.xxxx
```

### Ablation Studies

```bash
# A1: without PMI (no cluster-level MI)
python ablation.py --study A1

# A2: without redundancy reduction and variance guard
python ablation.py --study A2

# A3: without entropy guard (expect cluster collapse)
python ablation.py --study A3

# A4: without prototype distillation
python ablation.py --study A4

# Missing rate sweep: ρ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
python ablation.py --study missing

# Run ablations on a different dataset
python ablation.py --study A3 --dataset bdgp
```

### Key Hyperparameters

All hyperparameters live in `config.py`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset` | `handwritten` | Dataset key from `DATASET_REGISTRY` |
| `missing_rate` | 0.5 | Fraction of views dropped per sample (ρ) |
| `missing_pattern` | `uniform` | `uniform` or `block` |
| `alpha` | 5.0 | Entropy guard weight (L_H) |
| `beta` | 1.0 | Redundancy reduction weight (L_RR) |
| `lambda_off` | 0.005 | Off-diagonal RR penalty |
| `recon_weight_joint` | 1e-4 | Initial reconstruction anchor weight in joint phase |
| `tau_start` / `tau_end` | 1.0 / 0.3 | Temperature annealing range during joint phase |
| `tau_finetune` | 0.5 | Fixed temperature during fine-tune phase |
| `tau_d` | 0.5 | Distillation temperature for PD loss |
| `d_z` | 128 | Latent dimension |
| `n_runs` | 5 | Repeated runs for mean ± std reporting |

---

## Project Structure

```
ProtoMI/
├── config.py       # All hyperparameters (dataclass)
├── data.py         # IMVCDataset, DATASET_REGISTRY, missing mask generation
├── models.py       # MLPEncoder/Decoder, ClusteringHead, ProtoMI
├── losses.py       # pmi_loss, entropy_loss, rr_loss, var_loss, pd_loss
├── metrics.py      # ACC (Hungarian), NMI, ARI, PUR
├── train.py        # Three-phase training loop + CLI
├── ablation.py     # Ablation studies A1–A5 + missing rate sweep
├── dataset/        # .mat files (handwritten, BDGP, CUB, Caltech101-7)
└── requirements.txt
```

---

## Why Not Contrastive Learning?

| Property | InfoNCE | ProtoMI |
|----------|---------|---------|
| Operates on | Instance representations | Cluster assignments |
| Requires negative pairs | Yes | No |
| Sensitive to batch size | Yes | No |
| False negative problem | Yes | No |
| Alignment granularity | Instance-level | Cluster-level |
| Paired sample efficiency | Low | High (estimates K×K matrix) |

Under extreme missing rates (ρ = 0.9), contrastive methods have very few paired samples to construct positive pairs. PMI only needs to estimate a K×K joint distribution (K = 10), requiring far fewer paired observations for equivalent statistical precision.

---

## Reference

Based on the research proposal:
> *ProtoMI: Prototype-Level Mutual Information Maximization with Redundancy Reduction for Incomplete Multi-View Clustering — A Non-Contrastive Framework for Cross-View Alignment Under Arbitrary Missing Patterns*
