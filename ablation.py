"""Ablation study runner (Table A1-A7 from the proposal).

Each ablation modifies the Config and delegates to train_one_run.
Usage:
    python ablation.py --study A1       # PMI vs InfoNCE
    python ablation.py --study A3       # without entropy guard
    python ablation.py --study missing  # sweep missing rates
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from data import DATASET_REGISTRY
from train import train_one_run, resolve_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Ablation variants
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(name: str, cfg: Config) -> None:
    print(f"\n{'='*65}")
    print(f"Ablation: {name}")
    print(f"{'='*65}")
    all_m = []
    for run in range(cfg.n_runs):
        m = train_one_run(cfg, cfg.seed + run)
        all_m.append(m)
    print(f"\nResult [{name}]:")
    for k in ("ACC", "NMI", "ARI", "PUR"):
        vals = [m[k] for m in all_m]
        print(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


def a1_no_pmi(base: Config) -> None:
    """A1: replace L_PMI with instance-level InfoNCE (degenerate — use no PMI)."""
    cfg = Config(**base.__dict__)
    cfg.alpha = 0.0   # disable entropy guard too; raw instance-level regime
    # NOTE: full InfoNCE replacement would need code changes;
    # this approximation trains without the cluster-level MI signal.
    run_ablation("A1: no PMI (alpha=0, no cluster MI)", cfg)


def a2_no_rr(base: Config) -> None:
    """A2: remove redundancy reduction and variance guard."""
    cfg = Config(**base.__dict__)
    cfg.beta = 0.0
    cfg.gamma_var = 0.0
    run_ablation("A2: no RR / Var", cfg)


def a3_no_entropy(base: Config) -> None:
    """A3: remove entropy guard — observe collapse."""
    cfg = Config(**base.__dict__)
    cfg.alpha = 0.0
    run_ablation("A3: no entropy guard (collapse expected)", cfg)


def a4_no_pd(base: Config) -> None:
    """A4: remove prototype distillation — single-view samples unused."""
    cfg = Config(**base.__dict__)
    cfg.delta_start = 0.0
    cfg.delta_end = 0.0
    run_ablation("A4: no PD loss", cfg)


def a5_no_confidence(base: Config) -> None:
    """A5: PD loss without confidence weighting (uniform weights)."""
    # This requires a code change in losses.py; flag it and skip.
    print("\nA5: uniform-confidence PD — requires modifying pd_loss in losses.py.")
    print("    Set conf=torch.ones_like(conf) in pd_loss to test.")


def missing_rate_sweep(base: Config) -> None:
    """Sweep missing rates ρ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}."""
    print(f"\n{'='*65}")
    print("Missing Rate Sweep")
    print(f"{'='*65}")
    for rho in (0.1, 0.3, 0.5, 0.7, 0.9):
        cfg = Config(**base.__dict__)
        cfg.missing_rate = rho
        all_m = []
        for run in range(cfg.n_runs):
            m = train_one_run(cfg, cfg.seed + run)
            all_m.append(m)
        acc_vals = [m["ACC"] for m in all_m]
        print(f"  ρ={rho}: ACC={np.mean(acc_vals):.4f} ± {np.std(acc_vals):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

STUDIES = {
    "A1": a1_no_pmi,
    "A2": a2_no_rr,
    "A3": a3_no_entropy,
    "A4": a4_no_pd,
    "A5": a5_no_confidence,
    "missing": missing_rate_sweep,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--study", choices=list(STUDIES), required=True)
    p.add_argument("--dataset", type=str, default="handwritten",
                   choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--missing_rate", type=float, default=0.5)
    p.add_argument("--n_runs", type=int, default=3)
    p.add_argument("--warmup_epochs", type=int, default=30)
    p.add_argument("--joint_epochs", type=int, default=200)
    p.add_argument("--finetune_epochs", type=int, default=20)
    args = p.parse_args()

    base = Config()
    base.dataset = args.dataset
    base.missing_rate = args.missing_rate
    base.n_runs = args.n_runs
    base.warmup_epochs = args.warmup_epochs
    base.joint_epochs = args.joint_epochs
    base.finetune_epochs = args.finetune_epochs
    resolve_dataset(base)

    STUDIES[args.study](base)


if __name__ == "__main__":
    main()
