from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    dataset: str = "handwritten"       # key in data.DATASET_REGISTRY
    missing_rate: float = 0.5          # ρ in {0.1, 0.3, 0.5, 0.7, 0.9}
    missing_pattern: str = "uniform"   # "uniform" | "block"
    # resolved automatically from registry + loaded data — do not set manually
    n_views: int = 6
    n_clusters: int = 10
    n_samples: int = 2000

    # ── Architecture ──────────────────────────────────────────────────────────
    d_z: int = 128                                     # latent dim
    encoder_hidden: Tuple[int, ...] = (1024, 512, 256) # MLP hidden dims

    # ── Training phases (epochs) ───────────────────────────────────────────────
    warmup_epochs: int = 30     # Phase 1: AE recon + RR + Var
    joint_epochs: int = 200     # Phase 2: full objective
    finetune_epochs: int = 20   # Phase 3: clustering head only

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256

    # ── Loss weights ──────────────────────────────────────────────────────────
    alpha: float = 5.0          # entropy guard weight (L_H)
    beta: float = 1.0           # redundancy reduction weight (L_RR)
    gamma_var: float = 1.0      # variance threshold for L_Var
    lambda_off: float = 0.005   # off-diagonal RR penalty
    delta_start: float = 0.1    # PD ramp start value
    delta_end: float = 1.0      # PD ramp end value
    pd_ramp_start: int = 50     # joint epoch to begin PD ramp
    pd_ramp_end: int = 100      # joint epoch where PD reaches delta_end
    tau_d: float = 0.5          # distillation temperature
    recon_weight_joint: float = 0.0   # 0 = auto-adaptive; >0 = manual override
    recon_anchor_target: float = 50.0 # desired recon contribution per batch at joint start

    # ── Temperature annealing (clustering head softmax) ───────────────────────
    tau_start: float = 1.0
    tau_end: float = 0.3        # raised from 0.1 — prevents over-commitment in late joint
    tau_finetune: float = 0.5   # fixed tau used during finetune phase

    # ── Joint distribution momentum buffer ────────────────────────────────────
    pmi_momentum: float = 0.9

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cuda"
    eval_interval: int = 10
    n_runs: int = 5             # repeat experiment for mean ± std
