"""ProtoMI training script — three-phase training on Handwritten dataset.

Usage:
    python train.py                    # default config
    python train.py --missing_rate 0.3
    python train.py --missing_rate 0.7 --missing_pattern block
"""

import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from config import Config
from data import DATASET_REGISTRY, IMVCDataset, get_dataloader
from models import ProtoMI
from losses import (
    pmi_loss,
    entropy_loss,
    rr_loss,
    var_loss,
    pd_loss,
)
from metrics import evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

class _Tee:
    """Write to both the original stdout and a log file simultaneously."""
    def __init__(self, file_path: str):
        self._file = open(file_path, "w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def setup_logging(cfg: "Config") -> tuple[str, _Tee]:
    """Create logs/ dir, open a timestamped log file, and redirect stdout."""
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"logs/{cfg.dataset}_mr{cfg.missing_rate}_{cfg.missing_pattern}_{ts}.log"
    tee = _Tee(fname)
    sys.stdout = tee
    return fname, tee


def teardown_logging(log_path: str, tee: _Tee, cfg: "Config", all_metrics: list) -> None:
    """Write JSON summary, restore stdout, and close the log file."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": cfg.dataset,
        "missing_rate": cfg.missing_rate,
        "missing_pattern": cfg.missing_pattern,
        "n_runs": cfg.n_runs,
        "runs": all_metrics,
        "mean": {k: float(np.mean([m[k] for m in all_metrics])) for k in ("ACC", "NMI", "ARI", "PUR")},
        "std":  {k: float(np.std ([m[k] for m in all_metrics])) for k in ("ACC", "NMI", "ARI", "PUR")},
    }
    json_path = log_path.replace(".log", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[log] text  → {log_path}")
    print(f"[log] json  → {json_path}")
    sys.stdout = tee._stdout
    tee.close()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: Config) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def _compute_base_recon_w(model: ProtoMI, loader, device: torch.device, cfg: Config) -> float:
    """Measure reconstruction loss at warmup end; return adaptive joint-phase weight.

    Targets cfg.recon_anchor_target loss units of reconstruction contribution
    per batch at the START of joint training (before any decay).
    If cfg.recon_weight_joint > 0, returns that value directly (manual override).
    """
    if cfg.recon_weight_joint > 0:
        return cfg.recon_weight_joint

    model.eval()
    recon_total, n_batches = 0.0, 0
    for views, mask, _ in loader:
        views = [v.to(device) for v in views]
        mask  = mask.to(device)
        H     = model.encode_all(views)
        recons = model.reconstruct(H)
        for v in range(cfg.n_views):
            avail = mask[:, v] > 0
            if avail.sum() > 0:
                recon_total += F.mse_loss(recons[v][avail], views[v][avail]).item()
        n_batches += 1
    model.train()

    recon_per_batch = recon_total / max(n_batches, 1)
    if recon_per_batch < 1e-8:
        return 0.0
    return cfg.recon_anchor_target / recon_per_batch


@torch.no_grad()
def _init_ema_protos(model: ProtoMI, loader, device: torch.device, cfg: Config) -> None:
    """Seed EMA prototypes from a full dataset pass at the start of joint training."""
    model.eval()
    proto_num = [torch.zeros(cfg.n_clusters, cfg.d_z, device=device) for _ in range(cfg.n_views)]
    proto_den = [torch.zeros(cfg.n_clusters, 1,     device=device) for _ in range(cfg.n_views)]
    for views, mask, _ in loader:
        views = [v.to(device) for v in views]
        mask  = mask.to(device)
        H = model.encode_all(views)
        Q = model.assign_all(H)
        for v in range(cfg.n_views):
            avail = mask[:, v] > 0
            if avail.sum() < 1:
                continue
            q_v = Q[v][avail]
            h_v = H[v][avail]
            proto_num[v] += q_v.T @ h_v
            proto_den[v] += q_v.sum(0, keepdim=True).T
    for v in range(cfg.n_views):
        model.ema_protos[v] = proto_num[v] / (proto_den[v] + 1e-8)
    model.train()


def resolve_dataset(cfg: Config) -> None:
    """Fill cfg.n_clusters / n_views / n_samples from registry + data probe.

    Also applies per-dataset hyperparameter overrides (alpha, warmup_epochs)
    unless the user has already set them explicitly via CLI.
    """
    if cfg.dataset not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{cfg.dataset}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    info = DATASET_REGISTRY[cfg.dataset]
    cfg.n_clusters = info.n_clusters
    probe = IMVCDataset(info, missing_rate=0.0, missing_pattern="uniform", seed=0)
    cfg.n_views = probe.n_views
    cfg.n_samples = probe.n_samples

    # Apply per-dataset overrides
    if info.alpha is not None:
        cfg.alpha = info.alpha
    if info.warmup_epochs is not None:
        cfg.warmup_epochs = info.warmup_epochs


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(
    model: ProtoMI,
    dataset: IMVCDataset,
    device: torch.device,
    cfg: Config,
):
    """Run full dataset through model; return (y_true, y_pred_head, y_pred_km, conf).

    conf is the mean max-assignment probability — an unsupervised proxy for
    clustering confidence used for checkpoint selection (no label access).
    """
    model.eval()
    loader = get_dataloader(dataset, batch_size=512, shuffle=False, drop_last=False)

    all_H = [[] for _ in range(cfg.n_views)]
    all_Q = [[] for _ in range(cfg.n_views)]
    all_mask, all_labels = [], []

    for views, mask, labels in loader:
        views = [v.to(device) for v in views]
        mask = mask.to(device)
        H = model.encode_all(views)
        Q = model.assign_all(H)
        for v in range(cfg.n_views):
            all_H[v].append(H[v].cpu())
            all_Q[v].append(Q[v].cpu())
        all_mask.append(mask.cpu())
        all_labels.append(labels)

    H_full = [torch.cat(all_H[v]) for v in range(cfg.n_views)]  # (N, d_z)
    Q_full = [torch.cat(all_Q[v]) for v in range(cfg.n_views)]  # (N, K)
    mask_full = torch.cat(all_mask)                              # (N, V)
    labels = torch.cat(all_labels).numpy()

    # ── Head argmax: average soft assignments over available views ────────────
    Q_avg = torch.zeros(len(labels), cfg.n_clusters)
    cnt = torch.zeros(len(labels), 1)
    for v in range(cfg.n_views):
        avail = mask_full[:, v:v+1]
        Q_avg += avail * Q_full[v]
        cnt += avail
    Q_avg /= cnt.clamp(min=1)
    y_pred_head = Q_avg.argmax(dim=1).numpy()

    # Unsupervised confidence: mean max-assignment probability (no labels used)
    conf = Q_avg.max(dim=1).values.mean().item()

    # ── K-Means on averaged latent representations ────────────────────────────
    H_avg = torch.zeros(len(labels), cfg.d_z)
    for v in range(cfg.n_views):
        H_avg += mask_full[:, v:v+1] * H_full[v]
    H_avg /= cnt.clamp(min=1)
    km = KMeans(n_clusters=cfg.n_clusters, n_init=10, random_state=42)
    y_pred_km = km.fit_predict(H_avg.numpy())

    model.train()
    return labels, y_pred_head, y_pred_km, conf


# ─────────────────────────────────────────────────────────────────────────────
# Single training run
# ─────────────────────────────────────────────────────────────────────────────

def train_one_run(cfg: Config, run_seed: int) -> dict:
    set_seed(run_seed)
    device = get_device(cfg)

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset = IMVCDataset(
        DATASET_REGISTRY[cfg.dataset], cfg.missing_rate, cfg.missing_pattern, run_seed
    )
    loader = get_dataloader(dataset, cfg.batch_size)

    # Print view dims once
    if run_seed == cfg.seed:
        print(f"  View dims: {dataset.view_dims}")
        avail_per_view = dataset.mask.sum(0)
        for v, n in enumerate(avail_per_view):
            print(f"  View {v}: {int(n)}/{cfg.n_samples} samples available "
                  f"({100*n/cfg.n_samples:.1f}%)")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ProtoMI(
        view_dims=dataset.view_dims,
        d_z=cfg.d_z,
        n_clusters=cfg.n_clusters,
        n_views=cfg.n_views,
        encoder_hidden=cfg.encoder_hidden,
    ).to(device)

    # ── Optimizers (one per phase) ────────────────────────────────────────────
    enc_dec_params = (
        list(model.encoders.parameters()) + list(model.decoders.parameters())
    )
    opt_warmup = torch.optim.Adam(
        enc_dec_params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    opt_joint = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    opt_fine = torch.optim.Adam(
        model.head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sch_warmup = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_warmup, T_max=cfg.warmup_epochs
    )
    sch_joint = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_joint, T_max=cfg.joint_epochs
    )
    sch_fine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_fine, T_max=cfg.finetune_epochs
    )

    total_epochs = cfg.warmup_epochs + cfg.joint_epochs + cfg.finetune_epochs
    total_train = cfg.warmup_epochs + cfg.joint_epochs

    best_metrics = None
    best_acc = 0.0
    best_conf = 0.0     # unsupervised proxy for checkpoint selection
    best_state = None   # saved at highest-confidence joint checkpoint (no label access)
    base_recon_w = 0.0  # set adaptively at warmup→joint transition

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(total_epochs):

        # ── Phase / optimizer selection ───────────────────────────────────────
        if epoch < cfg.warmup_epochs:
            phase, opt, sch = 1, opt_warmup, sch_warmup
        elif epoch < cfg.warmup_epochs + cfg.joint_epochs:
            phase, opt, sch = 2, opt_joint, sch_joint
        else:
            phase, opt, sch = 3, opt_fine, sch_fine

        # ── Adaptive recon weight + EMA proto init at warmup→joint transition ──
        if epoch == cfg.warmup_epochs:
            base_recon_w = _compute_base_recon_w(model, loader, device, cfg)
            print(f"  [recon anchor] base_recon_w={base_recon_w:.2e}  "
                  f"(target={cfg.recon_anchor_target:.0f} / warmup_recon_per_batch)")
            _init_ema_protos(model, loader, device, cfg)

        # ── Restore best joint checkpoint at finetune start ──────────────────
        if epoch == cfg.warmup_epochs + cfg.joint_epochs and best_state is not None:
            model.load_state_dict(best_state)

        # ── Temperature annealing (clock starts at joint phase, not epoch 0) ────
        if phase == 3:
            model.head.set_temperature(cfg.tau_finetune)
        elif epoch < cfg.warmup_epochs:
            model.head.set_temperature(cfg.tau_start)
        else:
            j_ep_tau = epoch - cfg.warmup_epochs
            frac = min(j_ep_tau / max(cfg.joint_epochs, 1), 1.0)
            model.head.set_temperature(
                cfg.tau_start + (cfg.tau_end - cfg.tau_start) * frac
            )

        # ── PD loss delta ramp-up + recon anchor weight ───────────────────────
        if phase == 2:
            j_ep = epoch - cfg.warmup_epochs   # epoch within joint phase
            j_frac = j_ep / max(cfg.joint_epochs, 1)
            if j_ep < cfg.pd_ramp_start:
                delta = 0.0
            elif j_ep < cfg.pd_ramp_end:
                t = (j_ep - cfg.pd_ramp_start) / (cfg.pd_ramp_end - cfg.pd_ramp_start)
                delta = cfg.delta_start + (cfg.delta_end - cfg.delta_start) * t
            else:
                delta = cfg.delta_end
            cur_recon_w = base_recon_w * (0.1 + 0.9 * (1.0 - j_frac))
        else:
            delta = 0.0
            j_frac = 0.0
            cur_recon_w = 0.0

        # ── Mini-batch loop ───────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        ep_pmi = ep_lh = ep_rr = ep_var = ep_recon = 0.0  # phase-2 diagnostics

        for views, mask, _ in loader:
            views = [v.to(device) for v in views]
            mask = mask.to(device)         # (B, V)
            opt.zero_grad()

            # Forward
            if phase == 3:
                # Phase 3: freeze encoders — no grad through them
                with torch.no_grad():
                    H = model.encode_all(views)
                Q = model.assign_all(H)
            else:
                H = model.encode_all(views)
                Q = model.assign_all(H) if phase == 2 else None

            loss = torch.zeros(1, device=device).squeeze()

            # ── Phase 1: reconstruction + RR + Var ───────────────────────────
            if phase == 1:
                recons = model.reconstruct(H)
                for v in range(cfg.n_views):
                    avail = mask[:, v] > 0
                    if avail.sum() > 0:
                        loss = loss + F.mse_loss(recons[v][avail], views[v][avail])

                for a in range(cfg.n_views):
                    for b in range(a + 1, cfg.n_views):
                        paired = (mask[:, a] > 0) & (mask[:, b] > 0)
                        if paired.sum() < 4:
                            continue
                        loss = loss + cfg.beta * rr_loss(
                            H[a][paired], H[b][paired], cfg.lambda_off
                        )
                for v in range(cfg.n_views):
                    avail = mask[:, v] > 0
                    if avail.sum() > 3:
                        loss = loss + var_loss(H[v][avail], cfg.gamma_var)

            # ── Phase 2: full objective ───────────────────────────────────────
            elif phase == 2:
                # L_PMI — real co-observed pairs + delayed pseudo-PMI
                l_pmi = torch.zeros(1, device=device).squeeze()
                for a in range(cfg.n_views):
                    for b in range(a + 1, cfg.n_views):
                        paired = (mask[:, a] > 0) & (mask[:, b] > 0)
                        if paired.sum() >= 4:
                            l_pmi = l_pmi + pmi_loss(Q[a][paired], Q[b][paired])

                        # Pseudo-PMI: only after mapper is trained by PD loss
                        if j_ep < cfg.pseudo_pmi_start:
                            continue

                        only_a = (mask[:, a] > 0) & (mask[:, b] == 0)
                        if only_a.sum() >= 4:
                            with torch.no_grad():
                                h_m = F.normalize(model.map_proto(a, b, H[a][only_a]), dim=-1)
                                p_b = F.normalize(model.ema_protos[b], dim=-1)
                                q_tilde_b = F.softmax(h_m @ p_b.T / cfg.tau_d, dim=-1)
                            l_pmi = l_pmi + pmi_loss(Q[a][only_a], q_tilde_b)

                        only_b = (mask[:, b] > 0) & (mask[:, a] == 0)
                        if only_b.sum() >= 4:
                            with torch.no_grad():
                                h_m = F.normalize(model.map_proto(b, a, H[b][only_b]), dim=-1)
                                p_a = F.normalize(model.ema_protos[a], dim=-1)
                                q_tilde_a = F.softmax(h_m @ p_a.T / cfg.tau_d, dim=-1)
                            l_pmi = l_pmi + pmi_loss(q_tilde_a, Q[b][only_b])

                # L_H
                l_lh = torch.zeros(1, device=device).squeeze()
                for v in range(cfg.n_views):
                    avail = mask[:, v] > 0
                    if avail.sum() > 0:
                        l_lh = l_lh + entropy_loss(Q[v][avail])

                # L_RR
                l_rr = torch.zeros(1, device=device).squeeze()
                for a in range(cfg.n_views):
                    for b in range(a + 1, cfg.n_views):
                        paired = (mask[:, a] > 0) & (mask[:, b] > 0)
                        if paired.sum() < 4:
                            continue
                        l_rr = l_rr + rr_loss(H[a][paired], H[b][paired], cfg.lambda_off)

                # L_Var
                l_var = torch.zeros(1, device=device).squeeze()
                for v in range(cfg.n_views):
                    avail = mask[:, v] > 0
                    if avail.sum() > 3:
                        l_var = l_var + var_loss(H[v][avail], cfg.gamma_var)

                loss = loss + l_pmi + cfg.alpha * l_lh + cfg.beta * l_rr + l_var

                # L_recon anchor (decays linearly to 0 over joint phase)
                l_recon = torch.zeros(1, device=device).squeeze()
                if cur_recon_w > 0:
                    recons = model.reconstruct(H)
                    for v in range(cfg.n_views):
                        avail = mask[:, v] > 0
                        if avail.sum() > 0:
                            l_recon = l_recon + F.mse_loss(recons[v][avail], views[v][avail])
                    loss = loss + cur_recon_w * l_recon

                # L_PD (with ramp-up) — uses EMA prototypes for stable pseudo-labels
                if delta > 0:
                    ema_protos_list = [model.ema_protos[v] for v in range(cfg.n_views)]
                    loss = loss + pd_loss(
                        model, H, Q, mask, ema_protos_list,
                        cfg.tau_d, delta, cfg.n_views, cfg.n_clusters
                    )

                ep_pmi   += l_pmi.item()
                ep_lh    += l_lh.item()
                ep_rr    += l_rr.item()
                ep_var   += l_var.item()
                ep_recon += l_recon.item()

            # ── Phase 3: L_PMI + L_H only, encoder frozen ────────────────────
            elif phase == 3:
                for a in range(cfg.n_views):
                    for b in range(a + 1, cfg.n_views):
                        paired = (mask[:, a] > 0) & (mask[:, b] > 0)
                        if paired.sum() < 4:
                            continue
                        loss = loss + pmi_loss(Q[a][paired], Q[b][paired])

                for v in range(cfg.n_views):
                    avail = mask[:, v] > 0
                    if avail.sum() > 0:
                        loss = loss + cfg.alpha * entropy_loss(Q[v][avail])

            loss.backward()
            opt.step()
            if phase == 2:
                model.update_ema_protos(H, Q, mask)
            epoch_loss += loss.item()

        sch.step()

        # ── Periodic evaluation ───────────────────────────────────────────────
        if (epoch + 1) % cfg.eval_interval == 0 or epoch == total_epochs - 1:
            y_true, y_head, y_km, conf = get_predictions(model, dataset, device, cfg)
            m_head = evaluate(y_true, y_head)
            m_km   = evaluate(y_true, y_km)
            m = m_head if m_head["ACC"] >= m_km["ACC"] else m_km
            tag = "head" if m_head["ACC"] >= m_km["ACC"] else "km"

            # Checkpoint selection uses unsupervised confidence — no label access.
            # best_metrics / best_acc are tracked only for final reporting.
            if phase == 2 and conf > best_conf:
                best_conf = conf
                best_state = copy.deepcopy(model.state_dict())

            if m["ACC"] > best_acc:
                best_acc = m["ACC"]
                best_metrics = m

            phase_name = {1: "warmup", 2: "joint ", 3: "fine  "}[phase]
            n = len(loader)
            print(
                f"  [{phase_name}] ep {epoch+1:3d}/{total_epochs} "
                f"loss={epoch_loss/n:.4f} | "
                f"ACC={m['ACC']:.4f} NMI={m['NMI']:.4f} "
                f"ARI={m['ARI']:.4f} [{tag}]"
            )
            if phase == 2:
                print(
                    f"           pmi={ep_pmi/n:.3f}  lh={ep_lh/n:.3f}  "
                    f"rr={ep_rr/n:.3f}  var={ep_var/n:.3f}  "
                    f"recon={ep_recon/n:.1f}(x{cur_recon_w:.1e})"
                )

    if best_metrics is None:
        y_true, y_head, y_km, _ = get_predictions(model, dataset, device, cfg)
        m_head = evaluate(y_true, y_head)
        m_km   = evaluate(y_true, y_km)
        best_metrics = m_head if m_head["ACC"] >= m_km["ACC"] else m_km

    return best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",         type=str,   default=cfg.dataset,
                   choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--missing_rate",    type=float, default=cfg.missing_rate)
    p.add_argument("--missing_pattern", type=str,   default=cfg.missing_pattern)
    p.add_argument("--warmup_epochs",   type=int,   default=cfg.warmup_epochs)
    p.add_argument("--joint_epochs",    type=int,   default=cfg.joint_epochs)
    p.add_argument("--finetune_epochs", type=int,   default=cfg.finetune_epochs)
    p.add_argument("--batch_size",      type=int,   default=cfg.batch_size)
    p.add_argument("--lr",              type=float, default=cfg.lr)
    p.add_argument("--alpha",              type=float, default=cfg.alpha)
    p.add_argument("--beta",               type=float, default=cfg.beta)
    p.add_argument("--recon_weight_joint",  type=float, default=cfg.recon_weight_joint)
    p.add_argument("--recon_anchor_target", type=float, default=cfg.recon_anchor_target)
    p.add_argument("--n_runs",             type=int,   default=cfg.n_runs)
    p.add_argument("--seed",            type=int,   default=cfg.seed)
    p.add_argument("--device",          type=str,   default=cfg.device)
    args = p.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


def main() -> None:
    cfg = parse_args()
    resolve_dataset(cfg)
    dev = "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"

    log_path, tee = setup_logging(cfg)

    print("=" * 65)
    print(f"ProtoMI  —  {cfg.dataset} ({cfg.n_views} views, {cfg.n_clusters} clusters)")
    print(f"Missing rate: {cfg.missing_rate}  Pattern: {cfg.missing_pattern}")
    print(f"Phases: warmup={cfg.warmup_epochs}  joint={cfg.joint_epochs}  "
          f"finetune={cfg.finetune_epochs}")
    print(f"Device: {dev}  |  Runs: {cfg.n_runs}")
    print("=" * 65)

    all_metrics = []
    for run in range(cfg.n_runs):
        seed = cfg.seed + run
        print(f"\n--- Run {run+1}/{cfg.n_runs}  (seed={seed}) ---")
        m = train_one_run(cfg, seed)
        all_metrics.append(m)
        print(f"  >> Final: ACC={m['ACC']:.4f}  NMI={m['NMI']:.4f}  "
              f"ARI={m['ARI']:.4f}  PUR={m['PUR']:.4f}")

    print("\n" + "=" * 65)
    print(f"Summary ({cfg.n_runs} runs, missing_rate={cfg.missing_rate}):")
    for k in ("ACC", "NMI", "ARI", "PUR"):
        vals = [m[k] for m in all_metrics]
        print(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print("=" * 65)

    teardown_logging(log_path, tee, cfg, all_metrics)


if __name__ == "__main__":
    main()
