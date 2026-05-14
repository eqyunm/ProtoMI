"""Model components: view-specific encoders/decoders, clustering head, ProtoMI."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────


class MLPEncoder(nn.Module):
    """MLP encoder: input_dim → hidden_dims → latent_dim."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], latent_dim: int):
        super().__init__()
        dims = [input_dim, *hidden_dims, latent_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.BatchNorm1d(dims[i + 1]), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPDecoder(nn.Module):
    """Symmetric MLP decoder: latent_dim → reversed(hidden_dims) → output_dim."""

    def __init__(self, latent_dim: int, hidden_dims: Tuple[int, ...], output_dim: int):
        super().__init__()
        dims = [latent_dim, *reversed(hidden_dims), output_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.BatchNorm1d(dims[i + 1]), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ClusteringHead(nn.Module):
    """Shared clustering head: d_z → hidden → K, with temperature-scaled softmax."""

    def __init__(self, d_z: int, n_clusters: int, hidden_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(d_z),
            nn.Linear(d_z, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_clusters),
        )
        self.tau: float = 1.0  # annealed externally

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.fc(z) / self.tau, dim=-1)

    def set_temperature(self, tau: float) -> None:
        self.tau = max(float(tau), 1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# ProtoMI main model
# ─────────────────────────────────────────────────────────────────────────────

class ProtoMI(nn.Module):
    """
    Prototype-level Mutual Information framework for IMVC.

    Components:
        encoders    — V view-specific MLP encoders
        decoders    — V view-specific MLP decoders (warm-up only)
        head        — shared clustering head
        mappers     — one linear layer per ordered view pair (v→u) for PD loss
    """

    def __init__(
        self,
        view_dims: List[int],
        d_z: int,
        n_clusters: int,
        n_views: int,
        encoder_hidden: Tuple[int, ...] = (1024, 512, 256),
    ):
        super().__init__()
        self.n_views = n_views
        self.n_clusters = n_clusters
        self.d_z = d_z

        self.encoders = nn.ModuleList(
            [MLPEncoder(d, encoder_hidden, d_z) for d in view_dims]
        )
        self.decoders = nn.ModuleList(
            [MLPDecoder(d_z, encoder_hidden, d) for d in view_dims]
        )
        self.head = ClusteringHead(d_z, n_clusters)

        # Proto mapper: φ_{v→u} — one linear layer per ordered pair, starts as identity
        self.mappers = nn.ModuleDict(
            {
                f"{v}_{u}": nn.Linear(d_z, d_z)
                for v in range(n_views)
                for u in range(n_views)
                if v != u
            }
        )
        for m in self.mappers.values():
            nn.init.eye_(m.weight)
            nn.init.zeros_(m.bias)

        # EMA prototypes — updated with momentum each batch, not a learnable param
        self.register_buffer("ema_protos", torch.zeros(n_views, n_clusters, d_z))

    # ── Forward helpers ───────────────────────────────────────────────────────

    def encode_all(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        """Encode every view; returns list of (B, d_z)."""
        return [self.encoders[v](views[v]) for v in range(self.n_views)]

    def assign_all(self, H: List[torch.Tensor]) -> List[torch.Tensor]:
        """Soft cluster assignments for every view; returns list of (B, K)."""
        return [self.head(H[v]) for v in range(self.n_views)]

    def reconstruct(self, H: List[torch.Tensor]) -> List[torch.Tensor]:
        """Decode every view; returns list of (B, d_v)."""
        return [self.decoders[v](H[v]) for v in range(self.n_views)]

    def map_proto(self, src: int, tgt: int, h: torch.Tensor) -> torch.Tensor:
        """Apply φ_{src→tgt} to latent vectors h: (B, d_z) → (B, d_z)."""
        return self.mappers[f"{src}_{tgt}"](h)

    @torch.no_grad()
    def update_ema_protos(
        self,
        H: List[torch.Tensor],
        Q: List[torch.Tensor],
        mask: torch.Tensor,
        momentum: float = 0.99,
    ) -> None:
        """EMA-update prototype buffer from the current batch (call after opt.step)."""
        for v in range(self.n_views):
            avail = mask[:, v] > 0
            if avail.sum() < 1:
                continue
            q_v = Q[v][avail].detach()                          # (N_v, K)
            h_v = H[v][avail].detach()                          # (N_v, d_z)
            w = q_v.T / (q_v.T.sum(dim=1, keepdim=True) + 1e-8)
            cur = w @ h_v                                        # (K, d_z)
            self.ema_protos[v].mul_(momentum).add_(cur, alpha=1.0 - momentum)
