"""Dataset loading and missing-mask generation for IMVC benchmarks."""

import numpy as np
import scipy.io
import torch
from dataclasses import dataclass
from typing import Optional
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetInfo:
    file: str           # path relative to project root
    n_clusters: int
    x_key: str = "X"            # mat key for feature cell-array
    y_key: Optional[str] = None # label key; None = auto-detect from common names
    normalize: bool = False     # per-feature z-score normalization (fit on full dataset)
    alpha: Optional[float] = None  # override Config.alpha (entropy guard weight)
    warmup_epochs: Optional[int] = None  # override Config.warmup_epochs


DATASET_REGISTRY: dict = {
    "handwritten": DatasetInfo("dataset/handwritten.mat", 10, normalize=True),
    "bdgp":        DatasetInfo("dataset/BDGP_fea.mat",    5,  normalize=True),
    "cub":         DatasetInfo("dataset/CUB.mat",          10, normalize=True),
    "caltech7":    DatasetInfo("dataset/Caltech101-7.mat", 7, normalize=True),
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class IMVCDataset(Dataset):
    """Generic incomplete multi-view clustering dataset.

    Loads any registered .mat file and simulates missing views via a binary
    mask.  Missing-view inputs are zeroed so no information leaks through the
    encoder for masked views.
    """

    def __init__(
        self,
        info: DatasetInfo,
        missing_rate: float = 0.5,
        missing_pattern: str = "uniform",
        seed: int = 42,
    ):
        self.views, self.labels, self.view_dims = _load(info)
        if info.normalize:
            self.views = _zscore_views(self.views)
        self.n_samples = len(self.labels)
        self.n_views = len(self.views)

        rng = np.random.RandomState(seed)
        self.mask = _generate_mask(
            self.n_samples, self.n_views, missing_rate, missing_pattern, rng
        )  # (N, V)  float32 {0, 1}

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        mask = torch.from_numpy(self.mask[idx])          # (V,)
        views = [
            torch.from_numpy(self.views[v][idx]) * mask[v]  # zero missing views
            for v in range(self.n_views)
        ]
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return views, mask, label


def get_dataloader(
    dataset: IMVCDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
        drop_last=drop_last,
        num_workers=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────

def _load(info: DatasetInfo):
    """Load a .mat file described by DatasetInfo.

    Returns:
        views     — list of (N, d_v) float32 arrays, one per view
        labels    — (N,) int64 array, 0-indexed
        view_dims — list of ints [d_0, d_1, ...]
    """
    raw = scipy.io.loadmat(info.file)

    # ── Views ─────────────────────────────────────────────────────────────────
    if info.x_key in raw and raw[info.x_key].dtype == object:
        flat = raw[info.x_key].flatten()
        views = [flat[v].astype(np.float32) for v in range(len(flat))]
    else:
        avail = [k for k in raw if not k.startswith("_")]
        raise ValueError(
            f"Cannot find view key '{info.x_key}' in {info.file}. "
            f"Available keys: {avail}"
        )

    # ── Labels ────────────────────────────────────────────────────────────────
    label_keys = [info.y_key] if info.y_key else ["Y", "y", "gt", "label", "labels", "gnd"]
    for key in label_keys:
        if key in raw:
            labels = raw[key].flatten().astype(np.int64)
            break
    else:
        raise ValueError(f"No label key found in {info.file}")

    labels -= labels.min()   # ensure 0-indexed
    view_dims = [v.shape[1] for v in views]
    return views, labels, view_dims


def _zscore_views(views: list) -> list:
    """Per-feature z-score normalization fitted on the full dataset."""
    normalized = []
    for v in views:
        mean = v.mean(axis=0, keepdims=True)
        std  = v.std(axis=0, keepdims=True)
        std  = np.where(std < 1e-8, 1.0, std)   # keep constant features unchanged
        normalized.append(((v - mean) / std).astype(np.float32))
    return normalized


def _generate_mask(N, V, missing_rate, pattern, rng):
    """Generate (N, V) binary mask; every sample retains ≥1 view."""
    mask = np.ones((N, V), dtype=np.float32)

    if pattern == "uniform":
        drop = rng.rand(N, V) < missing_rate
        mask[drop] = 0.0
    elif pattern == "block":
        n_drop = max(0, min(V - 1, int(round(V * missing_rate))))
        for i in range(N):
            if n_drop > 0:
                idxs = rng.choice(V, n_drop, replace=False)
                mask[i, idxs] = 0.0
    else:
        raise ValueError(f"Unknown missing pattern: {pattern!r}")

    # Guarantee at least one view per sample
    all_missing = mask.sum(1) == 0
    if all_missing.any():
        fallback = rng.randint(V, size=all_missing.sum())
        mask[all_missing, fallback] = 1.0

    return mask


def _collate(batch):
    """Stack a list of (views_list, mask, label) into batch tensors."""
    n_views = len(batch[0][0])
    views = [torch.stack([item[0][v] for item in batch]) for v in range(n_views)]
    mask  = torch.stack([item[1] for item in batch])   # (B, V)
    labels = torch.stack([item[2] for item in batch])  # (B,)
    return views, mask, labels
