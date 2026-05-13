"""Clustering evaluation metrics: ACC (Hungarian), NMI, ARI, PUR."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def cluster_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clustering accuracy with optimal Hungarian label matching."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    K = max(int(y_true.max()), int(y_pred.max())) + 1
    cost = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cost[p, t] += 1
    row, col = linear_sum_assignment(-cost)
    return float(cost[row, col].sum()) / len(y_true)


def purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n_labels = int(y_true.max()) + 1
    total = 0
    for k in np.unique(y_pred):
        mask = y_pred == k
        total += np.bincount(y_true[mask], minlength=n_labels).max()
    return total / len(y_true)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "ACC": cluster_acc(y_true, y_pred),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "PUR": purity(y_true, y_pred),
    }
