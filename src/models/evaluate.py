# src/models/evaluate.py
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def ks_statistic(y_true, p_good) -> float:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_good)
    # sort by score
    order = np.argsort(p)
    y_sorted = y[order]
    p_sorted = p[order]

    # cumulative distributions of goods and bads
    goods = (y_sorted == 1).astype(int)
    bads = (y_sorted == 0).astype(int)

    cg = np.cumsum(goods) / max(goods.sum(), 1)
    cb = np.cumsum(bads) / max(bads.sum(), 1)

    return float(np.max(np.abs(cg - cb)))

def lift_at_k(y_true, p_good, k=0.10) -> float:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_good)
    n = len(y)
    top_n = max(int(np.ceil(k * n)), 1)
    idx = np.argsort(-p)[:top_n]
    base_rate = y.mean() if n else 0.0
    top_rate = y[idx].mean() if top_n else 0.0
    return float(top_rate / base_rate) if base_rate > 0 else 0.0

def compute_metrics(y_true, p_good) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_good)
    return {
        "auc_roc": float(roc_auc_score(y, p)),
        "auc_pr": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ks": ks_statistic(y, p),
        "lift_10pct": lift_at_k(y, p, k=0.10),
        "positive_rate": float(y.mean())
    }
