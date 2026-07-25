"""Classification metric helpers used uniformly by all benchmark families."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute standard binary metrics with clear errors for degenerate labels."""

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Metric inputs differ in shape: {y_true.shape} != {y_prob.shape}")
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics for an empty split")
    if len(np.unique(y_true)) != 2:
        raise ValueError("AUROC/AUPRC require both classes in the evaluated split")
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc": float(roc_auc_score(y_true, y_prob)),
        "prc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
