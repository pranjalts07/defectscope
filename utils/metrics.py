import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    labels: list[int],
    preds: list[int],
    probs: list[float],
) -> dict:
    """
    Compute the full set of metrics we care about for defect detection.

    In a manufacturing context, recall matters more than precision:
    a missed defect (false negative) reaching a customer is worse than
    rejecting a good product (false positive).
    """
    return {
        "auc_roc":   round(roc_auc_score(labels, probs), 4),
        "f1":        round(f1_score(labels, preds), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall":    round(recall_score(labels, preds, zero_division=0), 4),
    }
