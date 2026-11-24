"""Evaluation metrics for highly imbalanced binary classification."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

Metric = Callable[[np.ndarray, np.ndarray], float]


def auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve, estimated by average precision."""
    return float(average_precision_score(y_true, scores))


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Highest recall reachable while keeping the false-positive rate at or below ``max_fpr``."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(tpr[fpr <= max_fpr].max())


def threshold_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Lowest score threshold whose false-positive rate does not exceed ``max_fpr``."""
    fpr, _, thresholds = roc_curve(y_true, scores)
    return float(thresholds[fpr <= max_fpr].min())


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Precision, recall, F1 and confusion matrix at a fixed decision threshold."""
    predicted = (np.asarray(scores) >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, average="binary", zero_division=0
    )
    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: Metric,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of ``metric``.

    Transactions are resampled with replacement. Resamples without any positive
    example are discarded, since ranking metrics are undefined on them.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_resamples):
        sample = rng.integers(0, n, n)
        if y_true[sample].sum() == 0:
            continue
        values.append(metric(y_true[sample], scores[sample]))
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return float(low), float(high)


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    max_fpr: float,
    n_resamples: int = 0,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict:
    """Full metric report for one set of predicted scores.

    Confidence intervals are added when ``n_resamples`` is positive.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    def recall_metric(y: np.ndarray, s: np.ndarray) -> float:
        return recall_at_fpr(y, s, max_fpr)

    report = {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "auprc": auprc(y_true, scores),
        "recall_at_fpr": {"max_fpr": max_fpr, "value": recall_metric(y_true, scores)},
        "at_threshold": threshold_metrics(y_true, scores, threshold),
    }
    if n_resamples > 0:
        args = (n_resamples, confidence_level, seed)
        report["auprc_ci"] = list(bootstrap_ci(y_true, scores, auprc, *args))
        report["recall_at_fpr"]["ci"] = list(bootstrap_ci(y_true, scores, recall_metric, *args))
        report["confidence_level"] = confidence_level
    return report
