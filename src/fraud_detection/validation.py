"""Cross-validated evaluation of a model configuration on chronologically ordered data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.calibration import METHODS, ScoreCalibrator, calibration_metrics
from fraud_detection.metrics import auprc, recall_at_fpr
from fraud_detection.models import build_model, positive_class_weight
from fraud_detection.sampling import resample_training_fold
from fraud_detection.splits import Indices

Folds = list[tuple[Indices, Indices]]


def fit_on_fold(
    family: str,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str,
    ratio: float,
    seed: int,
):
    """Fit one model on one training partition, with the given imbalance treatment."""
    X_fit, y_fit = resample_training_fold(X_train, y_train, strategy, ratio, seed)
    pos_weight = positive_class_weight(y_fit.to_numpy()) if strategy == "class_weight" else None
    model = build_model(family, params, seed, pos_weight)
    model.fit(X_fit, y_fit)
    return model


def cross_validate(
    family: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    folds: Folds,
    *,
    seed: int,
    max_fpr: float,
    strategy: str = "none",
    ratio: float = 0.1,
) -> dict[str, Any]:
    """Ranking metrics of one configuration on each validation fold."""
    fold_auprc, fold_recall = [], []
    for train_idx, val_idx in folds:
        model = fit_on_fold(
            family, params, X.iloc[train_idx], y.iloc[train_idx], strategy, ratio, seed
        )
        scores = model.predict_proba(X.iloc[val_idx])[:, 1]
        y_val = y.iloc[val_idx].to_numpy()
        fold_auprc.append(auprc(y_val, scores))
        fold_recall.append(recall_at_fpr(y_val, scores, max_fpr))
    return {
        "fold_auprc": fold_auprc,
        "fold_recall_at_fpr": fold_recall,
        "auprc_mean": float(np.mean(fold_auprc)),
        "auprc_std": float(np.std(fold_auprc, ddof=1)),
        "recall_at_fpr_mean": float(np.mean(fold_recall)),
    }


def cross_validate_calibration(
    family: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    folds: Folds,
    *,
    seed: int,
    strategy: str,
    ratio: float,
    calibration_fraction: float,
) -> dict[str, dict[str, list[float]]]:
    """Compare calibration methods without reusing training data.

    Each training window is cut chronologically: the classifier is fitted on its first
    part, the calibrators on its last ``calibration_fraction``, and all are evaluated on
    the validation fold that follows.
    """
    per_method: dict[str, dict[str, list[float]]] = {
        method: {"brier": [], "log_loss": [], "auprc": []} for method in METHODS
    }
    for train_idx, val_idx in folds:
        cut = int(round(len(train_idx) * (1.0 - calibration_fraction)))
        fit_idx, cal_idx = train_idx[:cut], train_idx[cut:]
        model = fit_on_fold(family, params, X.iloc[fit_idx], y.iloc[fit_idx], strategy, ratio, seed)
        cal_scores = model.predict_proba(X.iloc[cal_idx])[:, 1]
        val_scores = model.predict_proba(X.iloc[val_idx])[:, 1]
        y_cal, y_val = y.iloc[cal_idx].to_numpy(), y.iloc[val_idx].to_numpy()
        for method in METHODS:
            calibrated = ScoreCalibrator(method).fit(cal_scores, y_cal).transform(val_scores)
            for name, value in calibration_metrics(y_val, calibrated).items():
                per_method[method][name].append(value)
            per_method[method]["auprc"].append(auprc(y_val, calibrated))
    return per_method
