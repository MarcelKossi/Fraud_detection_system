"""Treatments of class imbalance.

Re-sampling is applied to training partitions only, never to validation or test data,
whose class proportions must remain those of the real transaction flow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

STRATEGIES = ("none", "class_weight", "undersampling", "smote")


def undersample(
    X: pd.DataFrame, y: pd.Series, ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Keep every fraud and a random subset of legitimate transactions.

    ``ratio`` is the target number of frauds per legitimate transaction. Row order is
    preserved.
    """
    y_values = y.to_numpy()
    positives = np.flatnonzero(y_values == 1)
    negatives = np.flatnonzero(y_values == 0)
    n_keep = min(len(negatives), int(round(len(positives) / ratio)))
    rng = np.random.default_rng(seed)
    kept = np.sort(np.concatenate([positives, rng.choice(negatives, n_keep, replace=False)]))
    return X.iloc[kept], y.iloc[kept]


def oversample_smote(
    X: pd.DataFrame, y: pd.Series, ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Add synthetic frauds by SMOTE until the fraud-to-legitimate ratio reaches ``ratio``."""
    return SMOTE(sampling_strategy=ratio, random_state=seed).fit_resample(X, y)


def resample_training_fold(
    X: pd.DataFrame, y: pd.Series, strategy: str, ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply a re-sampling strategy to one training partition.

    ``none`` and ``class_weight`` leave the data unchanged; the latter acts through the
    loss function (see :func:`fraud_detection.models.build_model`).
    """
    if strategy in ("none", "class_weight"):
        return X, y
    if strategy == "undersampling":
        return undersample(X, y, ratio, seed)
    if strategy == "smote":
        return oversample_smote(X, y, ratio, seed)
    raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}")
