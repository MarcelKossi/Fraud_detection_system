"""Model definitions."""

from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FAMILIES = ("logistic_regression", "random_forest", "lightgbm", "xgboost")


def build_baseline(n_estimators: int, seed: int) -> Pipeline:
    """Model of the first version of the project: standardisation, then a random forest.

    Scaler and classifier are held in a single pipeline, so that the scaler is always
    fitted on the training partition only and both are serialised together.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1),
            ),
        ]
    )


def positive_class_weight(y: np.ndarray) -> float:
    """Ratio of negative to positive examples, the usual weight of the minority class."""
    y = np.asarray(y)
    n_positive = int(y.sum())
    if n_positive == 0:
        raise ValueError("No positive example to weight")
    return float((len(y) - n_positive) / n_positive)


def build_model(
    family: str,
    params: dict[str, Any],
    seed: int,
    pos_weight: float | None = None,
) -> BaseEstimator:
    """Instantiate a classifier of the requested family.

    Parameters
    ----------
    family
        One of :data:`FAMILIES`.
    params
        Hyper-parameters passed to the underlying estimator.
    seed
        Random seed of the estimator.
    pos_weight
        Weight applied to the positive class in the loss; ``None`` leaves the classes
        unweighted.
    """
    if family == "logistic_regression":
        class_weight = None if pos_weight is None else {0: 1.0, 1: pos_weight}
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        **params, class_weight=class_weight, max_iter=2000, random_state=seed
                    ),
                ),
            ]
        )
    if family == "random_forest":
        class_weight = None if pos_weight is None else {0: 1.0, 1: pos_weight}
        return RandomForestClassifier(
            **params, class_weight=class_weight, random_state=seed, n_jobs=-1
        )
    if family == "lightgbm":
        return LGBMClassifier(
            **params,
            scale_pos_weight=1.0 if pos_weight is None else pos_weight,
            subsample_freq=1,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if family == "xgboost":
        return XGBClassifier(
            **params,
            scale_pos_weight=1.0 if pos_weight is None else pos_weight,
            tree_method="hist",
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model family {family!r}; expected one of {FAMILIES}")
