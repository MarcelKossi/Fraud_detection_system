"""Hyper-parameter search spaces and reproducible random sampling."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import ParameterSampler

SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "C": loguniform(1e-6, 1e2),
    },
    "lightgbm": {
        "n_estimators": randint(100, 501),
        "learning_rate": loguniform(0.005, 0.2),
        "num_leaves": randint(4, 64),
        "min_child_samples": randint(10, 201),
        "min_child_weight": loguniform(1e-3, 10.0),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_lambda": loguniform(1e-2, 100.0),
    },
    "xgboost": {
        "n_estimators": randint(100, 501),
        "learning_rate": loguniform(0.005, 0.2),
        "max_depth": randint(2, 9),
        "min_child_weight": loguniform(0.5, 20.0),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_lambda": loguniform(1e-2, 100.0),
    },
}


def sample_configurations(family: str, n_trials: int, seed: int) -> list[dict[str, Any]]:
    """Draw ``n_trials`` configurations from the search space of a model family.

    The draw depends only on ``family``, ``n_trials`` and ``seed``. Values are converted
    to built-in types so that they can be written to JSON.
    """
    if family not in SEARCH_SPACES:
        raise ValueError(f"No search space defined for {family!r}")
    sampler = ParameterSampler(SEARCH_SPACES[family], n_iter=n_trials, random_state=seed)
    return [
        {
            name: int(value) if isinstance(value, (int, np.integer)) else float(value)
            for name, value in configuration.items()
        }
        for configuration in sampler
    ]
