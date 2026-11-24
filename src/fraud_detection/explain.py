"""Attribution of a prediction to the input features.

The contributions are Shapley values computed exactly for tree ensembles (TreeSHAP), as
implemented in XGBoost. They are expressed on the log-odds scale of the classifier and
are additive: for each transaction, the contributions and the base value sum to the
log-odds predicted by the model.

A contribution describes how the model uses a feature for one transaction. It is not a
causal statement about fraud.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier


def shap_contributions(model: XGBClassifier, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature contributions to the predicted log-odds.

    Returns
    -------
    contributions
        Array of shape ``(n_rows, n_features)``, columns in the order of ``X``.
    base_values
        Array of shape ``(n_rows,)``: the log-odds predicted in the absence of any
        feature information.
    """
    if not isinstance(model, XGBClassifier):
        raise TypeError("Exact contributions are implemented for XGBoost classifiers only")
    matrix = xgboost.DMatrix(X, feature_names=list(X.columns))
    values = model.get_booster().predict(matrix, pred_contribs=True)
    return values[:, :-1], values[:, -1]


def predicted_log_odds(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Log-odds predicted by the classifier, before the logistic transformation."""
    return model.predict(X, output_margin=True)


def global_importance(contributions: np.ndarray, feature_names: list[str]) -> pd.Series:
    """Mean absolute contribution of each feature, in decreasing order."""
    importance = pd.Series(np.abs(contributions).mean(axis=0), index=feature_names)
    return importance.sort_values(ascending=False)


def explain_transaction(model: XGBClassifier, row: pd.DataFrame, top_k: int | None = None) -> dict:
    """Explanation of a single prediction.

    Parameters
    ----------
    row
        A one-row frame with the feature columns expected by the model.
    top_k
        Number of features to report, ranked by absolute contribution; all by default.

    Returns
    -------
    dict
        ``base_log_odds``, ``log_odds`` and ``contributions``, a list of
        ``{"feature", "value", "contribution"}`` entries. When ``top_k`` truncates the
        list, ``other_contributions`` holds the sum of the features left out, so that
        the reported terms still add up to ``log_odds``.
    """
    if len(row) != 1:
        raise ValueError("explain_transaction expects exactly one transaction")
    contributions, base = shap_contributions(model, row)
    contributions = contributions[0]
    order = np.argsort(-np.abs(contributions))
    kept = order if top_k is None else order[:top_k]
    return {
        "base_log_odds": float(base[0]),
        "log_odds": float(base[0] + contributions.sum()),
        "contributions": [
            {
                "feature": row.columns[i],
                "value": float(row.iloc[0, i]),
                "contribution": float(contributions[i]),
            }
            for i in kept
        ],
        "other_contributions": float(contributions.sum() - contributions[kept].sum()),
    }
