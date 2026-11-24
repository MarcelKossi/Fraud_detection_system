"""Probability calibration of classifier scores."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

METHODS = ("none", "sigmoid", "isotonic")

_EPS = 1e-12


def _logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, _EPS, 1.0 - _EPS)
    return np.log(clipped / (1.0 - clipped))


class ScoreCalibrator:
    """Map raw scores to calibrated probabilities.

    ``sigmoid`` fits a logistic regression on the log-odds of the scores (Platt scaling);
    ``isotonic`` fits a monotone step function; ``none`` returns the scores unchanged.
    The calibrator must be fitted on data that the classifier has not been trained on.
    """

    def __init__(self, method: str):
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
        self.method = method
        self._model = None

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> ScoreCalibrator:
        scores = np.asarray(scores, dtype=float)
        if self.method == "sigmoid":
            self._model = LogisticRegression(C=1e6, max_iter=1000)
            self._model.fit(_logit(scores).reshape(-1, 1), y_true)
        elif self.method == "isotonic":
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._model.fit(scores, y_true)
        return self

    def coefficients(self) -> tuple[float, float]:
        """Slope and intercept of the sigmoid calibration on the log-odds scale."""
        if self.method != "sigmoid" or self._model is None:
            raise ValueError("Coefficients are defined for a fitted sigmoid calibrator only")
        return float(self._model.coef_[0, 0]), float(self._model.intercept_[0])

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self.method == "sigmoid":
            return self._model.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self._model.predict(scores)
        return scores


def calibration_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Brier score and logarithmic loss; both are proper scoring rules (lower is better)."""
    probabilities = np.clip(np.asarray(probabilities, dtype=float), _EPS, 1.0 - _EPS)
    return {
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
    }
