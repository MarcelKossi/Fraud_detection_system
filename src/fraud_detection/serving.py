"""The deployable model: classifier, calibrator, decision rule and explanation.

The model is stored as a directory holding the classifier in the native JSON format of
XGBoost and a ``metadata.json`` file. No pickled object is involved, so loading a model
cannot execute arbitrary code and does not depend on the library versions used for
training beyond XGBoost's own format guarantees.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fraud_detection.calibration import ScoreCalibrator
from fraud_detection.explain import shap_contributions
from fraud_detection.features import add_hour_of_day
from fraud_detection.models import build_model

MODEL_FILE = "model.json"
METADATA_FILE = "metadata.json"

RAW_TIME_FIELD = "time_seconds"
AMOUNT_FIELD = "Amount"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class FraudModel:
    """Calibrated classifier with its decision rule.

    Attributes
    ----------
    classifier
        Fitted XGBoost classifier.
    feature_names
        Columns expected by the classifier, in order.
    calibration_slope, calibration_intercept
        Sigmoid calibration: the calibrated log-odds are
        ``slope * classifier log-odds + intercept``.
    alert_cost
        Handling cost of an alert, used by the expected-cost decision rule.
    metadata
        Provenance of the model (data checksum, configuration, library versions).
    """

    classifier: XGBClassifier
    feature_names: list[str]
    calibration_slope: float
    calibration_intercept: float
    alert_cost: float
    metadata: dict[str, Any]

    @property
    def input_fields(self) -> list[str]:
        """Fields expected for one transaction, before feature construction."""
        raw = [name for name in self.feature_names if not name.startswith("hour_")]
        return [RAW_TIME_FIELD, *raw]

    def build_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Derive the model features from raw transaction fields.

        ``transactions`` must hold the columns of :attr:`input_fields`. The elapsed time
        is replaced by the cyclic time-of-day encoding used in training.
        """
        missing = set(self.input_fields) - set(transactions.columns)
        if missing:
            raise ValueError(f"Missing fields: {sorted(missing)}")
        features = add_hour_of_day(transactions[self.input_fields], RAW_TIME_FIELD)
        return features[self.feature_names].astype(float)

    def calibrated_log_odds(self, features: pd.DataFrame) -> np.ndarray:
        raw = self.classifier.predict(features, output_margin=True)
        return self.calibration_slope * raw + self.calibration_intercept

    def predict_proba(self, transactions: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of fraud for each transaction."""
        return _sigmoid(self.calibrated_log_odds(self.build_features(transactions)))

    def decide(self, probability: np.ndarray, amount: np.ndarray) -> dict[str, np.ndarray]:
        """Expected-cost decision: alert when the expected loss exceeds the alert cost."""
        expected_loss = np.asarray(probability) * np.asarray(amount)
        return {"expected_loss": expected_loss, "alert": expected_loss > self.alert_cost}

    def explain(self, transaction: pd.DataFrame, top_k: int | None = None) -> dict[str, Any]:
        """Contributions of the features to the calibrated log-odds of one transaction.

        Sigmoid calibration is an affine map of the log-odds; the contributions of the
        classifier are therefore multiplied by the calibration slope, and the base value
        is transformed by the same map. The terms add up to the calibrated log-odds.
        """
        if len(transaction) != 1:
            raise ValueError("explain expects exactly one transaction")
        features = self.build_features(transaction)
        contributions, base = shap_contributions(self.classifier, features)
        contributions = self.calibration_slope * contributions[0]
        base_value = self.calibration_slope * float(base[0]) + self.calibration_intercept
        order = np.argsort(-np.abs(contributions))
        kept = order if top_k is None else order[:top_k]
        return {
            "scale": "log-odds",
            "base_value": base_value,
            "prediction": base_value + float(contributions.sum()),
            "contributions": [
                {
                    "feature": self.feature_names[i],
                    "value": float(features.iloc[0, i]),
                    "contribution": float(contributions[i]),
                }
                for i in kept
            ],
            "other_contributions": float(contributions.sum() - contributions[kept].sum()),
        }

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.classifier.save_model(directory / MODEL_FILE)
        document = {
            "feature_names": self.feature_names,
            "calibration": {
                "method": "sigmoid",
                "slope": self.calibration_slope,
                "intercept": self.calibration_intercept,
            },
            "decision": {"rule": "expected_cost", "alert_cost": self.alert_cost},
            **self.metadata,
        }
        with open(directory / METADATA_FILE, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> FraudModel:
        directory = Path(directory)
        with open(directory / METADATA_FILE, encoding="utf-8") as handle:
            document = json.load(handle)
        classifier = XGBClassifier()
        classifier.load_model(directory / MODEL_FILE)
        feature_names = document.pop("feature_names")
        calibration = document.pop("calibration")
        decision = document.pop("decision")
        return cls(
            classifier=classifier,
            feature_names=feature_names,
            calibration_slope=float(calibration["slope"]),
            calibration_intercept=float(calibration["intercept"]),
            alert_cost=float(decision["alert_cost"]),
            metadata=document,
        )


def fit_fraud_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    params: dict[str, Any],
    seed: int,
    calibration_fraction: float,
    alert_cost: float,
    metadata: dict[str, Any] | None = None,
) -> FraudModel:
    """Fit the classifier and its sigmoid calibrator on chronologically ordered data.

    The classifier is fitted on the first part of the rows and the calibrator on the
    last ``calibration_fraction``, which the classifier has not seen, as in the
    cross-validation that selected this configuration.
    """
    cut = int(round(len(X) * (1.0 - calibration_fraction)))
    classifier = build_model("xgboost", params, seed).fit(X.iloc[:cut], y.iloc[:cut])
    calibrator = ScoreCalibrator("sigmoid").fit(
        classifier.predict_proba(X.iloc[cut:])[:, 1], y.iloc[cut:].to_numpy()
    )
    slope, intercept = calibrator.coefficients()
    return FraudModel(
        classifier=classifier,
        feature_names=list(X.columns),
        calibration_slope=slope,
        calibration_intercept=intercept,
        alert_cost=alert_cost,
        metadata={
            "n_fit": int(cut),
            "n_fit_frauds": int(y.iloc[:cut].sum()),
            "n_calibration": int(len(X) - cut),
            "n_calibration_frauds": int(y.iloc[cut:].sum()),
            **(metadata or {}),
        },
    )
