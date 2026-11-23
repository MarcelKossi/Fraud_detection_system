"""Model definitions."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
