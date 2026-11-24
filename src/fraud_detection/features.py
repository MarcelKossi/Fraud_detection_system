"""Feature sets derived from the raw columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400

FEATURE_SETS = ("raw_time", "hour_of_day")


def add_hour_of_day(frame: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """Replace the elapsed-time column by a cyclic encoding of the time of day.

    ``Time`` counts the seconds since the first transaction of the file. Its raw value
    increases monotonically and is therefore meaningless for any transaction recorded
    after the collection period. Its value modulo 24 hours is a periodic quantity that
    remains defined at inference time. The clock time of the first transaction is not
    documented, so the encoding is a phase within the day, not a wall-clock hour.
    """
    phase = 2.0 * np.pi * (frame[time_column] % SECONDS_PER_DAY) / SECONDS_PER_DAY
    out = frame.drop(columns=[time_column]).copy()
    out["hour_sin"] = np.sin(phase)
    out["hour_cos"] = np.cos(phase)
    return out


def build_features(
    frame: pd.DataFrame, target: str, time_column: str, feature_set: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the design matrix and the target for a named feature set.

    ``raw_time`` keeps every original column, as in the first version of the project.
    ``hour_of_day`` substitutes the cyclic time-of-day encoding for the elapsed time.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}; expected one of {FEATURE_SETS}")
    y = frame[target]
    X = frame.drop(columns=[target])
    if feature_set == "hour_of_day":
        X = add_hour_of_day(X, time_column)
    return X, y
