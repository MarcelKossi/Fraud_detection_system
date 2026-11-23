"""Shared fixtures: a small synthetic transaction table with the schema of the real file."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def transactions() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 2_000
    frame = pd.DataFrame(
        {
            "Time": np.sort(rng.integers(0, 172_800, n)).astype(float),
            "V1": rng.normal(size=n),
            "V2": rng.normal(size=n),
            "Amount": rng.gamma(2.0, 40.0, n).round(2),
            "Class": (rng.random(n) < 0.05).astype(int),
        }
    )
    # Inject exact duplicates, as observed in the real file.
    copies = frame.iloc[[10, 500, 1500]]
    return pd.concat([frame, copies], ignore_index=True)
