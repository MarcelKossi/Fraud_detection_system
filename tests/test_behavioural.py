"""Simulator, point-in-time correctness of behavioural features, time-based folds."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.behavioural import (
    SECONDS_PER_DAY,
    add_behavioural_features,
    feature_columns,
)
from fraud_detection.simulation import (
    AMOUNT_THRESHOLD,
    SimulationConfig,
    simulate_transactions,
    table_digest,
)
from fraud_detection.splits import assert_disjoint, expanding_time_folds, time_holdout

WINDOWS = [1, 7]
DELAY = 3
SMALL = SimulationConfig(
    n_customers=200, n_terminals=400, n_days=40, start_date="2018-04-01", radius=10.0, seed=0
)


@pytest.fixture(scope="module")
def transactions() -> pd.DataFrame:
    return simulate_transactions(SMALL)


@pytest.fixture(scope="module")
def featured(transactions) -> pd.DataFrame:
    return add_behavioural_features(transactions, WINDOWS, DELAY)


def test_simulation_is_reproducible_and_seed_dependent(transactions):
    assert table_digest(simulate_transactions(SMALL)) == table_digest(transactions)
    other = SimulationConfig(**{**SMALL.__dict__, "seed": 1})
    assert table_digest(simulate_transactions(other)) != table_digest(transactions)


def test_simulated_table_is_chronological_and_well_formed(transactions):
    assert transactions["time_seconds"].is_monotonic_increasing
    assert transactions["transaction_id"].is_unique
    assert transactions["amount"].ge(0).all()
    assert transactions["customer_id"].between(0, SMALL.n_customers - 1).all()
    assert transactions["terminal_id"].between(0, SMALL.n_terminals - 1).all()
    assert transactions["day"].max() == SMALL.n_days - 1
    seconds_in_day = transactions["time_seconds"] % SECONDS_PER_DAY
    assert seconds_in_day.gt(0).all()


def test_fraud_labels_follow_the_scenarios(transactions):
    assert set(transactions["fraud_scenario"].unique()) == {0, 1, 2, 3}
    assert (transactions["is_fraud"] == (transactions["fraud_scenario"] > 0)).all()
    # Scenario 1: every legitimate transaction is at or below the amount threshold.
    assert transactions.loc[transactions["is_fraud"] == 0, "amount"].le(AMOUNT_THRESHOLD).all()
    assert 0.0 < transactions["is_fraud"].mean() < 0.2


def test_customer_window_matches_a_direct_computation(featured):
    row = featured.iloc[len(featured) // 2]
    history = featured[
        (featured["customer_id"] == row["customer_id"])
        & (featured["time_seconds"] <= row["time_seconds"])
        & (featured["time_seconds"] > row["time_seconds"] - 7 * SECONDS_PER_DAY)
    ]
    assert row["customer_count_7d"] == len(history)
    assert row["customer_mean_amount_7d"] == pytest.approx(history["amount"].mean())


def test_terminal_risk_matches_a_direct_computation(featured):
    busiest = featured["terminal_id"].value_counts().idxmax()
    row = featured[featured["terminal_id"] == busiest].iloc[-1]
    upper = row["time_seconds"] - DELAY * SECONDS_PER_DAY
    history = featured[
        (featured["terminal_id"] == busiest)
        & (featured["time_seconds"] <= upper)
        & (featured["time_seconds"] > upper - 7 * SECONDS_PER_DAY)
    ]
    assert row["terminal_count_7d"] == len(history)
    expected = history["is_fraud"].mean() if len(history) else 0.0
    assert row["terminal_risk_7d"] == pytest.approx(expected)


def test_features_do_not_depend_on_the_future(transactions, featured):
    """Features of past transactions are unchanged when later transactions are removed."""
    columns = feature_columns(["raw", "calendar", "customer", "terminal"], WINDOWS)
    cut = transactions["time_seconds"].quantile(0.6)
    past = transactions[transactions["time_seconds"] <= cut]
    recomputed = add_behavioural_features(past, WINDOWS, DELAY)
    pd.testing.assert_frame_equal(recomputed[columns], featured.loc[past.index, columns])


def test_terminal_features_never_read_labels_younger_than_the_delay(transactions, featured):
    """Changing every label of the last ``DELAY`` days leaves earlier-known features intact."""
    risk_columns = [f"terminal_risk_{days}d" for days in WINDOWS]
    last = transactions["time_seconds"].max()
    recent = transactions["time_seconds"] > last - DELAY * SECONDS_PER_DAY
    altered = transactions.copy()
    altered.loc[recent, "is_fraud"] = 1 - altered.loc[recent, "is_fraud"]
    recomputed = add_behavioural_features(altered, WINDOWS, DELAY)
    pd.testing.assert_frame_equal(recomputed[risk_columns], featured[risk_columns])


def test_current_label_is_not_a_feature(transactions, featured):
    columns = feature_columns(["raw", "calendar", "customer", "terminal"], WINDOWS)
    flipped = transactions.assign(is_fraud=1 - transactions["is_fraud"])
    recomputed = add_behavioural_features(flipped, WINDOWS, DELAY)
    customer_columns = [c for c in columns if not c.startswith("terminal_risk")]
    pd.testing.assert_frame_equal(recomputed[customer_columns], featured[customer_columns])
    assert "is_fraud" not in columns and "fraud_scenario" not in columns
    assert "customer_id" not in columns and "terminal_id" not in columns


def test_feature_columns_by_group():
    assert feature_columns(["raw"], WINDOWS) == ["amount", "hour_sin", "hour_cos"]
    assert len(feature_columns(["raw", "calendar", "customer", "terminal"], WINDOWS)) == 15
    with pytest.raises(ValueError, match="Unknown feature groups"):
        feature_columns(["unknown"], WINDOWS)


def test_time_holdout_and_folds_respect_chronology_and_gap(transactions):
    times = transactions["time_seconds"].to_numpy()
    dev_idx, holdout_idx = time_holdout(times, 0.2)
    assert_disjoint(dev_idx, holdout_idx)
    assert times[dev_idx].max() < times[holdout_idx].min()

    gap = DELAY * SECONDS_PER_DAY
    dev_times = times[dev_idx]
    folds = expanding_time_folds(dev_times, 3, gap)
    assert len(folds) == 3
    previous = 0
    for train_idx, val_idx in folds:
        assert_disjoint(train_idx, val_idx)
        assert dev_times[train_idx].max() + gap <= dev_times[val_idx].min()
        assert len(train_idx) > previous
        previous = len(train_idx)
    validation = np.concatenate([val_idx for _, val_idx in folds])
    assert len(np.unique(validation)) == len(validation)


def test_folds_reject_a_gap_longer_than_the_first_block(transactions):
    times = transactions["time_seconds"].to_numpy()
    with pytest.raises(ValueError, match="fold is empty"):
        expanding_time_folds(times, 4, gap=30 * SECONDS_PER_DAY)
