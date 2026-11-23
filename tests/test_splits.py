"""Leakage guards: no shared row and no inverted chronology between partitions."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import prepare
from fraud_detection.splits import (
    assert_chronological,
    assert_disjoint,
    expanding_window_folds,
    random_stratified_split,
    temporal_holdout,
)


def test_temporal_holdout_is_disjoint_exhaustive_and_chronological(transactions):
    prepared = prepare(transactions, "Time")
    dev_idx, test_idx = temporal_holdout(len(prepared), 0.2)

    assert_disjoint(dev_idx, test_idx)
    assert len(dev_idx) + len(test_idx) == len(prepared)
    assert len(test_idx) == pytest.approx(0.2 * len(prepared), abs=1)
    assert_chronological(prepared["Time"].to_numpy(), dev_idx, test_idx)


def test_no_identical_transaction_on_both_sides_of_the_holdout(transactions):
    prepared = prepare(transactions, "Time")
    dev_idx, test_idx = temporal_holdout(len(prepared), 0.2)
    merged = pd.merge(prepared.iloc[dev_idx], prepared.iloc[test_idx], how="inner")
    assert merged.empty


def test_expanding_window_folds_never_validate_on_the_past(transactions):
    prepared = prepare(transactions, "Time")
    times = prepared["Time"].to_numpy()
    folds = expanding_window_folds(len(prepared), 4)

    assert len(folds) == 4
    previous_train_size = 0
    for train_idx, val_idx in folds:
        assert_disjoint(train_idx, val_idx)
        assert_chronological(times, train_idx, val_idx)
        assert train_idx[0] == 0
        assert val_idx[0] == train_idx[-1] + 1
        assert len(train_idx) > previous_train_size
        previous_train_size = len(train_idx)


def test_validation_blocks_do_not_overlap(transactions):
    folds = expanding_window_folds(len(transactions), 4)
    validation = np.concatenate([val_idx for _, val_idx in folds])
    assert len(np.unique(validation)) == len(validation)


def test_chronology_guard_detects_a_shuffled_split(transactions):
    prepared = prepare(transactions, "Time")
    train_idx, test_idx = random_stratified_split(prepared["Class"].to_numpy(), 0.2, 42)
    with pytest.raises(AssertionError, match="Temporal leakage"):
        assert_chronological(prepared["Time"].to_numpy(), train_idx, test_idx)


def test_disjoint_guard_detects_overlap():
    with pytest.raises(AssertionError, match="overlap"):
        assert_disjoint(np.array([0, 1, 2]), np.array([2, 3]))


def test_random_split_is_stratified_and_reproducible(transactions):
    y = transactions["Class"].to_numpy()
    first = random_stratified_split(y, 0.2, 42)
    second = random_stratified_split(y, 0.2, 42)
    assert np.array_equal(first[1], second[1])
    assert y[first[1]].mean() == pytest.approx(y.mean(), abs=0.005)


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_degenerate_holdout_is_rejected(fraction):
    with pytest.raises(ValueError):
        temporal_holdout(100, fraction)
