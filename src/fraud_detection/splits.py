"""Data partitioning schemes.

All temporal functions operate on positional indices and assume that the rows are
already in chronological order (see :func:`fraud_detection.data.prepare`).
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

Indices = np.ndarray


def random_stratified_split(
    y: np.ndarray, test_fraction: float, seed: int
) -> tuple[Indices, Indices]:
    """Stratified random split, kept only to replicate the first version of the project."""
    positions = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        positions, test_size=test_fraction, random_state=seed, stratify=y
    )
    return train_idx, test_idx


def temporal_holdout(n_rows: int, test_fraction: float) -> tuple[Indices, Indices]:
    """Split chronologically ordered rows into a development set and a final hold-out set.

    The hold-out set is the last ``test_fraction`` of the rows: every hold-out
    transaction is at least as recent as every development transaction.
    """
    cut = int(round(n_rows * (1.0 - test_fraction)))
    if not 0 < cut < n_rows:
        raise ValueError("The split leaves an empty development or hold-out set")
    return np.arange(cut), np.arange(cut, n_rows)


def expanding_window_folds(n_rows: int, n_folds: int) -> list[tuple[Indices, Indices]]:
    """Expanding-window cross-validation on chronologically ordered rows.

    The rows are cut into ``n_folds + 1`` consecutive blocks of near-equal size. Fold
    ``i`` trains on blocks ``0..i`` and validates on block ``i + 1``, so that a model is
    never validated on transactions older than its training data.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    blocks = np.array_split(np.arange(n_rows), n_folds + 1)
    if any(len(block) == 0 for block in blocks):
        raise ValueError("Not enough rows for the requested number of folds")
    return [(np.concatenate(blocks[: i + 1]), blocks[i + 1]) for i in range(n_folds)]


def time_holdout(times: np.ndarray, test_fraction: float) -> tuple[Indices, Indices]:
    """Split on the time axis: the last ``test_fraction`` of the period is the hold-out set.

    Unlike :func:`temporal_holdout`, the cut is placed on elapsed time, not on the number
    of rows, which suits data whose transaction rate is steady.
    """
    start, end = float(times.min()), float(times.max())
    cut = start + (1.0 - test_fraction) * (end - start)
    development = np.flatnonzero(times < cut)
    holdout = np.flatnonzero(times >= cut)
    if len(development) == 0 or len(holdout) == 0:
        raise ValueError("The split leaves an empty development or hold-out set")
    return development, holdout


def expanding_time_folds(
    times: np.ndarray, n_folds: int, gap: float
) -> list[tuple[Indices, Indices]]:
    """Expanding-window folds on the time axis, with a gap before each validation block.

    The period is cut into ``n_folds + 1`` blocks of equal duration. Fold ``i`` validates
    on block ``i + 1`` and trains on every transaction older than the start of that block
    minus ``gap``. The gap represents the delay after which labels become available: a
    model deployed at a given date cannot have been trained on the labels of the days
    just before.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    edges = np.linspace(float(times.min()), float(times.max()) + 1.0, n_folds + 2)
    folds = []
    for i in range(n_folds):
        train = np.flatnonzero(times < edges[i + 1] - gap)
        validation = np.flatnonzero((times >= edges[i + 1]) & (times < edges[i + 2]))
        if len(train) == 0 or len(validation) == 0:
            raise ValueError("A fold is empty: reduce the gap or the number of folds")
        folds.append((train, validation))
    return folds


def assert_chronological(times: np.ndarray, earlier: Indices, later: Indices) -> None:
    """Raise if any transaction of ``later`` precedes a transaction of ``earlier``."""
    if times[earlier].max() > times[later].min():
        raise AssertionError("Temporal leakage: the later partition starts before the earlier ends")


def assert_disjoint(first: Indices, second: Indices) -> None:
    """Raise if the two partitions share a row."""
    if np.intersect1d(first, second).size:
        raise AssertionError("The partitions overlap")
