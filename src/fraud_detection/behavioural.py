"""Behavioural features computed from the history of each customer and each terminal.

Every feature of a transaction made at time ``t`` uses information available at ``t``
only:

* customer aggregates cover the window ``(t - w, t]``: the earlier transactions of the
  customer and the current one, whose amount is known when the decision is taken;
* terminal aggregates cover the window ``(t - d - w, t - d]``, where ``d`` is the delay
  after which the fraud label of a transaction becomes known. Labels more recent than
  ``d`` are never read.

The aggregates are obtained from cumulative sums over the table sorted by entity and
time, the bounds of each window being located by binary search.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400
NIGHT_END_HOUR = 6

FEATURE_GROUPS = ("raw", "calendar", "customer", "terminal")


def _window_sums(
    entity: np.ndarray,
    time: np.ndarray,
    values: dict[str, np.ndarray],
    window_seconds: int,
    delay_seconds: int,
) -> dict[str, np.ndarray]:
    """Count and sums of ``values`` over ``(t - delay - window, t - delay]`` per entity.

    Results are returned in the original row order under the keys ``count`` and those of
    ``values``.
    """
    order = np.lexsort((time, entity))
    # A single sortable key per row: rows of one entity are contiguous and time-ordered.
    span = int(time.max()) + delay_seconds + window_seconds + 2
    key = entity[order].astype(np.int64) * span + time[order].astype(np.int64)
    base = entity.astype(np.int64) * span + time.astype(np.int64)

    # The window of an entity can never reach the rows of the previous entity, because
    # each bound is clipped to the first row of the entity.
    first_row = np.searchsorted(key, entity.astype(np.int64) * span, side="left")
    upper = np.searchsorted(key, base - delay_seconds, side="right")
    lower = np.searchsorted(key, base - delay_seconds - window_seconds, side="right")
    upper = np.maximum(upper, first_row)
    lower = np.maximum(lower, first_row)

    out = {"count": (upper - lower).astype(float)}
    for name, column in values.items():
        cumulative = np.concatenate([[0.0], np.cumsum(column[order], dtype=float)])
        out[name] = cumulative[upper] - cumulative[lower]
    return out


def add_behavioural_features(
    transactions: pd.DataFrame, windows_days: list[int], delay_days: int
) -> pd.DataFrame:
    """Return a copy of the transaction table with all feature groups added.

    Added columns
    -------------
    raw
        ``hour_sin``, ``hour_cos`` (the amount is already present).
    calendar
        ``is_weekend``, ``is_night``.
    customer, for each window ``w``
        ``customer_count_{w}d``, ``customer_mean_amount_{w}d`` and
        ``customer_amount_ratio_{w}d`` (current amount over the mean of the window).
    terminal, for each window ``w``
        ``terminal_count_{w}d`` and ``terminal_risk_{w}d`` (share of frauds), both
        delayed by ``delay_days``.
    """
    out = transactions.copy()
    time = out["time_seconds"].to_numpy()
    amount = out["amount"].to_numpy(dtype=float)

    phase = 2.0 * np.pi * (time % SECONDS_PER_DAY) / SECONDS_PER_DAY
    out["hour_sin"] = np.sin(phase)
    out["hour_cos"] = np.cos(phase)

    out["is_weekend"] = (out["datetime"].dt.dayofweek >= 5).astype(int)
    out["is_night"] = (out["datetime"].dt.hour <= NIGHT_END_HOUR).astype(int)

    customer = out["customer_id"].to_numpy()
    terminal = out["terminal_id"].to_numpy()
    is_fraud = out["is_fraud"].to_numpy(dtype=float)
    for days in windows_days:
        window = days * SECONDS_PER_DAY

        sums = _window_sums(customer, time, {"amount": amount}, window, delay_seconds=0)
        # The window contains at least the current transaction: the count is never zero.
        mean_amount = sums["amount"] / sums["count"]
        out[f"customer_count_{days}d"] = sums["count"]
        out[f"customer_mean_amount_{days}d"] = mean_amount
        out[f"customer_amount_ratio_{days}d"] = np.divide(
            amount, mean_amount, out=np.ones_like(amount), where=mean_amount > 0
        )

        sums = _window_sums(
            terminal, time, {"frauds": is_fraud}, window, delay_days * SECONDS_PER_DAY
        )
        out[f"terminal_count_{days}d"] = sums["count"]
        out[f"terminal_risk_{days}d"] = np.divide(
            sums["frauds"], sums["count"], out=np.zeros_like(amount), where=sums["count"] > 0
        )
    return out


def feature_columns(groups: tuple[str, ...] | list[str], windows_days: list[int]) -> list[str]:
    """Names of the columns belonging to the requested feature groups."""
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups {sorted(unknown)}; expected {FEATURE_GROUPS}")
    columns: list[str] = []
    if "raw" in groups:
        columns += ["amount", "hour_sin", "hour_cos"]
    if "calendar" in groups:
        columns += ["is_weekend", "is_night"]
    for days in windows_days:
        if "customer" in groups:
            columns += [
                f"customer_count_{days}d",
                f"customer_mean_amount_{days}d",
                f"customer_amount_ratio_{days}d",
            ]
        if "terminal" in groups:
            columns += [f"terminal_count_{days}d", f"terminal_risk_{days}d"]
    return columns
