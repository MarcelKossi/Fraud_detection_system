"""Simulator of card transactions with customer and terminal identifiers.

Independent, vectorised implementation of the generative process specified in the
*Fraud Detection Handbook* (Le Borgne et al., chapter 3, "Transaction data simulator").
The specification is followed; the code and the random streams are not those of the
handbook, so the generated tables are statistically equivalent, not identical, to its
baseline dataset.

Generative process
------------------
* Customers and terminals are placed uniformly on a 100 x 100 grid. A customer can use
  the terminals located at a distance below ``radius``.
* Each customer has a mean amount drawn from U(5, 100), a standard deviation equal to
  half of that mean, and a mean number of daily transactions drawn from U(0, 4).
* Each day, the number of transactions of a customer is Poisson distributed. The time of
  day is N(43200 s, 20000 s), discarded outside the day. The amount is normal; a negative
  draw is replaced by U(0, 2 x mean amount).

Fraud scenarios
---------------
1. Any transaction above 220 is a fraud.
2. Each day, two terminals are compromised: all their transactions over the following
   28 days are frauds.
3. Each day, three customers are compromised: over the following 14 days, one third of
   their transactions have their amount multiplied by 5 and are frauds.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SECONDS_PER_DAY = 86_400
GRID_SIZE = 100.0

AMOUNT_THRESHOLD = 220.0
COMPROMISED_TERMINALS_PER_DAY = 2
TERMINAL_COMPROMISE_DAYS = 28
COMPROMISED_CUSTOMERS_PER_DAY = 3
CUSTOMER_COMPROMISE_DAYS = 14
CUSTOMER_FRAUD_FRACTION = 1.0 / 3.0
CUSTOMER_FRAUD_MULTIPLIER = 5.0


@dataclass(frozen=True)
class SimulationConfig:
    n_customers: int
    n_terminals: int
    n_days: int
    start_date: str
    radius: float
    seed: int


def _legitimate_transactions(config: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    customer_xy = rng.uniform(0.0, GRID_SIZE, size=(config.n_customers, 2))
    mean_amount = rng.uniform(5.0, 100.0, size=config.n_customers)
    daily_rate = rng.uniform(0.0, 4.0, size=config.n_customers)
    terminal_xy = rng.uniform(0.0, GRID_SIZE, size=(config.n_terminals, 2))

    # Terminals within reach of each customer, stored as one flat array with offsets.
    reachable = cKDTree(terminal_xy).query_ball_point(customer_xy, r=config.radius)
    n_reachable = np.array([len(terminals) for terminals in reachable])
    offsets = np.concatenate([[0], np.cumsum(n_reachable)])
    flat_terminals = np.concatenate([np.sort(terminals) for terminals in reachable]).astype(int)

    # Customers without any terminal in reach make no transaction.
    rates = np.where(n_reachable > 0, daily_rate, 0.0)
    counts = rng.poisson(rates[:, None], size=(config.n_customers, config.n_days))
    customer = np.repeat(np.arange(config.n_customers), counts.sum(axis=1))
    day = np.concatenate([np.repeat(np.arange(config.n_days), row) for row in counts])
    n = len(customer)

    second = rng.normal(SECONDS_PER_DAY / 2, 20_000.0, size=n).astype(int)
    amount = rng.normal(mean_amount[customer], mean_amount[customer] / 2.0)
    negative = amount < 0
    amount[negative] = rng.uniform(0.0, 2.0 * mean_amount[customer[negative]])
    pick = np.floor(rng.uniform(size=n) * n_reachable[customer]).astype(int)
    terminal = flat_terminals[offsets[customer] + pick]

    keep = (second > 0) & (second < SECONDS_PER_DAY)
    return pd.DataFrame(
        {
            "time_seconds": day[keep] * SECONDS_PER_DAY + second[keep],
            "day": day[keep],
            "customer_id": customer[keep],
            "terminal_id": terminal[keep],
            "amount": np.round(amount[keep], 2),
        }
    )


def _add_frauds(
    frame: pd.DataFrame, config: SimulationConfig, rng: np.random.Generator
) -> pd.DataFrame:
    day = frame["day"].to_numpy()
    scenario = np.zeros(len(frame), dtype=int)

    scenario[frame["amount"].to_numpy() > AMOUNT_THRESHOLD] = 1

    terminal = frame["terminal_id"].to_numpy()
    for start in range(config.n_days):
        compromised = rng.choice(config.n_terminals, COMPROMISED_TERMINALS_PER_DAY, replace=False)
        window = (day >= start) & (day < start + TERMINAL_COMPROMISE_DAYS)
        scenario[window & np.isin(terminal, compromised)] = 2

    customer = frame["customer_id"].to_numpy()
    amount = frame["amount"].to_numpy().copy()
    for start in range(config.n_days):
        compromised = rng.choice(config.n_customers, COMPROMISED_CUSTOMERS_PER_DAY, replace=False)
        window = (day >= start) & (day < start + CUSTOMER_COMPROMISE_DAYS)
        candidates = np.flatnonzero(window & np.isin(customer, compromised))
        n_frauds = int(len(candidates) * CUSTOMER_FRAUD_FRACTION)
        chosen = rng.choice(candidates, n_frauds, replace=False)
        amount[chosen] = np.round(amount[chosen] * CUSTOMER_FRAUD_MULTIPLIER, 2)
        scenario[chosen] = 3

    out = frame.copy()
    out["amount"] = amount
    out["is_fraud"] = (scenario > 0).astype(int)
    out["fraud_scenario"] = scenario
    return out


def simulate_transactions(config: SimulationConfig) -> pd.DataFrame:
    """Generate a labelled transaction table, in chronological order.

    Columns: ``transaction_id``, ``datetime``, ``time_seconds`` (since the start date),
    ``day``, ``customer_id``, ``terminal_id``, ``amount``, ``is_fraud`` and
    ``fraud_scenario`` (0 for legitimate transactions; when several scenarios apply to a
    transaction, the last one applied is recorded).
    """
    rng = np.random.default_rng(config.seed)
    frame = _legitimate_transactions(config, rng)
    frame = frame.sort_values("time_seconds", kind="mergesort").reset_index(drop=True)
    frame = _add_frauds(frame, config, rng)
    frame.insert(0, "transaction_id", np.arange(len(frame)))
    frame.insert(
        1,
        "datetime",
        pd.Timestamp(config.start_date) + pd.to_timedelta(frame["time_seconds"], unit="s"),
    )
    return frame


def table_digest(frame: pd.DataFrame) -> str:
    """SHA-256 of the numerical content of a simulated table.

    Recorded with each result, so that two runs can be checked to rest on the same data.
    """
    columns = ["time_seconds", "customer_id", "terminal_id", "amount", "is_fraud"]
    digest = hashlib.sha256()
    for column in columns:
        digest.update(np.ascontiguousarray(frame[column].to_numpy(dtype=np.float64)).tobytes())
    return digest.hexdigest()
