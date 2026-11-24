"""Cost of decisions and cost-based decision rules.

Cost model
----------
Each alert costs a fixed handling cost, whether it turns out to be justified or not. A
fraud that raises no alert costs the amount of the transaction. A legitimate transaction
that raises no alert costs nothing.

==============  ==================  ==================
Decision        Fraud               Legitimate
==============  ==================  ==================
Alert           ``alert_cost``      ``alert_cost``
No alert        ``amount``          0
==============  ==================  ==================

Under this model, raising an alert costs ``alert_cost`` with certainty, and not raising
it costs ``p * amount`` in expectation, where ``p`` is the probability of fraud. The
decision that minimises the expected cost is therefore to alert when
``p * amount > alert_cost``; it requires calibrated probabilities and no fitted
threshold.
"""

from __future__ import annotations

import numpy as np


def total_cost(
    y_true: np.ndarray, amount: np.ndarray, alert: np.ndarray, alert_cost: float
) -> float:
    """Cost of a set of decisions under the cost model of this module."""
    y_true = np.asarray(y_true).astype(bool)
    alert = np.asarray(alert).astype(bool)
    missed = y_true & ~alert
    return float(alert_cost * alert.sum() + np.asarray(amount)[missed].sum())


def no_detection_cost(y_true: np.ndarray, amount: np.ndarray) -> float:
    """Cost when no alert is ever raised: the sum of the fraudulent amounts."""
    return float(np.asarray(amount)[np.asarray(y_true).astype(bool)].sum())


def oracle_cost(y_true: np.ndarray, amount: np.ndarray, alert_cost: float) -> float:
    """Lowest achievable cost: alert exactly on the frauds worth more than an alert."""
    fraud_amounts = np.asarray(amount)[np.asarray(y_true).astype(bool)]
    return float(np.minimum(fraud_amounts, alert_cost).sum())


def savings(cost: float, reference_cost: float) -> float:
    """Share of the no-detection cost that is avoided; negative if the rule costs more."""
    return 1.0 - cost / reference_cost if reference_cost > 0 else 0.0


def threshold_rule(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Alert when the probability of fraud reaches a fixed threshold."""
    return np.asarray(probability) >= threshold


def expected_cost_rule(
    probability: np.ndarray, amount: np.ndarray, alert_cost: float
) -> np.ndarray:
    """Alert when the expected loss of not alerting exceeds the cost of an alert."""
    return np.asarray(probability) * np.asarray(amount) > alert_cost


def best_threshold(
    y_true: np.ndarray, amount: np.ndarray, probability: np.ndarray, alert_cost: float
) -> float:
    """Fixed probability threshold that minimises the total cost on the given data.

    Transactions are ranked by decreasing probability; alerting on the first ``k`` costs
    ``k`` alerts plus the fraudulent amounts beyond rank ``k``. Every distinct
    probability is examined, and a threshold above all probabilities (no alert) is
    allowed.
    """
    y_true = np.asarray(y_true).astype(bool)
    amount = np.asarray(amount, dtype=float)
    probability = np.asarray(probability, dtype=float)

    order = np.argsort(-probability, kind="mergesort")
    sorted_probability = probability[order]
    fraud_amount = np.where(y_true[order], amount[order], 0.0)
    caught = np.cumsum(fraud_amount)
    k = np.arange(1, len(order) + 1)
    cost = alert_cost * k + (fraud_amount.sum() - caught)

    # A threshold can only cut between two distinct probabilities.
    is_cut = np.r_[sorted_probability[1:] != sorted_probability[:-1], True]
    candidates = np.flatnonzero(is_cut)
    best = candidates[np.argmin(cost[candidates])]
    if cost[best] >= fraud_amount.sum():
        return float(np.nextafter(sorted_probability[0], np.inf))
    return float(sorted_probability[best])


def decision_report(
    y_true: np.ndarray, amount: np.ndarray, alert: np.ndarray, alert_cost: float
) -> dict:
    """Cost, savings and operating characteristics of a set of decisions."""
    y_true = np.asarray(y_true).astype(bool)
    alert = np.asarray(alert).astype(bool)
    cost = total_cost(y_true, amount, alert, alert_cost)
    reference = no_detection_cost(y_true, amount)
    n_alerts = int(alert.sum())
    true_alerts = int((alert & y_true).sum())
    return {
        "cost": cost,
        "savings": savings(cost, reference),
        "n_alerts": n_alerts,
        "alerts_per_100k": 1e5 * n_alerts / len(y_true),
        "precision": true_alerts / n_alerts if n_alerts else 0.0,
        "recall": true_alerts / int(y_true.sum()) if y_true.any() else 0.0,
    }
