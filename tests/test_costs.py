import numpy as np
import pytest

from fraud_detection.costs import (
    best_threshold,
    decision_report,
    expected_cost_rule,
    no_detection_cost,
    oracle_cost,
    savings,
    threshold_rule,
    total_cost,
)

Y = np.array([1, 1, 0, 0, 1])
AMOUNT = np.array([100.0, 4.0, 50.0, 20.0, 30.0])
PROBABILITY = np.array([0.9, 0.6, 0.7, 0.1, 0.2])


def test_total_cost_on_a_hand_computed_case():
    alert = np.array([True, False, True, False, False])
    # Two alerts at 10 each, two missed frauds worth 4 and 30.
    assert total_cost(Y, AMOUNT, alert, alert_cost=10.0) == pytest.approx(54.0)


def test_reference_costs():
    assert no_detection_cost(Y, AMOUNT) == pytest.approx(134.0)
    # The fraud worth 4 is cheaper to miss than to investigate.
    assert oracle_cost(Y, AMOUNT, alert_cost=10.0) == pytest.approx(24.0)
    assert savings(24.0, 134.0) == pytest.approx(1 - 24 / 134)


def test_expected_cost_rule_depends_on_the_amount():
    alert = expected_cost_rule(PROBABILITY, AMOUNT, alert_cost=10.0)
    # 0.9 x 100, 0.7 x 50 exceed 10; 0.6 x 4, 0.1 x 20 and 0.2 x 30 do not.
    assert alert.tolist() == [True, False, True, False, False]


def test_threshold_rule_is_inclusive():
    assert threshold_rule(PROBABILITY, 0.7).tolist() == [True, False, True, False, False]


def test_best_threshold_minimises_the_cost_over_all_cuts():
    threshold = best_threshold(Y, AMOUNT, PROBABILITY, alert_cost=10.0)
    best = total_cost(Y, AMOUNT, threshold_rule(PROBABILITY, threshold), 10.0)
    for candidate in [*PROBABILITY, 1.1]:
        other = total_cost(Y, AMOUNT, threshold_rule(PROBABILITY, candidate), 10.0)
        assert best <= other


def test_best_threshold_raises_no_alert_when_alerts_are_too_expensive():
    threshold = best_threshold(Y, AMOUNT, PROBABILITY, alert_cost=1_000.0)
    assert not threshold_rule(PROBABILITY, threshold).any()


def test_expected_cost_rule_is_optimal_for_calibrated_probabilities():
    """With exact probabilities, no fixed threshold has a lower expected cost."""
    rng = np.random.default_rng(0)
    n = 200_000
    probability = rng.beta(0.3, 20.0, n)
    amount = rng.gamma(2.0, 40.0, n)
    y = (rng.random(n) < probability).astype(int)
    alert_cost = 10.0

    rule_cost = total_cost(y, amount, expected_cost_rule(probability, amount, alert_cost), 10.0)
    threshold = best_threshold(y, amount, probability, alert_cost)
    threshold_cost = total_cost(y, amount, threshold_rule(probability, threshold), alert_cost)
    assert rule_cost < threshold_cost


def test_decision_report_fields():
    alert = np.array([True, False, True, False, False])
    report = decision_report(Y, AMOUNT, alert, alert_cost=10.0)
    assert report["n_alerts"] == 2
    assert report["precision"] == pytest.approx(0.5)
    assert report["recall"] == pytest.approx(1 / 3)
    assert report["alerts_per_100k"] == pytest.approx(40_000.0)
    assert report["savings"] == pytest.approx(1 - 54 / 134)
