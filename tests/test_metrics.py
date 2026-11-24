import numpy as np
import pytest

from fraud_detection.metrics import (
    auprc,
    bootstrap_ci,
    evaluate_scores,
    recall_at_fpr,
    threshold_metrics,
)


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    y = (rng.random(5_000) < 0.02).astype(int)
    scores = np.clip(0.6 * y + rng.normal(0.2, 0.15, len(y)), 0, 1)
    return y, scores


def test_auprc_of_a_perfect_ranking_is_one():
    y = np.array([0, 0, 0, 1, 1])
    assert auprc(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == pytest.approx(1.0)


def test_auprc_of_uninformative_scores_equals_prevalence():
    y = np.array([0] * 90 + [1] * 10)
    assert auprc(y, np.full(100, 0.5)) == pytest.approx(0.1)


def test_recall_at_fpr_on_a_hand_computed_case():
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.95])
    # Without any false positive only the highest-scored fraud is caught.
    assert recall_at_fpr(y, scores, max_fpr=0.0) == pytest.approx(0.5)
    # Allowing one false positive out of four legitimate transactions catches both.
    assert recall_at_fpr(y, scores, max_fpr=0.25) == pytest.approx(1.0)


def test_threshold_metrics_match_the_confusion_matrix():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.6, 0.2, 0.7, 0.4, 0.9])
    report = threshold_metrics(y, scores, 0.5)
    assert report["confusion_matrix"] == [[2, 1], [1, 2]]
    assert report["precision"] == pytest.approx(2 / 3)
    assert report["recall"] == pytest.approx(2 / 3)


def test_bootstrap_interval_contains_the_point_estimate_and_is_reproducible(scored):
    y, scores = scored
    first = bootstrap_ci(y, scores, auprc, 200, 0.95, seed=1)
    second = bootstrap_ci(y, scores, auprc, 200, 0.95, seed=1)
    assert first == second
    assert first[0] < auprc(y, scores) < first[1]


def test_evaluate_scores_adds_intervals_only_on_request(scored):
    y, scores = scored
    plain = evaluate_scores(y, scores, threshold=0.5, max_fpr=0.001)
    assert "auprc_ci" not in plain
    assert plain["n_positive"] == int(y.sum())

    full = evaluate_scores(y, scores, threshold=0.5, max_fpr=0.001, n_resamples=100, seed=0)
    assert len(full["auprc_ci"]) == 2
    assert len(full["recall_at_fpr"]["ci"]) == 2
