import numpy as np
import pandas as pd
import pytest

from fraud_detection.explain import (
    explain_transaction,
    global_importance,
    predicted_log_odds,
    shap_contributions,
)
from fraud_detection.models import build_model


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    n = 3_000
    X = pd.DataFrame(
        {
            "signal": rng.normal(size=n),
            "weak": rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    logit = 3.0 * X["signal"] + 0.5 * X["weak"] - 4.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    model = build_model("xgboost", {"n_estimators": 50, "max_depth": 3}, seed=0).fit(X, y)
    return model, X


def test_contributions_add_up_to_the_predicted_log_odds(fitted):
    model, X = fitted
    contributions, base = shap_contributions(model, X)
    assert contributions.shape == X.shape
    np.testing.assert_allclose(
        contributions.sum(axis=1) + base, predicted_log_odds(model, X), atol=1e-3
    )


def test_base_value_is_the_same_for_every_transaction(fitted):
    model, X = fitted
    _, base = shap_contributions(model, X)
    assert np.ptp(base) == pytest.approx(0.0, abs=1e-6)


def test_global_importance_ranks_the_informative_feature_first(fitted):
    model, X = fitted
    contributions, _ = shap_contributions(model, X)
    importance = global_importance(contributions, list(X.columns))
    assert list(importance.index) == ["signal", "weak", "noise"]
    assert importance.is_monotonic_decreasing


def test_explanation_of_one_transaction_is_complete_and_ordered(fitted):
    model, X = fitted
    row = X.iloc[[5]]
    explanation = explain_transaction(model, row)
    terms = explanation["contributions"]
    assert [term["feature"] for term in terms][0] in X.columns
    assert len(terms) == X.shape[1]
    magnitudes = [abs(term["contribution"]) for term in terms]
    assert magnitudes == sorted(magnitudes, reverse=True)
    total = explanation["base_log_odds"] + sum(term["contribution"] for term in terms)
    assert total == pytest.approx(explanation["log_odds"], abs=1e-6)
    assert explanation["log_odds"] == pytest.approx(
        float(predicted_log_odds(model, row)[0]), abs=1e-3
    )
    assert explanation["other_contributions"] == pytest.approx(0.0, abs=1e-9)


def test_truncated_explanation_still_adds_up(fitted):
    model, X = fitted
    explanation = explain_transaction(model, X.iloc[[5]], top_k=1)
    assert len(explanation["contributions"]) == 1
    total = (
        explanation["base_log_odds"]
        + explanation["contributions"][0]["contribution"]
        + explanation["other_contributions"]
    )
    assert total == pytest.approx(explanation["log_odds"], abs=1e-6)


def test_explanation_requires_a_single_row_and_an_xgboost_model(fitted):
    model, X = fitted
    with pytest.raises(ValueError, match="exactly one"):
        explain_transaction(model, X.iloc[:2])
    other = build_model("logistic_regression", {"C": 1.0}, seed=0).fit(X, [0, 1] * (len(X) // 2))
    with pytest.raises(TypeError, match="XGBoost"):
        shap_contributions(other, X)
