"""The deployable model and the prediction service."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_detection.serving import AMOUNT_FIELD, RAW_TIME_FIELD, FraudModel, fit_fraud_model

REPOSITORY = Path(__file__).resolve().parents[1]
SHIPPED_MODEL = REPOSITORY / "models" / "fraud_model"
COMPONENTS = [f"V{i}" for i in range(1, 29)]


def _transactions(n: int = 4_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, 28)), columns=COMPONENTS)
    frame[AMOUNT_FIELD] = rng.gamma(2.0, 40.0, n).round(2)
    frame.insert(0, RAW_TIME_FIELD, np.sort(rng.integers(0, 172_800, n)).astype(float))
    logit = 2.5 * frame["V1"] - 1.5 * frame["V2"] - 3.0
    frame["Class"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return frame


@pytest.fixture(scope="module")
def transactions() -> pd.DataFrame:
    return _transactions()


@pytest.fixture(scope="module")
def model(transactions) -> FraudModel:
    from fraud_detection.features import add_hour_of_day

    X = add_hour_of_day(transactions.drop(columns="Class"), RAW_TIME_FIELD)
    return fit_fraud_model(
        X,
        transactions["Class"],
        params={"n_estimators": 40, "max_depth": 3},
        seed=0,
        calibration_fraction=0.25,
        alert_cost=10.0,
        metadata={"model_version": "test"},
    )


@pytest.fixture(scope="module")
def client(model, tmp_path_factory):
    directory = model.save(tmp_path_factory.mktemp("model"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("FRAUD_MODEL_DIR", str(directory))
        from api.main import app

        with TestClient(app) as test_client:
            yield test_client


def _payload(transactions: pd.DataFrame, position: int = 0) -> dict:
    row = transactions.iloc[position]
    payload = {name: float(row[name]) for name in COMPONENTS}
    payload["time_seconds"] = float(row[RAW_TIME_FIELD])
    payload["amount"] = float(row[AMOUNT_FIELD])
    return payload


def test_model_round_trip_preserves_predictions(model, transactions, tmp_path):
    reloaded = FraudModel.load(model.save(tmp_path))
    np.testing.assert_allclose(
        reloaded.predict_proba(transactions), model.predict_proba(transactions), atol=1e-7
    )
    assert reloaded.feature_names == model.feature_names
    assert reloaded.alert_cost == model.alert_cost
    assert reloaded.metadata["model_version"] == "test"
    assert not list(tmp_path.glob("*.pkl"))


def test_probabilities_do_not_depend_on_column_order_nor_on_the_day(model, transactions):
    reference = model.predict_proba(transactions)
    shuffled = transactions[transactions.columns[::-1]]
    np.testing.assert_allclose(model.predict_proba(shuffled), reference)
    next_day = transactions.assign(**{RAW_TIME_FIELD: transactions[RAW_TIME_FIELD] + 86_400})
    np.testing.assert_allclose(model.predict_proba(next_day), reference, atol=1e-7)


def test_missing_field_is_reported(model, transactions):
    with pytest.raises(ValueError, match="Missing fields"):
        model.predict_proba(transactions.drop(columns="V3"))


def test_decision_follows_the_expected_cost_rule(model):
    decision = model.decide(np.array([0.5, 0.5, 0.01]), np.array([30.0, 10.0, 2_000.0]))
    assert decision["expected_loss"].tolist() == [15.0, 5.0, 20.0]
    assert decision["alert"].tolist() == [True, False, True]


def test_explanation_adds_up_to_the_calibrated_log_odds(model, transactions):
    row = transactions.iloc[[7]]
    explanation = model.explain(row)
    probability = float(model.predict_proba(row)[0])
    total = explanation["base_value"] + sum(
        term["contribution"] for term in explanation["contributions"]
    )
    assert total == pytest.approx(explanation["prediction"], abs=1e-6)
    assert total == pytest.approx(np.log(probability / (1 - probability)), abs=1e-3)
    assert len(explanation["contributions"]) == len(model.feature_names)


def test_health_and_version(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    version = client.get("/version").json()
    assert version["decision"] == {"rule": "expected_cost", "alert_cost": 10.0}
    assert version["input_fields"][:2] == ["time_seconds", "amount"]
    assert len(version["input_fields"]) == 30


def test_predict_matches_the_model(client, model, transactions):
    response = client.post("/predict", json=_payload(transactions, 3))
    assert response.status_code == 200
    body = response.json()
    expected = float(model.predict_proba(transactions.iloc[[3]])[0])
    assert body["fraud_probability"] == pytest.approx(expected, abs=1e-7)
    amount = float(transactions[AMOUNT_FIELD].iloc[3])
    assert body["decision"]["expected_loss"] == pytest.approx(expected * amount, abs=1e-6)
    assert body["decision"]["alert"] == (expected * amount > 10.0)
    assert body["model_version"] == "test"
    assert "explanation" not in body


def test_predict_with_explanation(client, transactions):
    response = client.post(
        "/predict", params={"explain": True, "top_k": 3}, json=_payload(transactions)
    )
    explanation = response.json()["explanation"]
    assert len(explanation["contributions"]) == 3
    total = (
        explanation["base_value"]
        + sum(term["contribution"] for term in explanation["contributions"])
        + explanation["other_contributions"]
    )
    assert total == pytest.approx(explanation["prediction"], abs=1e-6)


@pytest.mark.parametrize(
    "change",
    [
        {"V5": None},
        {"amount": -1.0},
        {"time_seconds": -5.0},
        {"V1": "abc"},
        {"unknown_field": 1.0},
    ],
)
def test_invalid_requests_are_rejected(client, transactions, change):
    payload = _payload(transactions)
    for key, value in change.items():
        if value is None:
            payload.pop(key)
        else:
            payload[key] = value
    assert client.post("/predict", json=payload).status_code == 422


def test_non_finite_values_are_rejected(client, transactions):
    body = json.dumps(_payload(transactions)).replace(
        f'"V2": {_payload(transactions)["V2"]}', '"V2": NaN'
    )
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_service_without_model_reports_it(tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("FRAUD_MODEL_DIR", str(tmp_path / "absent"))
        from api.main import app

        with TestClient(app) as test_client:
            assert test_client.get("/health").json() == {
                "status": "degraded",
                "model_loaded": False,
            }
            assert test_client.get("/version").status_code == 503


@pytest.mark.skipif(not SHIPPED_MODEL.exists(), reason="no model shipped with the repository")
def test_shipped_model_reproduces_its_recorded_examples():
    """Guards the committed model files against corruption or format drift."""
    model = FraudModel.load(SHIPPED_MODEL)
    examples = json.loads((SHIPPED_MODEL / "examples.json").read_text(encoding="utf-8"))
    assert examples
    for example in examples:
        fields = dict(example["transaction"])
        fields[AMOUNT_FIELD] = fields.pop("amount")
        frame = pd.DataFrame([fields])
        probability = float(model.predict_proba(frame)[0])
        assert probability == pytest.approx(example["expected"]["fraud_probability"], abs=1e-6)
        decision = model.decide(probability, fields[AMOUNT_FIELD])
        assert bool(decision["alert"]) == example["expected"]["alert"]
