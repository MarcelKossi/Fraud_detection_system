"""Models, imbalance treatments, search, calibration and cross-validation."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.calibration import METHODS, ScoreCalibrator, calibration_metrics
from fraud_detection.data import prepare
from fraud_detection.features import build_features
from fraud_detection.models import FAMILIES, build_model, positive_class_weight
from fraud_detection.sampling import STRATEGIES, resample_training_fold, undersample
from fraud_detection.search import SEARCH_SPACES, sample_configurations
from fraud_detection.splits import expanding_window_folds
from fraud_detection.validation import cross_validate, cross_validate_calibration

SMALL = {
    "logistic_regression": {"C": 1.0},
    "random_forest": {"n_estimators": 10},
    "lightgbm": {"n_estimators": 10, "num_leaves": 4, "min_child_samples": 5},
    "xgboost": {"n_estimators": 10, "max_depth": 2},
}


@pytest.fixture
def design(transactions):
    prepared = prepare(transactions, "Time")
    X, y = build_features(prepared, "Class", "Time", "hour_of_day")
    # Make the target learnable so that fitted models are not degenerate.
    X = X.assign(V1=X["V1"] + 3.0 * y)
    return X, y, expanding_window_folds(len(X), 3)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("pos_weight", [None, 20.0])
def test_every_family_fits_and_returns_probabilities(design, family, pos_weight):
    X, y, _ = design
    model = build_model(family, SMALL[family], seed=0, pos_weight=pos_weight).fit(X, y)
    probabilities = model.predict_proba(X)[:, 1]
    assert probabilities.shape == (len(X),)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="Unknown model family"):
        build_model("unknown", {}, seed=0)


def test_positive_class_weight_is_the_class_ratio():
    assert positive_class_weight(np.array([0] * 90 + [1] * 10)) == pytest.approx(9.0)


def test_undersampling_keeps_every_fraud_and_reaches_the_ratio(design):
    X, y, _ = design
    X_small, y_small = undersample(X, y, ratio=0.5, seed=0)
    assert y_small.sum() == y.sum()
    assert (y_small == 0).sum() == 2 * y.sum()
    assert X_small.index.is_monotonic_increasing
    assert X_small.index.equals(y_small.index)


def test_smote_adds_only_positive_examples(design):
    X, y, _ = design
    X_big, y_big = resample_training_fold(X, y, "smote", ratio=0.5, seed=0)
    assert (y_big == 0).sum() == (y == 0).sum()
    assert y_big.sum() == pytest.approx(0.5 * (y == 0).sum(), abs=1)
    assert len(X_big) == len(y_big)


@pytest.mark.parametrize("strategy", ["none", "class_weight"])
def test_loss_based_strategies_leave_the_data_untouched(design, strategy):
    X, y, _ = design
    X_out, y_out = resample_training_fold(X, y, strategy, ratio=0.5, seed=0)
    assert X_out is X and y_out is y


def test_unknown_strategy_is_rejected(design):
    X, y, _ = design
    with pytest.raises(ValueError, match="Unknown strategy"):
        resample_training_fold(X, y, "unknown", ratio=0.5, seed=0)


@pytest.mark.parametrize("family", list(SEARCH_SPACES))
def test_search_is_reproducible_and_json_serialisable(family):
    first = sample_configurations(family, 5, seed=42)
    assert first == sample_configurations(family, 5, seed=42)
    assert first != sample_configurations(family, 5, seed=43)
    for configuration in first:
        assert all(type(value) in (int, float) for value in configuration.values())
        build_model(family, configuration, seed=0)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_cross_validation_never_resamples_validation_data(design, strategy, monkeypatch):
    X, y, folds = design
    seen_sizes = []

    import fraud_detection.validation as validation

    original = validation.resample_training_fold

    def spy(X_train, y_train, *args, **kwargs):
        seen_sizes.append(len(X_train))
        return original(X_train, y_train, *args, **kwargs)

    monkeypatch.setattr(validation, "resample_training_fold", spy)
    result = cross_validate(
        "logistic_regression",
        {"C": 1.0},
        X,
        y,
        folds,
        seed=0,
        max_fpr=0.01,
        strategy=strategy,
        ratio=0.5,
    )
    assert seen_sizes == [len(train_idx) for train_idx, _ in folds]
    assert len(result["fold_auprc"]) == len(folds)
    assert result["auprc_mean"] == pytest.approx(np.mean(result["fold_auprc"]))


def test_cross_validation_is_reproducible(design):
    X, y, folds = design
    kwargs = dict(seed=3, max_fpr=0.01, strategy="undersampling", ratio=0.5)
    first = cross_validate("lightgbm", SMALL["lightgbm"], X, y, folds, **kwargs)
    second = cross_validate("lightgbm", SMALL["lightgbm"], X, y, folds, **kwargs)
    assert first == second


@pytest.mark.parametrize("method", METHODS)
def test_calibrators_return_probabilities_and_preserve_order(method):
    rng = np.random.default_rng(0)
    y = (rng.random(4_000) < 0.1).astype(int)
    scores = np.clip(0.5 * y + rng.normal(0.3, 0.1, len(y)), 0.001, 0.999)
    calibrated = ScoreCalibrator(method).fit(scores, y).transform(scores)
    assert np.all((calibrated >= 0) & (calibrated <= 1))
    order = np.argsort(scores)
    assert np.all(np.diff(calibrated[order]) >= -1e-12)


def test_calibration_corrects_inflated_scores():
    rng = np.random.default_rng(1)
    y = (rng.random(20_000) < 0.02).astype(int)
    inflated = np.clip(0.5 + 0.4 * y + rng.normal(0, 0.05, len(y)), 0.001, 0.999)
    half = len(y) // 2
    raw = calibration_metrics(y[half:], inflated[half:])
    for method in ("sigmoid", "isotonic"):
        calibrator = ScoreCalibrator(method).fit(inflated[:half], y[:half])
        calibrated = calibration_metrics(y[half:], calibrator.transform(inflated[half:]))
        assert calibrated["brier"] < raw["brier"]
        assert calibrated["log_loss"] < raw["log_loss"]


def test_unknown_calibration_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown method"):
        ScoreCalibrator("unknown")


def test_calibration_cross_validation_reports_every_method(design):
    X, y, folds = design
    result = cross_validate_calibration(
        "logistic_regression",
        {"C": 1.0},
        X,
        y,
        folds,
        seed=0,
        strategy="class_weight",
        ratio=0.5,
        calibration_fraction=0.3,
    )
    assert set(result) == set(METHODS)
    for metrics in result.values():
        assert all(len(values) == len(folds) for values in metrics.values())
    assert isinstance(X, pd.DataFrame)
