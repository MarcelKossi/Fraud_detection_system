import numpy as np
import pytest

from fraud_detection.data import deduplicate, load_raw, prepare, sha256_of_file
from fraud_detection.features import build_features


def test_deduplicate_removes_exact_copies_only(transactions):
    clean = deduplicate(transactions)
    assert len(clean) == len(transactions) - 3
    assert not clean.duplicated().any()


def test_prepare_orders_chronologically(transactions):
    prepared = prepare(transactions.sample(frac=1.0, random_state=1), "Time")
    assert prepared["Time"].is_monotonic_increasing
    assert not prepared.duplicated().any()


def test_load_raw_rejects_wrong_checksum(tmp_path, transactions):
    path = tmp_path / "data.csv"
    transactions.to_csv(path, index=False)
    assert len(load_raw(path, sha256_of_file(path))) == len(transactions)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_raw(path, "0" * 64)


def test_load_raw_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        load_raw(tmp_path / "absent.csv")


def test_raw_time_feature_set_keeps_all_columns(transactions):
    X, y = build_features(transactions, "Class", "Time", "raw_time")
    assert list(X.columns) == ["Time", "V1", "V2", "Amount"]
    assert y.name == "Class"


def test_hour_of_day_is_periodic_and_drops_elapsed_time(transactions):
    X, _ = build_features(transactions, "Class", "Time", "hour_of_day")
    assert "Time" not in X.columns
    assert np.allclose(X["hour_sin"] ** 2 + X["hour_cos"] ** 2, 1.0)

    shifted = transactions.assign(Time=transactions["Time"] + 86_400)
    X_shifted, _ = build_features(shifted, "Class", "Time", "hour_of_day")
    assert np.allclose(X[["hour_sin", "hour_cos"]], X_shifted[["hour_sin", "hour_cos"]])


def test_unknown_feature_set_is_rejected(transactions):
    with pytest.raises(ValueError, match="Unknown feature set"):
        build_features(transactions, "Class", "Time", "unknown")
