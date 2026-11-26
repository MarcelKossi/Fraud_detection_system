"""Train and save the model served by the API.

The model is trained on the development period only: the classifier on its first part,
the sigmoid calibrator on its last part. The hold-out set is not read, so that the saved
model can be evaluated on it once, in the final step.

The model is written to the directory given by ``model_dir`` in the configuration,
together with a few example transactions of the development period and the outputs of
the model for them. These examples are used by the tests and by the dashboard.

Usage::

    python scripts/train_final_model.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fraud_detection import __version__
from fraud_detection.config import load_config
from fraud_detection.data import load_raw, prepare, sha256_of_file
from fraud_detection.experiment import run_metadata
from fraud_detection.features import build_features
from fraud_detection.serving import AMOUNT_FIELD, RAW_TIME_FIELD, FraudModel, fit_fraud_model
from fraud_detection.splits import temporal_holdout

EXAMPLES_FILE = "examples.json"


def select_examples(model: FraudModel, transactions, target: str) -> list[dict]:
    """Four development transactions that illustrate the behaviour of the model.

    A fraud that raises an alert, a fraud that does not, a legitimate transaction that
    raises an alert (if any) and an ordinary legitimate transaction.
    """
    probability = model.predict_proba(transactions)
    decision = model.decide(probability, transactions[AMOUNT_FIELD].to_numpy())
    is_fraud = transactions[target].to_numpy() == 1
    alert = decision["alert"]

    def pick(mask: np.ndarray, by: np.ndarray, largest: bool) -> int | None:
        candidates = np.flatnonzero(mask)
        if len(candidates) == 0:
            return None
        ranked = candidates[np.argsort(by[candidates])]
        return int(ranked[-1] if largest else ranked[0])

    chosen = {
        "Fraud, alert raised": pick(is_fraud & alert, decision["expected_loss"], True),
        "Fraud, no alert (small amount)": pick(is_fraud & ~alert, probability, True),
        "Legitimate, alert raised": pick(~is_fraud & alert, decision["expected_loss"], True),
        "Legitimate, no alert": pick(~is_fraud & ~alert, probability, False),
    }
    examples = []
    for label, position in chosen.items():
        if position is None:
            continue
        row = transactions.iloc[[position]]
        examples.append(
            {
                "label": label,
                "is_fraud": bool(is_fraud[position]),
                # Field names of the API request: the amount is lower-cased.
                "transaction": {
                    ("amount" if name == AMOUNT_FIELD else name): float(row[name].iloc[0])
                    for name in model.input_fields
                },
                "expected": {
                    "fraud_probability": float(probability[position]),
                    "expected_loss": float(decision["expected_loss"][position]),
                    "alert": bool(alert[position]),
                },
            }
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    selected, selection = config.selected_model, config.model_selection
    if selected.family != "xgboost" or selected.calibration != "sigmoid":
        raise ValueError("The serving model supports XGBoost with sigmoid calibration only")
    if selected.imbalance != "none":
        raise ValueError("The serving model is trained without imbalance treatment")

    prepared = prepare(load_raw(config.data.raw_path, config.data.sha256), config.data.time_column)
    dev_idx, holdout_idx = temporal_holdout(len(prepared), config.split.test_fraction)
    development = prepared.iloc[dev_idx]
    X, y = build_features(
        development, config.data.target, config.data.time_column, selection.feature_set
    )

    provenance = run_metadata(config.to_dict(), sha256_of_file(config.data.raw_path))
    model = fit_fraud_model(
        X,
        y,
        params=selected.params,
        seed=config.seed,
        calibration_fraction=selection.calibration_fraction,
        alert_cost=config.cost.alert_cost,
        metadata={
            "model_version": __version__,
            "family": selected.family,
            "params": selected.params,
            "seed": config.seed,
            "training_period": "development set (hold-out set excluded)",
            "n_holdout_excluded": int(len(holdout_idx)),
            "trained_at_utc": provenance["timestamp_utc"],
            "git_commit": provenance["git_commit"],
            "git_uncommitted_changes": provenance["git_uncommitted_changes"],
            "code_sha256": provenance["code_sha256"],
            "data_sha256": provenance["data_sha256"],
            "environment": provenance["environment"],
        },
    )
    directory = model.save(config.model_dir)

    raw_fields = development.rename(columns={config.data.time_column: RAW_TIME_FIELD})
    examples = select_examples(model, raw_fields, config.data.target)
    with open(Path(directory) / EXAMPLES_FILE, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(examples, handle, indent=2)
        handle.write("\n")

    reloaded = FraudModel.load(directory)
    check = raw_fields.iloc[:1000]
    if not np.allclose(model.predict_proba(check), reloaded.predict_proba(check), atol=1e-7):
        raise RuntimeError("The reloaded model does not reproduce the trained model")
    print(f"Model saved to {directory} ({len(examples)} examples); reload check passed.")
    print(
        f"Calibration: slope {model.calibration_slope:.4f}, "
        f"intercept {model.calibration_intercept:.4f}"
    )


if __name__ == "__main__":
    main()
