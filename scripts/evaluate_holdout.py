"""Final evaluation on the hold-out sets. To be run once.

Main dataset
------------
The stored model (``models/fraud_model``), trained on the development period, is
evaluated on the temporal hold-out set, which no modelling choice has used. It is
compared, on the same transactions, with the model and the decision rule of the first
version: a random forest with default settings and a probability threshold of 0.5,
trained on the same development period. Differences are given with paired bootstrap
confidence intervals.

Simulated dataset
-----------------
The retained configuration is trained on the development period of the simulated
transactions, minus the label delay, and evaluated on their hold-out period with raw
features and with all behavioural features.

The script refuses to overwrite an existing result, so that the hold-out sets are not
evaluated repeatedly. Results are written to ``results/final/``.

Usage::

    python scripts/evaluate_holdout.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fraud_detection.behavioural import (
    SECONDS_PER_DAY,
    add_behavioural_features,
    feature_columns,
)
from fraud_detection.calibration import calibration_metrics
from fraud_detection.config import Config, load_config
from fraud_detection.costs import (
    decision_report,
    expected_cost_rule,
    no_detection_cost,
    oracle_cost,
    savings,
    threshold_rule,
    total_cost,
)
from fraud_detection.data import load_raw, prepare, sha256_of_file
from fraud_detection.experiment import run_metadata, write_result
from fraud_detection.features import build_features
from fraud_detection.metrics import auprc, evaluate_scores, recall_at_fpr
from fraud_detection.models import build_baseline, build_model
from fraud_detection.serving import RAW_TIME_FIELD, FraudModel
from fraud_detection.simulation import simulate_transactions, table_digest
from fraud_detection.splits import (
    assert_chronological,
    assert_disjoint,
    temporal_holdout,
    time_holdout,
)


def _interval(values: list[float], level: float) -> list[float]:
    alpha = (1.0 - level) / 2.0
    return [float(q) for q in np.quantile(values, [alpha, 1.0 - alpha])]


def paired_bootstrap(y, amount, new, old, config: Config) -> dict:
    """Bootstrap intervals of the metrics of two systems and of their differences.

    ``new`` and ``old`` each hold the scores and the alerts of one system on the same
    transactions; the same resamples are applied to both.
    """
    ev, alert_cost = config.evaluation, config.cost.alert_cost
    rng = np.random.default_rng(config.seed)
    n = len(y)

    def metrics(index, system) -> dict[str, float]:
        reference = no_detection_cost(y[index], amount[index])
        cost = total_cost(y[index], amount[index], system["alert"][index], alert_cost)
        return {
            "auprc": auprc(y[index], system["scores"][index]),
            "recall_at_fpr": recall_at_fpr(y[index], system["scores"][index], ev.max_fpr),
            "savings": savings(cost, reference),
        }

    samples = {"new": [], "old": [], "difference": []}
    for _ in range(ev.bootstrap_resamples):
        index = rng.integers(0, n, n)
        if y[index].sum() == 0:
            continue
        m_new, m_old = metrics(index, new), metrics(index, old)
        samples["new"].append(m_new)
        samples["old"].append(m_old)
        samples["difference"].append({k: m_new[k] - m_old[k] for k in m_new})

    everything = np.arange(n)
    point = {"new": metrics(everything, new), "old": metrics(everything, old)}
    point["difference"] = {k: point["new"][k] - point["old"][k] for k in point["new"]}
    return {
        name: {
            key: {
                "value": point[name][key],
                "ci": _interval([s[key] for s in samples[name]], ev.confidence_level),
            }
            for key in point[name]
        }
        for name in samples
    } | {
        "share_of_resamples_where_new_is_better": {
            key: float(np.mean([s[key] > 0 for s in samples["difference"]]))
            for key in point["difference"]
        },
        "share_of_resamples_with_no_difference": {
            key: float(np.mean([s[key] == 0 for s in samples["difference"]]))
            for key in point["difference"]
        },
    }


def evaluate_main_dataset(config: Config) -> dict:
    data, ev, cost_config = config.data, config.evaluation, config.cost
    prepared = prepare(load_raw(data.raw_path, data.sha256), data.time_column)
    dev_idx, test_idx = temporal_holdout(len(prepared), config.split.test_fraction)
    assert_disjoint(dev_idx, test_idx)
    assert_chronological(prepared[data.time_column].to_numpy(), dev_idx, test_idx)
    holdout = prepared.iloc[test_idx]
    y = holdout[data.target].to_numpy()
    amount = holdout[cost_config.amount_column].to_numpy(dtype=float)

    model = FraudModel.load(config.model_dir)
    if model.metadata["data_sha256"] != data.sha256:
        raise ValueError("The stored model was not trained on the configured data file")
    if model.metadata["n_fit"] + model.metadata["n_calibration"] != len(dev_idx):
        raise ValueError("The stored model was not trained on the development set")
    probability = model.predict_proba(holdout.rename(columns={data.time_column: RAW_TIME_FIELD}))

    # First version: random forest with default settings, raw features, threshold 0.5.
    X_dev, y_dev = build_features(prepared.iloc[dev_idx], data.target, data.time_column, "raw_time")
    X_test, _ = build_features(holdout, data.target, data.time_column, "raw_time")
    first_version = build_baseline(config.baseline_model.n_estimators, config.seed)
    first_version.fit(X_dev, y_dev)
    first_scores = first_version.predict_proba(X_test)[:, 1]

    systems = {
        "first_version": {
            "scores": first_scores,
            "alert": threshold_rule(first_scores, ev.threshold),
        },
        "retained_model_threshold": {
            "scores": probability,
            "alert": threshold_rule(probability, ev.threshold),
        },
        "retained_model_expected_cost": {
            "scores": probability,
            "alert": expected_cost_rule(probability, amount, cost_config.alert_cost),
        },
    }
    reports = {}
    for name, system in systems.items():
        reports[name] = {
            "ranking": evaluate_scores(
                y,
                system["scores"],
                threshold=ev.threshold,
                max_fpr=ev.max_fpr,
                n_resamples=ev.bootstrap_resamples,
                confidence_level=ev.confidence_level,
                seed=config.seed,
            ),
            "decision": decision_report(y, amount, system["alert"], cost_config.alert_cost),
        }
    reports["retained_model_expected_cost"]["calibration"] = calibration_metrics(y, probability)
    reports["retained_model_threshold"]["calibration"] = calibration_metrics(y, probability)

    sensitivity = {}
    for alert_cost in cost_config.sensitivity_alert_costs:
        reference = no_detection_cost(y, amount)
        sensitivity[f"{alert_cost:g}"] = {
            "first_version": savings(
                total_cost(y, amount, systems["first_version"]["alert"], alert_cost), reference
            ),
            "retained_model_threshold": savings(
                total_cost(y, amount, threshold_rule(probability, ev.threshold), alert_cost),
                reference,
            ),
            "retained_model_expected_cost": savings(
                total_cost(
                    y, amount, expected_cost_rule(probability, amount, alert_cost), alert_cost
                ),
                reference,
            ),
            "oracle": savings(oracle_cost(y, amount, alert_cost), reference),
        }

    # Per-transaction scores are kept so that further analyses of this evaluation
    # (ties, reliability, other decision rules) do not require reading the data again.
    scores_path = Path(config.results_dir) / "final" / "holdout_scores.npz"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scores_path,
        is_fraud=y,
        amount=amount,
        first_version_score=first_scores,
        retained_model_probability=probability,
    )

    return {
        "n_holdout": int(len(y)),
        "n_holdout_frauds": int(y.sum()),
        "fraudulent_amount": no_detection_cost(y, amount),
        "alert_cost": cost_config.alert_cost,
        "model": {
            key: model.metadata[key] for key in ("model_version", "git_commit", "trained_at_utc")
        },
        "systems": reports,
        "savings_by_alert_cost": sensitivity,
        "comparison_with_first_version": {
            "retained_model_expected_cost": paired_bootstrap(
                y, amount, systems["retained_model_expected_cost"], systems["first_version"], config
            ),
            "retained_model_threshold": paired_bootstrap(
                y, amount, systems["retained_model_threshold"], systems["first_version"], config
            ),
        },
    }


def evaluate_simulated_dataset(config: Config) -> dict:
    settings, selected, ev = config.behavioural, config.selected_model, config.evaluation
    transactions = simulate_transactions(config.simulation)
    table = add_behavioural_features(transactions, settings.windows_days, settings.delay_days)
    times = table["time_seconds"].to_numpy()
    dev_idx, test_idx = time_holdout(times, settings.test_fraction)
    assert_disjoint(dev_idx, test_idx)
    gap = settings.delay_days * SECONDS_PER_DAY
    train_idx = dev_idx[times[dev_idx] < times[test_idx].min() - gap]
    y = table["is_fraud"].to_numpy()

    feature_sets = {}
    for name in ("raw", "all"):
        columns = feature_columns(settings.feature_sets[name], settings.windows_days)
        X = table[columns]
        values = {"auprc": [], "recall_at_fpr": []}
        for seed in settings.seeds:
            model = build_model(selected.family, selected.params, seed)
            model.fit(X.iloc[train_idx], y[train_idx])
            scores = model.predict_proba(X.iloc[test_idx])[:, 1]
            values["auprc"].append(auprc(y[test_idx], scores))
            values["recall_at_fpr"].append(recall_at_fpr(y[test_idx], scores, ev.max_fpr))
            print(
                f"  simulated / {name} / seed {seed}: AUPRC {values['auprc'][-1]:.4f}", flush=True
            )
        feature_sets[name] = {
            "n_features": len(columns),
            "auprc_mean": float(np.mean(values["auprc"])),
            "auprc_seed_std": float(np.std(values["auprc"], ddof=1)),
            "recall_at_fpr_mean": float(np.mean(values["recall_at_fpr"])),
        }
    return {
        "simulated_table_sha256": table_digest(transactions),
        "n_train": int(len(train_idx)),
        "n_train_frauds": int(y[train_idx].sum()),
        "n_holdout": int(len(test_idx)),
        "n_holdout_frauds": int(y[test_idx].sum()),
        "seeds": settings.seeds,
        "feature_sets": feature_sets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    out_path = Path(config.results_dir) / "final" / "holdout_evaluation.json"
    if out_path.exists():
        raise SystemExit(
            f"{out_path} already exists. The hold-out sets are meant to be evaluated once; "
            "delete the file deliberately if a new evaluation is justified, and record why."
        )

    print("Main dataset ...", flush=True)
    main_dataset = evaluate_main_dataset(config)
    print("Simulated dataset ...", flush=True)
    simulated = evaluate_simulated_dataset(config)
    payload = {
        "experiment": "holdout_evaluation",
        "main_dataset": main_dataset,
        "simulated_dataset": simulated,
        "metadata": run_metadata(config.to_dict(), sha256_of_file(config.data.raw_path)),
    }
    print(f"Written to {write_result(out_path, payload)}")
    print(json.dumps(main_dataset["comparison_with_first_version"], indent=1)[:1500])


if __name__ == "__main__":
    main()
