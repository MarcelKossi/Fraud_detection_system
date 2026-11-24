"""Cost-based comparison of decision rules for the retained model.

Calibrated probabilities are obtained on each validation fold of the development
period, then four decision rules are costed for several values of the alert cost:

``default_threshold``
    Alert when the probability reaches 0.5, the rule of the first version.
``cost_optimal_threshold``
    A single probability threshold minimising the cost. For each fold, it is chosen on
    the other folds and applied to that fold.
``expected_cost``
    Alert when probability x amount exceeds the alert cost. No parameter is fitted.
``oracle``
    Alert exactly on the frauds worth more than an alert: the lowest achievable cost.

The hold-out set is not read. Results are written to ``results/cost/``.

Usage::

    python scripts/run_cost_analysis.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fraud_detection.config import load_config
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
from fraud_detection.data import load_raw, prepare, sha256_of_file
from fraud_detection.experiment import run_metadata, write_result
from fraud_detection.features import build_features
from fraud_detection.splits import expanding_window_folds, temporal_holdout
from fraud_detection.validation import out_of_fold_probabilities

RULES = ("default_threshold", "cost_optimal_threshold", "expected_cost")


def _mean_reports(reports: list[dict]) -> dict:
    return {key: float(np.mean([report[key] for report in reports])) for key in reports[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    selection, selected, cost_config = config.model_selection, config.selected_model, config.cost

    prepared = prepare(load_raw(config.data.raw_path, config.data.sha256), config.data.time_column)
    dev_idx, _ = temporal_holdout(len(prepared), config.split.test_fraction)
    development = prepared.iloc[dev_idx]
    folds = expanding_window_folds(len(development), config.split.n_cv_folds)
    X, y = build_features(
        development, config.data.target, config.data.time_column, selection.feature_set
    )
    amount = development[cost_config.amount_column].to_numpy(dtype=float)
    y_values = y.to_numpy()
    validation_idx = np.concatenate([val_idx for _, val_idx in folds])
    fold_of_row = np.concatenate([np.full(len(val), i) for i, (_, val) in enumerate(folds)])
    y_val, amount_val = y_values[validation_idx], amount[validation_idx]

    probabilities_by_seed = []
    for seed in selection.seeds:
        print(f"Out-of-fold probabilities, seed {seed}", flush=True)
        per_fold = out_of_fold_probabilities(
            selected.family,
            selected.params,
            X,
            y,
            folds,
            seed=seed,
            strategy=selected.imbalance,
            ratio=selection.resampling_ratio,
            calibration=selected.calibration,
            calibration_fraction=selection.calibration_fraction,
        )
        probabilities_by_seed.append(np.concatenate(per_fold))

    reference = no_detection_cost(y_val, amount_val)
    fraud_amounts = amount_val[y_val == 1]
    by_cost = {}
    for alert_cost in cost_config.sensitivity_alert_costs:
        reports = {rule: [] for rule in RULES}
        alerts_by_rule = {rule: [] for rule in RULES}
        thresholds = []
        for probability in probabilities_by_seed:
            alerts_by_rule["default_threshold"].append(threshold_rule(probability, 0.5))
            alerts_by_rule["expected_cost"].append(
                expected_cost_rule(probability, amount_val, alert_cost)
            )
            # Threshold of each fold chosen on the other folds.
            alert = np.zeros(len(y_val), dtype=bool)
            for fold in range(len(folds)):
                inside = fold_of_row == fold
                threshold = best_threshold(
                    y_val[~inside], amount_val[~inside], probability[~inside], alert_cost
                )
                alert[inside] = threshold_rule(probability[inside], threshold)
                thresholds.append(threshold)
            alerts_by_rule["cost_optimal_threshold"].append(alert)
            for rule in RULES:
                reports[rule].append(
                    decision_report(y_val, amount_val, alerts_by_rule[rule][-1], alert_cost)
                )
        oracle = oracle_cost(y_val, amount_val, alert_cost)
        per_fold = {
            rule: [
                float(
                    np.mean(
                        [
                            savings(
                                total_cost(
                                    y_val[fold_of_row == fold],
                                    amount_val[fold_of_row == fold],
                                    alerts[fold_of_row == fold],
                                    alert_cost,
                                ),
                                no_detection_cost(
                                    y_val[fold_of_row == fold], amount_val[fold_of_row == fold]
                                ),
                            )
                            for alerts in alerts_by_rule[rule]
                        ]
                    )
                )
                for fold in range(len(folds))
            ]
            for rule in RULES
        }
        by_cost[f"{alert_cost:g}"] = {
            "alert_cost": alert_cost,
            "rules": {rule: _mean_reports(reports[rule]) for rule in RULES},
            "savings_per_fold": per_fold,
            "savings_seed_std": {
                rule: float(np.std([r["savings"] for r in reports[rule]], ddof=1)) for rule in RULES
            },
            "cost_optimal_threshold_mean": float(np.mean(thresholds)),
            "cost_optimal_threshold_range": [float(np.min(thresholds)), float(np.max(thresholds))],
            "oracle": {"cost": oracle, "savings": savings(oracle, reference)},
        }
        print(
            f"  alert cost {alert_cost:g}: savings "
            + ", ".join(
                f"{rule} {by_cost[f'{alert_cost:g}']['rules'][rule]['savings']:.3f}"
                for rule in RULES
            )
        )

    payload = {
        "experiment": "cost_analysis",
        "n_validation_transactions": int(len(y_val)),
        "n_validation_frauds": int(y_val.sum()),
        "no_detection_cost": reference,
        "fraud_amount_quantiles": {
            f"{q:g}": float(np.quantile(fraud_amounts, q)) for q in (0.25, 0.5, 0.75, 0.9)
        },
        "fraud_amount_share_below_alert_cost": {
            f"{c:g}": float((fraud_amounts <= c).mean())
            for c in cost_config.sensitivity_alert_costs
        },
        "seeds": selection.seeds,
        "alert_costs": by_cost,
        "metadata": run_metadata(config.to_dict(), sha256_of_file(config.data.raw_path)),
    }
    out_dir = Path(config.results_dir) / "cost"
    print(f"Written to {write_result(out_dir / 'decision_rules.json', payload)}")
    write_summary(out_dir)


def write_summary(out_dir: Path) -> None:
    """Regenerate ``summary.md`` from ``decision_rules.json``."""
    result = json.loads((out_dir / "decision_rules.json").read_text(encoding="utf-8"))
    labels = {
        "default_threshold": "Threshold 0.5",
        "cost_optimal_threshold": "Cost-optimal threshold",
        "expected_cost": "Expected-cost rule",
    }
    lines = [
        "# Cost-based decision rules",
        "",
        "Generated by `scripts/run_cost_analysis.py`; do not edit.",
        "",
        f"{result['n_validation_transactions']:,} validation transactions, "
        f"{result['n_validation_frauds']} frauds, fraudulent amounts totalling "
        f"{result['no_detection_cost']:,.2f}. Mean over {len(result['seeds'])} seeds.",
        "",
        "| Alert cost | Rule | Savings | Total cost | Alerts per 100,000 | Precision | Recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in result["alert_costs"].values():
        for rule, report in entry["rules"].items():
            lines.append(
                f"| {entry['alert_cost']:g} | {labels[rule]} | {report['savings']:.1%} "
                f"| {report['cost']:,.0f} | {report['alerts_per_100k']:.0f} "
                f"| {report['precision']:.3f} | {report['recall']:.3f} |"
            )
        lines.append(
            f"| {entry['alert_cost']:g} | Oracle (upper bound) | {entry['oracle']['savings']:.1%} "
            f"| {entry['oracle']['cost']:,.0f} | | | |"
        )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written to {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
