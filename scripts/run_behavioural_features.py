"""Ablation of behavioural feature groups on simulated transactions.

A transaction table with customer and terminal identifiers is simulated, behavioural
features are derived from it, and the retained model is cross-validated on the
development period for each feature set of the configuration. Each training window ends
``delay_days`` before its validation block, as labels are assumed to be known with that
delay. The hold-out period is not read.

Results are written to ``results/behavioural/``.

Usage::

    python scripts/run_behavioural_features.py
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
from fraud_detection.config import load_config
from fraud_detection.experiment import run_metadata, write_result
from fraud_detection.metrics import auprc, recall_at_fpr, threshold_at_fpr
from fraud_detection.models import build_model
from fraud_detection.simulation import simulate_transactions, table_digest
from fraud_detection.splits import assert_disjoint, expanding_time_folds, time_holdout

SCENARIOS = {1: "amount above threshold", 2: "compromised terminal", 3: "compromised customer"}


def describe(table) -> dict:
    return {
        "n_transactions": int(len(table)),
        "n_frauds": int(table["is_fraud"].sum()),
        "fraud_rate": float(table["is_fraud"].mean()),
        "frauds_by_scenario": {
            SCENARIOS[k]: int((table["fraud_scenario"] == k).sum()) for k in SCENARIOS
        },
        "first_datetime": str(table["datetime"].min()),
        "last_datetime": str(table["datetime"].max()),
    }


def evaluate_fold(model, X_val, y_val, scenario_val, max_fpr: float) -> dict:
    scores = model.predict_proba(X_val)[:, 1]
    flagged = scores >= threshold_at_fpr(y_val, scores, max_fpr)
    return {
        "auprc": auprc(y_val, scores),
        "recall_at_fpr": recall_at_fpr(y_val, scores, max_fpr),
        "recall_by_scenario": {
            name: float(flagged[scenario_val == k].mean()) for k, name in SCENARIOS.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config.behavioural
    selected = config.selected_model
    max_fpr = config.evaluation.max_fpr

    print("Simulating transactions ...", flush=True)
    transactions = simulate_transactions(config.simulation)
    digest = table_digest(transactions)
    table = add_behavioural_features(transactions, settings.windows_days, settings.delay_days)

    times = table["time_seconds"].to_numpy()
    dev_idx, holdout_idx = time_holdout(times, settings.test_fraction)
    assert_disjoint(dev_idx, holdout_idx)
    development = table.iloc[dev_idx]
    dev_times = development["time_seconds"].to_numpy()
    gap = settings.delay_days * SECONDS_PER_DAY
    folds = expanding_time_folds(dev_times, settings.n_cv_folds, gap)
    for train_idx, val_idx in folds:
        assert_disjoint(train_idx, val_idx)
        if dev_times[train_idx].max() + gap > dev_times[val_idx].min():
            raise AssertionError(
                "A training window ends less than the label delay before validation"
            )

    y = development["is_fraud"].to_numpy()
    scenario = development["fraud_scenario"].to_numpy()

    feature_sets = {}
    for name, groups in settings.feature_sets.items():
        columns = feature_columns(groups, settings.windows_days)
        X = development[columns]
        runs = []
        for seed in settings.seeds:
            fold_reports = []
            for train_idx, val_idx in folds:
                model = build_model(selected.family, selected.params, seed)
                model.fit(X.iloc[train_idx], y[train_idx])
                fold_reports.append(
                    evaluate_fold(model, X.iloc[val_idx], y[val_idx], scenario[val_idx], max_fpr)
                )
            runs.append({"seed": seed, "folds": fold_reports})
            mean = np.mean([fold["auprc"] for fold in fold_reports])
            print(f"  {name} seed {seed}: AUPRC {mean:.4f}", flush=True)

        auprc_matrix = np.array([[fold["auprc"] for fold in run["folds"]] for run in runs])
        recall_matrix = np.array([[fold["recall_at_fpr"] for fold in run["folds"]] for run in runs])
        feature_sets[name] = {
            "groups": groups,
            "n_features": len(columns),
            "auprc_mean": float(auprc_matrix.mean()),
            "auprc_seed_std": float(auprc_matrix.mean(axis=1).std(ddof=1)),
            "auprc_per_fold": auprc_matrix.mean(axis=0).tolist(),
            "recall_at_fpr_mean": float(recall_matrix.mean()),
            "recall_by_scenario": {
                scenario_name: float(
                    np.mean(
                        [
                            fold["recall_by_scenario"][scenario_name]
                            for run in runs
                            for fold in run["folds"]
                        ]
                    )
                )
                for scenario_name in SCENARIOS.values()
            },
            "runs": runs,
        }

    payload = {
        "experiment": "behavioural_feature_ablation",
        "simulated_table_sha256": digest,
        "dataset": describe(table),
        "development": describe(development),
        "holdout": {"n_transactions": int(len(holdout_idx)), "evaluated": False},
        "folds": [
            {
                "n_train": int(len(train_idx)),
                "n_train_frauds": int(y[train_idx].sum()),
                "n_validation": int(len(val_idx)),
                "n_validation_frauds": int(y[val_idx].sum()),
            }
            for train_idx, val_idx in folds
        ],
        "model": {"family": selected.family, "params": selected.params},
        "feature_sets": feature_sets,
        "metadata": run_metadata(config.to_dict(), digest),
    }
    out_dir = Path(config.results_dir) / "behavioural"
    print(f"Written to {write_result(out_dir / 'ablation.json', payload)}")
    write_summary(out_dir)


def write_summary(out_dir: Path) -> None:
    """Regenerate ``summary.md`` from ``ablation.json``."""
    result = json.loads((out_dir / "ablation.json").read_text(encoding="utf-8"))
    dataset = result["dataset"]
    max_fpr = result["metadata"]["config"]["evaluation"]["max_fpr"]
    scenario_names = list(SCENARIOS.values())
    lines = [
        "# Behavioural features on simulated transactions",
        "",
        "Generated by `scripts/run_behavioural_features.py`; do not edit.",
        "",
        f"Simulated table: {dataset['n_transactions']:,} transactions, "
        f"{dataset['n_frauds']:,} frauds ({dataset['fraud_rate']:.2%}); frauds by scenario: "
        + ", ".join(f"{name} {count:,}" for name, count in dataset["frauds_by_scenario"].items())
        + ".",
        "",
        "Cross-validated on the development period; the hold-out period is not used.",
        "",
        "| Feature set | Features | AUPRC mean (std over seeds) | AUPRC per fold "
        f"| Recall at FPR ≤ {max_fpr:.1%} | "
        + " | ".join(f"Recall, {name}" for name in scenario_names)
        + " |",
        "|---|---|---|---|---|" + "---|" * len(scenario_names),
    ]
    for name, summary in result["feature_sets"].items():
        per_fold = ", ".join(f"{value:.3f}" for value in summary["auprc_per_fold"])
        by_scenario = " | ".join(
            f"{summary['recall_by_scenario'][scenario]:.3f}" for scenario in scenario_names
        )
        lines.append(
            f"| `{name}` | {summary['n_features']} | {summary['auprc_mean']:.3f} "
            f"({summary['auprc_seed_std']:.3f}) | {per_fold} "
            f"| {summary['recall_at_fpr_mean']:.3f} | {by_scenario} |"
        )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Summary written to {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
