"""Regenerate the result tables of the documents from the files in ``results/``.

Every generated table of the README, the model card and the report lies between two
markers, ``<!-- BEGIN:<name> -->`` and
``<!-- END:<name> -->``. This script rewrites what lies between them, so that no figure
of the README is typed by hand.

Usage::

    python scripts/update_readme.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from fraud_detection.config import load_config

FAMILY_LABELS = {
    "random_forest": "Random forest, default settings",
    "logistic_regression": "Logistic regression",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}
TREATMENT_LABELS = {
    "none": "None",
    "class_weight": "Class weighting",
    "undersampling": "Random under-sampling",
    "smote": "SMOTE",
}
CALIBRATION_LABELS = {"none": "None", "sigmoid": "Sigmoid (Platt)", "isotonic": "Isotonic"}


def _load(results: Path, relative: str) -> dict:
    return json.loads((results / relative).read_text(encoding="utf-8"))


def protocol_table(results: Path) -> str:
    rows = [
        "| Evaluation protocol | Frauds in test | Precision | Recall | F1 | AUPRC [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for name, label in (
        ("v1_replication", "Random stratified split, duplicates kept (first version)"),
        ("v1_temporal_holdout", "Temporal hold-out, duplicates removed"),
    ):
        test = _load(results, f"baseline/{name}.json")["test"]
        at = test["at_threshold"]
        low, high = test["auprc_ci"]
        rows.append(
            f"| {label} | {test['n_positive']} | {at['precision']:.3f} | {at['recall']:.3f} "
            f"| {at['f1']:.3f} | {test['auprc']:.3f} [{low:.3f}, {high:.3f}] |"
        )
    return "\n".join(rows)


def model_table(results: Path) -> str:
    imbalance = _load(results, "model_selection/imbalance.json")
    max_fpr = imbalance["metadata"]["config"]["evaluation"]["max_fpr"]
    rows = [
        f"| Model | AUPRC, mean over seeds (seed std) | AUPRC per validation period "
        f"| Recall at FPR ≤ {max_fpr:.1%} |",
        "|---|---|---|---|",
    ]
    for family, entry in imbalance["families"].items():
        summary = entry["strategies"]["none"]
        per_fold = ", ".join(f"{value:.3f}" for value in summary["auprc_per_fold"])
        rows.append(
            f"| {FAMILY_LABELS[family]} | {summary['auprc_mean']:.3f} "
            f"({summary['auprc_seed_std']:.3f}) | {per_fold} "
            f"| {summary['recall_at_fpr_mean']:.3f} |"
        )
    return "\n".join(rows)


def imbalance_table(results: Path, family: str) -> str:
    strategies = _load(results, "model_selection/imbalance.json")["families"][family]["strategies"]
    rows = [
        "| Imbalance treatment | AUPRC, mean over seeds (seed std) | Recall at low FPR |",
        "|---|---|---|",
    ]
    for strategy, summary in strategies.items():
        rows.append(
            f"| {TREATMENT_LABELS[strategy]} | {summary['auprc_mean']:.3f} "
            f"({summary['auprc_seed_std']:.3f}) | {summary['recall_at_fpr_mean']:.3f} |"
        )
    return "\n".join(rows)


def calibration_table(results: Path, family: str) -> str:
    methods = _load(results, "model_selection/calibration.json")["families"][family]["methods"]
    rows = ["| Calibration | Brier score | Log loss | AUPRC |", "|---|---|---|---|"]
    for method, summary in methods.items():
        rows.append(
            f"| {CALIBRATION_LABELS[method]} | {summary['brier_mean']:.6f} "
            f"| {summary['log_loss_mean']:.5f} | {summary['auprc_mean']:.3f} |"
        )
    return "\n".join(rows)


def behavioural_table(results: Path) -> str:
    result = _load(results, "behavioural/ablation.json")
    max_fpr = result["metadata"]["config"]["evaluation"]["max_fpr"]
    labels = {
        "raw": "Amount and time of day",
        "raw_calendar": "Amount, time of day and calendar flags",
        "raw_customer": "Amount, time of day and customer aggregates",
        "raw_terminal": "Amount, time of day and terminal aggregates",
        "all": "All groups",
    }
    scenarios = list(next(iter(result["feature_sets"].values()))["recall_by_scenario"])
    rows = [
        "| Feature set | Features | AUPRC, mean over seeds (seed std) "
        f"| Recall at FPR ≤ {max_fpr:.1%} | "
        + " | ".join(f"Recall, {name}" for name in scenarios)
        + " |",
        "|---|---|---|---|" + "---|" * len(scenarios),
    ]
    for name, summary in result["feature_sets"].items():
        by_scenario = " | ".join(f"{summary['recall_by_scenario'][s]:.3f}" for s in scenarios)
        rows.append(
            f"| {labels.get(name, name)} | {summary['n_features']} "
            f"| {summary['auprc_mean']:.3f} ({summary['auprc_seed_std']:.3f}) "
            f"| {summary['recall_at_fpr_mean']:.3f} | {by_scenario} |"
        )
    return "\n".join(rows)


def cost_table(results: Path, reference_cost: float) -> str:
    result = _load(results, "cost/decision_rules.json")
    entries = result["alert_costs"]
    reference = entries[f"{reference_cost:g}"]
    labels = {
        "default_threshold": "Probability threshold of 0.5 (first version)",
        "cost_optimal_threshold": "Cost-optimal probability threshold",
        "expected_cost": "Expected-cost rule (probability x amount > alert cost)",
    }
    rows = [
        "| Decision rule | "
        + " | ".join(f"Savings, alert cost {entry['alert_cost']:g}" for entry in entries.values())
        + f" | Alerts per 100,000 (alert cost {reference_cost:g}) | Precision | Recall |",
        "|---|" + "---|" * len(entries) + "---|---|---|",
    ]
    for rule, label in labels.items():
        report = reference["rules"][rule]
        rows.append(
            f"| {label} | "
            + " | ".join(f"{entry['rules'][rule]['savings']:.1%}" for entry in entries.values())
            + f" | {report['alerts_per_100k']:.0f} | {report['precision']:.3f} "
            f"| {report['recall']:.3f} |"
        )
    rows.append(
        "| Oracle (upper bound) | "
        + " | ".join(f"{entry['oracle']['savings']:.1%}" for entry in entries.values())
        + " | | | |"
    )
    return "\n".join(rows)


def attribution_table(results: Path) -> str:
    result = _load(results, "explanations/global_importance.json")
    rows = [
        "| Rank | Feature | Mean absolute contribution (log-odds) | Rank in each period |",
        "|---|---|---|---|",
    ]
    for rank, entry in enumerate(result["features"][: result["top_k"]], start=1):
        rows.append(
            f"| {rank} | `{entry['feature']}` | {entry['mean_abs_contribution']:.3f} "
            f"| {', '.join(str(r) for r in entry['rank_per_fold'])} |"
        )
    return "\n".join(rows)


FINAL_LABELS = {
    "first_version": "First version: random forest, threshold 0.5",
    "retained_model_threshold": "Retained model, threshold 0.5",
    "retained_model_expected_cost": "Retained model, expected-cost rule",
}


def _ci(entry: dict, fmt: str = ".3f", signed: bool = False) -> str:
    sign = "+" if signed else ""
    low, high = entry["ci"]
    return f"{entry['value']:{sign}{fmt}} [{low:{sign}{fmt}}, {high:{sign}{fmt}}]"


def final_table(results: Path) -> str:
    main = _load(results, "final/holdout_evaluation.json")["main_dataset"]
    comparison = main["comparison_with_first_version"]
    measured = {
        "first_version": comparison["retained_model_expected_cost"]["old"],
        "retained_model_threshold": comparison["retained_model_threshold"]["new"],
        "retained_model_expected_cost": comparison["retained_model_expected_cost"]["new"],
    }
    max_fpr = main["systems"]["first_version"]["ranking"]["recall_at_fpr"]["max_fpr"]
    rows = [
        f"| System | AUPRC | Recall at FPR ≤ {max_fpr:.1%} "
        f"| Savings, alert cost {main['alert_cost']:g} | Alerts | Frauds flagged |",
        "|---|---|---|---|---|---|",
    ]
    for name, label in FINAL_LABELS.items():
        decision = main["systems"][name]["decision"]
        rows.append(
            f"| {label} | {_ci(measured[name]['auprc'])} "
            f"| {_ci(measured[name]['recall_at_fpr'])} | {_ci(measured[name]['savings'], '.1%')} "
            f"| {decision['n_alerts']} | {decision['recall']:.1%} |"
        )
    difference = comparison["retained_model_expected_cost"]["difference"]
    share = comparison["retained_model_expected_cost"]["share_of_resamples_where_new_is_better"]
    rows.append(
        f"| Difference, last row minus first | {_ci(difference['auprc'], signed=True)} "
        f"| {_ci(difference['recall_at_fpr'], signed=True)} "
        f"| {_ci(difference['savings'], '.1%', signed=True)} | | |"
    )
    rows.append(
        f"| Share of bootstrap resamples in which the last row is strictly better "
        f"| {share['auprc']:.0%} | {share['recall_at_fpr']:.0%} (ties counted as not better) "
        f"| {share['savings']:.0%} | | |"
    )
    return "\n".join(rows)


def final_sensitivity_table(results: Path) -> str:
    main = _load(results, "final/holdout_evaluation.json")["main_dataset"]
    costs = main["savings_by_alert_cost"]
    rows = [
        "| System | " + " | ".join(f"Savings, alert cost {cost}" for cost in costs) + " |",
        "|---|" + "---|" * len(costs),
    ]
    for name, label in {**FINAL_LABELS, "oracle": "Oracle (upper bound)"}.items():
        rows.append(
            f"| {label} | " + " | ".join(f"{costs[cost][name]:.1%}" for cost in costs) + " |"
        )
    return "\n".join(rows)


def final_simulated_table(results: Path) -> str:
    simulated = _load(results, "final/holdout_evaluation.json")["simulated_dataset"]
    labels = {"raw": "Amount and time of day", "all": "All behavioural groups"}
    rows = [
        "| Feature set | Features | AUPRC, mean over seeds (seed std) | Recall at low FPR |",
        "|---|---|---|---|",
    ]
    for name, summary in simulated["feature_sets"].items():
        rows.append(
            f"| {labels[name]} | {summary['n_features']} | {summary['auprc_mean']:.3f} "
            f"({summary['auprc_seed_std']:.3f}) | {summary['recall_at_fpr_mean']:.3f} |"
        )
    return "\n".join(rows)


def cost_interval_table(results: Path) -> str:
    entries = _load(results, "cost/decision_rules.json")["alert_costs"]
    rows = [
        "| Alert cost | Savings of the expected-cost rule minus savings of the threshold "
        "of 0.5 | 95% paired bootstrap interval | Share of resamples above zero |",
        "|---|---|---|---|",
    ]
    for entry in entries.values():
        d = entry["expected_cost_minus_default_threshold"]
        rows.append(
            f"| {entry['alert_cost']:g} | {100 * d['value']:+.1f} points "
            f"| [{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}] | {d['share_positive']:.0%} |"
        )
    return "\n".join(rows)


def reliability_block(results: Path) -> str:
    rows = [
        "| Predicted probability | Predictions | Frauds | Mean predicted | Observed rate |",
        "|---|---|---|---|---|",
    ]
    for row in _load(results, "cost/decision_rules.json")["reliability"]:
        low, high = row["bin"]
        rows.append(
            f"| {low:g} to {high:g} | {row['n']:,} | {row['n_frauds']:,} "
            f"| {row['mean_predicted']:.4f} | {row['observed_rate']:.4f} |"
        )
    return "\n".join(rows)


def model_comparison_block(results: Path) -> str:
    r = _load(results, "robustness/model_comparison.json")
    d = r["difference"]
    per_period = ", ".join(f"{v:+.3f}" for v in r["difference_per_period"])
    wins = ", ".join(
        f"{w} of {r['seed_pairs_per_period']}" for w in r["wins_over_seed_pairs_per_period"]
    )
    return "\n".join(
        [
            "| Quantity | Value |",
            "|---|---|",
            f"| Mean AUPRC, retained model | {r['retained']['auprc']:.3f} |",
            f"| Mean AUPRC, random forest | {r['reference']['auprc']:.3f} |",
            f"| Difference, with 95% paired bootstrap interval | {d['value']:+.3f} "
            f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] |",
            f"| Share of resamples in which the difference is positive "
            f"| {d['share_positive']:.0%} |",
            f"| Difference in each validation period | {per_period} |",
            f"| Pairs of seeds won by the retained model, in each period | {wins} |",
        ]
    )


def decomposition_block(results: Path) -> str:
    r = _load(results, "robustness/protocol_decomposition.json")
    cv = _load(results, "baseline/cv_development.json")["feature_sets"]["raw_time"]
    labels = {
        "raw_file": "Random splits, duplicates kept (protocol of the first version)",
        "deduplicated_file": "Random splits, duplicates removed",
    }
    rows = [
        "| Protocol | AUPRC, mean (std over splits) | Recall at 0.5, mean (std) "
        "| Test frauds also in training, mean |",
        "|---|---|---|---|",
    ]
    for name, v in r["variants"].items():
        shared = sum(run["test_frauds_also_in_train"] for run in v["runs"]) / len(v["runs"])
        rows.append(
            f"| {labels[name]} | {v['auprc_mean']:.3f} ({v['auprc_std']:.3f}) "
            f"| {v['recall_mean']:.3f} ({v['recall_std']:.3f}) | {shared:.1f} |"
        )
    recalls = [fold["at_threshold"]["recall"] for fold in cv["folds"]]
    mean_recall = sum(recalls) / len(recalls)
    std_recall = (sum((x - mean_recall) ** 2 for x in recalls) / (len(recalls) - 1)) ** 0.5
    rows.append(
        f"| Temporal folds, duplicates removed | {cv['auprc_mean']:.3f} "
        f"({cv['auprc_std']:.3f}) | {mean_recall:.3f} ({std_recall:.3f}) | 0 |"
    )
    return "\n".join(rows)


def replace_block(text: str, name: str, content: str) -> str:
    pattern = re.compile(rf"(<!-- BEGIN:{name} -->\n).*?(<!-- END:{name} -->)", flags=re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"Markers for block {name!r} not found in the README")
    return pattern.sub(lambda match: match.group(1) + content + "\n" + match.group(2), text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--documents",
        nargs="+",
        default=["README.md", "docs/MODEL_CARD.md", "docs/REPORT.md"],
        help="Files whose marked blocks are regenerated.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results = Path(config.results_dir)
    family = config.selected_model.family
    blocks = {
        "protocol": lambda: protocol_table(results),
        "models": lambda: model_table(results),
        "imbalance": lambda: imbalance_table(results, family),
        "calibration": lambda: calibration_table(results, family),
        "behavioural": lambda: behavioural_table(results),
        "costs": lambda: cost_table(results, config.cost.alert_cost),
        "attributions": lambda: attribution_table(results),
        "cost_intervals": lambda: cost_interval_table(results),
        "reliability": lambda: reliability_block(results),
        "model_comparison": lambda: model_comparison_block(results),
        "decomposition": lambda: decomposition_block(results),
        "final": lambda: final_table(results),
        "final_sensitivity": lambda: final_sensitivity_table(results),
        "final_simulated": lambda: final_simulated_table(results),
    }
    for document in args.documents:
        path = Path(document)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        present = [name for name in blocks if f"<!-- BEGIN:{name} -->" in text]
        for name in present:
            text = replace_block(text, name, blocks[name]())
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"{path}: {len(present)} blocks regenerated from {results}/")


if __name__ == "__main__":
    main()
