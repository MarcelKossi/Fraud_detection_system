"""Draw the figures of the README from the files in ``results/``.

No model is fitted and no data file is read: every figure is a rendering of numbers
already stored in ``results/``. Figures are written to ``images/``.

Usage::

    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fraud_detection.config import load_config  # noqa: E402

# Categorical colours in fixed order; each series is also labelled directly, since the
# third and fourth have a low contrast against the background.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

MODEL_STYLE = {
    "random_forest": ("Random forest (first version)", BLUE),
    "logistic_regression": ("Logistic regression", ORANGE),
    "lightgbm": ("LightGBM", AQUA),
    "xgboost": ("XGBoost", YELLOW),
}
RULE_STYLE = {
    "default_threshold": ("Threshold 0.5", BLUE),
    "cost_optimal_threshold": ("Cost-optimal threshold", ORANGE),
    "expected_cost": ("Expected-cost rule", AQUA),
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
        }
    )


def _load(results: Path, relative: str) -> dict:
    return json.loads((results / relative).read_text(encoding="utf-8"))


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_dir / name}")


def figure_evaluation(results: Path, out_dir: Path) -> None:
    """AUPRC and its interval under each protocol, for the first version and the model."""
    replication = _load(results, "baseline/v1_replication.json")["test"]
    final = _load(results, "final/holdout_evaluation.json")["main_dataset"]
    comparison = final["comparison_with_first_version"]["retained_model_expected_cost"]
    rows = [
        (
            "First version\nrandom split, duplicates kept",
            replication["auprc"],
            replication["auprc_ci"],
            replication["n_positive"],
            MUTED,
        ),
        (
            "First version\ntemporal hold-out",
            comparison["old"]["auprc"]["value"],
            comparison["old"]["auprc"]["ci"],
            final["n_holdout_frauds"],
            BLUE,
        ),
        (
            "Retained model\ntemporal hold-out",
            comparison["new"]["auprc"]["value"],
            comparison["new"]["auprc"]["ci"],
            final["n_holdout_frauds"],
            ORANGE,
        ),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    for position, (_label, value, (low, high), n_frauds, colour) in enumerate(rows):
        y = len(rows) - 1 - position
        ax.plot([low, high], [y, y], color=colour, linewidth=2, solid_capstyle="round")
        ax.plot(value, y, "o", color=colour, markersize=8, markeredgecolor=SURFACE)
        ax.annotate(
            f"{value:.3f}  [{low:.3f}, {high:.3f}]  ·  {n_frauds} frauds in test",
            (high, y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=MUTED,
            fontsize=9,
        )
    ax.axhline(1.5, color=GRID, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_yticks(range(len(rows)), [row[0] for row in reversed(rows)])
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0.6, 1.12)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Area under the precision-recall curve (AUPRC), with 95% bootstrap interval")
    ax.set_title("Effect of the evaluation protocol, and final comparison on the hold-out set")
    _save(fig, out_dir, "evaluation.png")


def figure_model_selection(results: Path, out_dir: Path) -> None:
    """Cross-validated AUPRC of each model family in each validation period."""
    families = _load(results, "model_selection/imbalance.json")["families"]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for family, (label, colour) in MODEL_STYLE.items():
        per_fold = families[family]["strategies"]["none"]["auprc_per_fold"]
        periods = range(1, len(per_fold) + 1)
        ax.plot(
            periods,
            per_fold,
            color=colour,
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            label=label,
        )
    ax.set_xticks(list(periods), [f"Period {i}" for i in periods])
    ax.set_xlim(0.8, len(per_fold) + 0.2)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("AUPRC, mean over 5 seeds")
    ax.set_title("Cross-validated AUPRC by validation period: periods differ more than models")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    _save(fig, out_dir, "model_selection.png")


def figure_cost_rules(results: Path, out_dir: Path) -> None:
    """Savings of each decision rule, in cross-validation and on the hold-out set."""
    cross_validation = _load(results, "cost/decision_rules.json")["alert_costs"]
    final = _load(results, "final/holdout_evaluation.json")["main_dataset"]["savings_by_alert_cost"]
    holdout_key = {
        "default_threshold": "retained_model_threshold",
        "expected_cost": "retained_model_expected_cost",
    }
    panels = [
        (
            "Cross-validation (development set)",
            {
                rule: [entry["rules"][rule]["savings"] for entry in cross_validation.values()]
                for rule in RULE_STYLE
            },
        ),
        (
            "Hold-out set",
            {rule: [final[cost][key] for cost in final] for rule, key in holdout_key.items()},
        ),
    ]
    costs = list(cross_validation)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), sharey=True)
    width = 0.25
    for ax, (title, series) in zip(axes, panels, strict=True):
        offsets = [(i - (len(series) - 1) / 2) * (width + 0.03) for i in range(len(series))]
        for offset, (rule, values) in zip(offsets, series.items(), strict=True):
            label, colour = RULE_STYLE[rule]
            positions = [i + offset for i in range(len(costs))]
            ax.bar(positions, [100 * v for v in values], width=width, color=colour, label=label)
            for x, value in zip(positions, values, strict=True):
                ax.annotate(
                    f"{100 * value:.0f}",
                    (x, 100 * value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=MUTED,
                )
        ax.set_xticks(range(len(costs)), [f"Alert cost {cost}" for cost in costs])
        ax.tick_params(axis="x", length=0, labelcolor=INK)
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
        ax.set_ylim(0, 85)
    axes[0].set_ylabel("Savings (% of fraudulent amount)")
    handles, names = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, names, loc="upper right", fontsize=8.5)
    fig.suptitle(
        "Savings by decision rule: the advantage seen in cross-validation "
        "is not confirmed on the hold-out set",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, out_dir, "cost_rules.png")


def figure_behavioural(results: Path, out_dir: Path) -> None:
    """Share of frauds flagged, by fraud scenario and feature set, on simulated data."""
    result = _load(results, "behavioural/ablation.json")
    labels = {
        "raw": "Amount and time of day",
        "raw_calendar": "+ calendar flags",
        "raw_customer": "+ customer aggregates",
        "raw_terminal": "+ terminal aggregates",
        "all": "All groups",
    }
    feature_sets = result["feature_sets"]
    scenarios = list(next(iter(feature_sets.values()))["recall_by_scenario"])
    names = list(feature_sets)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(10.4, 2.9), sharey=True)
    for ax, scenario in zip(axes, scenarios, strict=True):
        values = [100 * feature_sets[name]["recall_by_scenario"][scenario] for name in names]
        positions = range(len(names) - 1, -1, -1)
        ax.barh(list(positions), values, height=0.62, color=BLUE)
        for y, value in zip(positions, values, strict=True):
            ax.annotate(
                f"{value:.0f}%",
                (value, y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=MUTED,
            )
        ax.set_xlim(0, 115)
        ax.set_xticks([0, 50, 100])
        ax.grid(axis="y", visible=False)
        ax.set_title(scenario.capitalize(), fontsize=10)
        ax.set_yticks(list(positions), [labels.get(name, name) for name in names])
        ax.tick_params(axis="y", length=0, labelcolor=INK)
    max_fpr = result["metadata"]["config"]["evaluation"]["max_fpr"]
    fig.supxlabel(
        f"Frauds flagged at a false-positive rate of at most {max_fpr:.1%} (%)",
        fontsize=9.5,
        color=MUTED,
    )
    fig.suptitle(
        "Simulated transactions: which features reveal which fraud scenario",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, out_dir, "behavioural.png")


def figure_attributions(results: Path, out_dir: Path) -> None:
    """Mean absolute contribution of the leading features, overall and per period."""
    result = _load(results, "explanations/global_importance.json")
    entries = result["features"][: result["top_k"]]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    positions = range(len(entries) - 1, -1, -1)
    ax.barh(
        list(positions),
        [entry["mean_abs_contribution"] for entry in entries],
        height=0.62,
        color=BLUE,
        label="Mean over validation periods",
    )
    for index, (y, entry) in enumerate(zip(positions, entries, strict=True)):
        ax.plot(
            entry["per_fold"],
            [y] * len(entry["per_fold"]),
            "o",
            color=INK,
            markersize=4.5,
            markeredgecolor=SURFACE,
            markeredgewidth=0.8,
            linestyle="none",
            label="Each validation period" if index == 0 else None,
        )
    ax.set_yticks(list(positions), [entry["feature"] for entry in entries])
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean absolute Shapley contribution (log-odds)")
    ax.set_title("Leading features of the retained model and their stability across periods")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, out_dir, "attributions.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out", default="images")
    args = parser.parse_args()

    results = Path(load_config(args.config).results_dir)
    out_dir = Path(args.out)
    _style()
    print("Figures written:")
    figure_evaluation(results, out_dir)
    figure_model_selection(results, out_dir)
    figure_cost_rules(results, out_dir)
    figure_behavioural(results, out_dir)
    figure_attributions(results, out_dir)


if __name__ == "__main__":
    main()
