"""Model selection by expanding-window cross-validation on the development period.

The hold-out set is never read by this script. Three stages, each written to
``results/model_selection/``:

``search``
    Random search of hyper-parameters for each model family, without any imbalance
    treatment.
``imbalance``
    For the best configuration of each family, comparison of imbalance treatments,
    repeated over several seeds. The random forest of the first version is included as
    a reference.
``calibration``
    For each family with its best imbalance treatment, comparison of probability
    calibration methods.

Usage::

    python scripts/run_model_selection.py                         # all stages
    python scripts/run_model_selection.py --stage search --family xgboost
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fraud_detection.calibration import METHODS
from fraud_detection.config import Config, load_config
from fraud_detection.data import load_raw, prepare, sha256_of_file
from fraud_detection.experiment import run_metadata, write_result
from fraud_detection.features import build_features
from fraud_detection.sampling import STRATEGIES
from fraud_detection.search import SEARCH_SPACES, sample_configurations
from fraud_detection.splits import (
    assert_chronological,
    assert_disjoint,
    expanding_window_folds,
    temporal_holdout,
)
from fraud_detection.validation import cross_validate, cross_validate_calibration

STAGES = ("search", "imbalance", "calibration")
SEARCHED_FAMILIES = tuple(SEARCH_SPACES)
REFERENCE = "random_forest"


def load_development(config: Config):
    """Features, target and folds of the development period; the hold-out set is dropped."""
    prepared = prepare(load_raw(config.data.raw_path, config.data.sha256), config.data.time_column)
    dev_idx, _ = temporal_holdout(len(prepared), config.split.test_fraction)
    development = prepared.iloc[dev_idx]
    folds = expanding_window_folds(len(development), config.split.n_cv_folds)
    times = development[config.data.time_column].to_numpy()
    for train_idx, val_idx in folds:
        assert_disjoint(train_idx, val_idx)
        assert_chronological(times, train_idx, val_idx)
    X, y = build_features(
        development,
        config.data.target,
        config.data.time_column,
        config.model_selection.feature_set,
    )
    return X, y, folds


def read(out_dir: Path, name: str) -> dict:
    return json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))


def run_search(config: Config, X, y, folds, family: str) -> dict:
    n_trials = config.model_selection.n_trials[family]
    trials = []
    for index, params in enumerate(sample_configurations(family, n_trials, config.seed)):
        start = time.perf_counter()
        scores = cross_validate(
            family, params, X, y, folds, seed=config.seed, max_fpr=config.evaluation.max_fpr
        )
        trials.append(
            {"trial": index, "params": params, **scores, "seconds": time.perf_counter() - start}
        )
        print(f"  {family} trial {index + 1}/{n_trials}: AUPRC {scores['auprc_mean']:.4f}")
    trials.sort(key=lambda trial: trial["auprc_mean"], reverse=True)
    return {"family": family, "n_trials": n_trials, "best": trials[0], "trials": trials}


def _aggregate(runs: list[dict]) -> dict:
    """Summarise the runs of one configuration over several seeds."""
    by_seed = np.array([run["auprc_mean"] for run in runs])
    fold_matrix = np.array([run["fold_auprc"] for run in runs])
    recall = np.array([run["recall_at_fpr_mean"] for run in runs])
    return {
        "auprc_mean": float(by_seed.mean()),
        "auprc_seed_std": float(by_seed.std(ddof=1)) if len(by_seed) > 1 else 0.0,
        "auprc_per_fold": fold_matrix.mean(axis=0).tolist(),
        "recall_at_fpr_mean": float(recall.mean()),
        "runs": runs,
    }


def retained_strategy(strategies: dict) -> str:
    """Select an imbalance treatment, favouring the absence of treatment.

    A treatment replaces ``none`` only if its mean AUPRC exceeds that of ``none`` by more
    than twice the larger of their standard deviations over seeds. A smaller difference
    is not distinguishable from run-to-run variability and does not justify the added
    complexity nor the distortion of the predicted probabilities.
    """
    best = max(strategies, key=lambda name: strategies[name]["auprc_mean"])
    gain = strategies[best]["auprc_mean"] - strategies["none"]["auprc_mean"]
    noise = max(strategies[best]["auprc_seed_std"], strategies["none"]["auprc_seed_std"])
    return best if gain > 2.0 * noise else "none"


def run_imbalance(config: Config, X, y, folds, out_dir: Path) -> dict:
    selection = config.model_selection
    common = {"max_fpr": config.evaluation.max_fpr, "ratio": selection.resampling_ratio}

    reference_params = {"n_estimators": config.baseline_model.n_estimators}
    print(f"  {REFERENCE} (reference)")
    families = {
        REFERENCE: {
            "params": reference_params,
            "strategies": {
                "none": _aggregate(
                    [
                        {
                            "seed": seed,
                            **cross_validate(
                                REFERENCE, reference_params, X, y, folds, seed=seed, **common
                            ),
                        }
                        for seed in selection.seeds
                    ]
                )
            },
        }
    }
    for family in SEARCHED_FAMILIES:
        params = read(out_dir, f"search_{family}")["best"]["params"]
        strategies = {}
        for strategy in STRATEGIES:
            print(f"  {family} / {strategy}")
            strategies[strategy] = _aggregate(
                [
                    {
                        "seed": seed,
                        **cross_validate(
                            family, params, X, y, folds, seed=seed, strategy=strategy, **common
                        ),
                    }
                    for seed in selection.seeds
                ]
            )
        families[family] = {
            "params": params,
            "best_strategy": retained_strategy(strategies),
            "strategies": strategies,
        }
    return {"seeds": selection.seeds, "families": families}


def run_calibration(config: Config, X, y, folds, out_dir: Path) -> dict:
    selection = config.model_selection
    imbalance = read(out_dir, "imbalance")["families"]
    families = {}
    for family in SEARCHED_FAMILIES:
        params, strategy = imbalance[family]["params"], imbalance[family]["best_strategy"]
        print(f"  {family} / {strategy}")
        collected = {method: {"brier": [], "log_loss": [], "auprc": []} for method in METHODS}
        for seed in selection.seeds:
            result = cross_validate_calibration(
                family,
                params,
                X,
                y,
                folds,
                seed=seed,
                strategy=strategy,
                ratio=selection.resampling_ratio,
                calibration_fraction=selection.calibration_fraction,
            )
            for method in METHODS:
                for name, values in result[method].items():
                    collected[method][name].append(values)
        families[family] = {
            "strategy": strategy,
            "methods": {
                method: {
                    f"{name}_mean": float(np.mean(values))
                    for name, values in collected[method].items()
                }
                | {
                    f"{name}_per_fold": np.mean(values, axis=0).tolist()
                    for name, values in collected[method].items()
                }
                for method in METHODS
            },
        }
    return {"seeds": selection.seeds, "families": families}


def write_summary(out_dir: Path) -> None:
    """Regenerate ``summary.md`` from the JSON files present in ``out_dir``."""
    lines = [
        "# Model selection results",
        "",
        "Generated by `scripts/run_model_selection.py`; do not edit.",
        "All figures are cross-validated on the development period; the hold-out set is not used.",
        "",
    ]

    searches = [f for f in SEARCHED_FAMILIES if (out_dir / f"search_{f}.json").exists()]
    if searches:
        lines += [
            "## Hyper-parameter search (no imbalance treatment, single seed)",
            "",
            "| Family | Trials | Best AUPRC | Median AUPRC | Worst AUPRC | Best configuration |",
            "|---|---|---|---|---|---|",
        ]
        for family in searches:
            result = read(out_dir, f"search_{family}")
            values = [trial["auprc_mean"] for trial in result["trials"]]
            params = ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in result["best"]["params"].items()
            )
            lines.append(
                f"| {family} | {result['n_trials']} | {max(values):.3f} "
                f"| {float(np.median(values)):.3f} | {min(values):.3f} | {params} |"
            )
        lines.append("")

    if (out_dir / "imbalance.json").exists():
        result = read(out_dir, "imbalance")
        n_seeds = len(result["seeds"])
        lines += [
            f"## Imbalance treatments (mean over {n_seeds} seeds)",
            "",
            "| Family | Treatment | AUPRC mean | Std over seeds | AUPRC per fold | Recall at FPR |",
            "|---|---|---|---|---|---|",
        ]
        for family, entry in result["families"].items():
            for strategy, summary in entry["strategies"].items():
                per_fold = ", ".join(f"{value:.3f}" for value in summary["auprc_per_fold"])
                marker = " (retained)" if entry.get("best_strategy") == strategy else ""
                lines.append(
                    f"| {family} | {strategy}{marker} | {summary['auprc_mean']:.3f} "
                    f"| {summary['auprc_seed_std']:.3f} | {per_fold} "
                    f"| {summary['recall_at_fpr_mean']:.3f} |"
                )
        lines.append("")

    if (out_dir / "calibration.json").exists():
        result = read(out_dir, "calibration")
        lines += [
            "## Probability calibration (classifier fitted on the first part of each "
            "training window, calibrator on the last part)",
            "",
            "| Family | Treatment | Method | Brier score | Log loss | AUPRC |",
            "|---|---|---|---|---|---|",
        ]
        for family, entry in result["families"].items():
            for method, summary in entry["methods"].items():
                lines.append(
                    f"| {family} | {entry['strategy']} | {method} "
                    f"| {summary['brier_mean']:.6f} | {summary['log_loss_mean']:.5f} "
                    f"| {summary['auprc_mean']:.3f} |"
                )
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stage", choices=STAGES, action="append")
    parser.add_argument("--family", choices=SEARCHED_FAMILIES, action="append")
    args = parser.parse_args()

    config = load_config(args.config)
    stages = args.stage or list(STAGES)
    out_dir = Path(config.results_dir) / "model_selection"
    metadata = run_metadata(config.to_dict(), sha256_of_file(config.data.raw_path))
    X, y, folds = load_development(config)

    def save(name: str, payload: dict) -> None:
        path = write_result(out_dir / f"{name}.json", {**payload, "metadata": metadata})
        print(f"  written to {path}")

    if "search" in stages:
        for family in args.family or SEARCHED_FAMILIES:
            print(f"Search: {family}", flush=True)
            save(f"search_{family}", run_search(config, X, y, folds, family))
    if "imbalance" in stages:
        print("Imbalance treatments", flush=True)
        save("imbalance", run_imbalance(config, X, y, folds, out_dir))
    if "calibration" in stages:
        print("Calibration", flush=True)
        save("calibration", run_calibration(config, X, y, folds, out_dir))
    write_summary(out_dir)
    print(f"Summary written to {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
