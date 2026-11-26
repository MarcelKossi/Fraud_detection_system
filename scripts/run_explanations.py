"""Global feature attributions of the retained model and their stability.

For each validation period of the development set and each seed, the retained model is
fitted on the preceding transactions and exact Shapley contributions are computed on
the validation transactions. The script reports the mean absolute contribution of each
feature, its rank in each period, the agreement of the rankings between periods, and
checks that contributions add up to the predicted log-odds.

The hold-out set is not read. Results are written to ``results/explanations/``.

Usage::

    python scripts/run_explanations.py
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from fraud_detection.config import load_config
from fraud_detection.data import load_raw, prepare, sha256_of_file
from fraud_detection.experiment import run_metadata, write_result
from fraud_detection.explain import predicted_log_odds, shap_contributions
from fraud_detection.features import build_features
from fraud_detection.splits import expanding_window_folds, temporal_holdout
from fraud_detection.validation import fit_on_fold

TOP_K = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    selection, selected = config.model_selection, config.selected_model

    prepared = prepare(load_raw(config.data.raw_path, config.data.sha256), config.data.time_column)
    dev_idx, _ = temporal_holdout(len(prepared), config.split.test_fraction)
    development = prepared.iloc[dev_idx]
    folds = expanding_window_folds(len(development), config.split.n_cv_folds)
    X, y = build_features(
        development, config.data.target, config.data.time_column, selection.feature_set
    )
    features = list(X.columns)

    # importance[seed, fold, feature]: mean absolute contribution on the validation fold.
    importance = np.zeros((len(selection.seeds), len(folds), len(features)))
    fraud_importance = np.zeros_like(importance)
    additivity_error = 0.0
    for s, seed in enumerate(selection.seeds):
        for f, (train_idx, val_idx) in enumerate(folds):
            model = fit_on_fold(
                selected.family,
                selected.params,
                X.iloc[train_idx],
                y.iloc[train_idx],
                selected.imbalance,
                selection.resampling_ratio,
                seed,
            )
            X_val = X.iloc[val_idx]
            contributions, base = shap_contributions(model, X_val)
            reconstructed = contributions.sum(axis=1) + base
            additivity_error = max(
                additivity_error,
                float(np.abs(reconstructed - predicted_log_odds(model, X_val)).max()),
            )
            importance[s, f] = np.abs(contributions).mean(axis=0)
            is_fraud = y.iloc[val_idx].to_numpy() == 1
            fraud_importance[s, f] = np.abs(contributions[is_fraud]).mean(axis=0)
        print(f"Seed {seed} done", flush=True)

    per_fold = importance.mean(axis=0)  # (fold, feature), averaged over seeds
    overall = per_fold.mean(axis=0)
    order = np.argsort(-overall)
    ranks = (-per_fold).argsort(axis=1).argsort(axis=1) + 1
    fold_agreement = [
        float(spearmanr(per_fold[a], per_fold[b]).statistic)
        for a, b in combinations(range(len(folds)), 2)
    ]
    seed_agreement = [
        float(spearmanr(importance[a].mean(axis=0), importance[b].mean(axis=0)).statistic)
        for a, b in combinations(range(len(selection.seeds)), 2)
    ]
    top_sets = [set(np.argsort(-per_fold[f])[:TOP_K]) for f in range(len(folds))]

    payload = {
        "experiment": "global_attributions",
        "scale": "log-odds of the uncalibrated classifier",
        "seeds": selection.seeds,
        "n_folds": len(folds),
        "max_additivity_error": additivity_error,
        "rank_agreement_between_folds": {
            "spearman_mean": float(np.mean(fold_agreement)),
            "spearman_min": float(np.min(fold_agreement)),
        },
        "rank_agreement_between_seeds": {
            "spearman_mean": float(np.mean(seed_agreement)),
            "spearman_min": float(np.min(seed_agreement)),
        },
        "top_k": TOP_K,
        "features_in_top_k_of_every_fold": sorted(features[i] for i in set.intersection(*top_sets)),
        "share_of_total_importance_in_top_k": float(overall[order[:TOP_K]].sum() / overall.sum()),
        "features": [
            {
                "feature": features[i],
                "mean_abs_contribution": float(overall[i]),
                "mean_abs_contribution_on_frauds": float(fraud_importance[:, :, i].mean()),
                "per_fold": per_fold[:, i].tolist(),
                "rank_per_fold": ranks[:, i].tolist(),
            }
            for i in order
        ],
        "metadata": run_metadata(config.to_dict(), sha256_of_file(config.data.raw_path)),
    }
    out_dir = Path(config.results_dir) / "explanations"
    print(f"Written to {write_result(out_dir / 'global_importance.json', payload)}")
    write_summary(out_dir)


def write_summary(out_dir: Path) -> None:
    """Regenerate ``summary.md`` from ``global_importance.json``."""
    result = json.loads((out_dir / "global_importance.json").read_text(encoding="utf-8"))
    folds, seeds = result["rank_agreement_between_folds"], result["rank_agreement_between_seeds"]
    lines = [
        "# Global feature attributions",
        "",
        "Generated by `scripts/run_explanations.py`; do not edit.",
        "",
        f"Mean absolute Shapley contribution, on the {result['scale']}, computed on the "
        f"validation periods of the development set ({len(result['seeds'])} seeds).",
        "",
        f"- Largest additivity error: {result['max_additivity_error']:.2e}.",
        f"- Spearman rank correlation of importances between periods: mean "
        f"{folds['spearman_mean']:.3f}, minimum {folds['spearman_min']:.3f}.",
        f"- Between seeds: mean {seeds['spearman_mean']:.3f}, minimum {seeds['spearman_min']:.3f}.",
        f"- Features in the top {result['top_k']} of every period: "
        f"{', '.join(result['features_in_top_k_of_every_fold'])}.",
        f"- Share of total importance carried by the top {result['top_k']}: "
        f"{result['share_of_total_importance_in_top_k']:.1%}.",
        "",
        "| Rank | Feature | Mean abs. contribution | On frauds only | Rank in each period |",
        "|---|---|---|---|---|",
    ]
    for rank, entry in enumerate(result["features"], start=1):
        lines.append(
            f"| {rank} | `{entry['feature']}` | {entry['mean_abs_contribution']:.3f} "
            f"| {entry['mean_abs_contribution_on_frauds']:.3f} "
            f"| {', '.join(str(r) for r in entry['rank_per_fold'])} |"
        )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Summary written to {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
