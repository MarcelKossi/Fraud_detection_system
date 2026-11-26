"""Checks on the two comparisons that carry the conclusions of the project.

``model_comparison``
    Paired bootstrap of the cross-validated AUPRC difference between the retained model
    and the random forest, on the validation periods of the development set. Validation
    transactions are resampled within each period; for each resample the mean AUPRC over
    periods and seeds is computed for both models. The interval reflects the sampling of
    transactions, which the spread over training seeds does not capture. Wins and losses
    are also counted for every pair of seed and period.

``protocol_decomposition``
    Separation of the two defects of the evaluation protocol of the first version. The
    model of the first version is evaluated on repeated random stratified splits of the
    raw file and of the de-duplicated file. The difference between the two isolates the
    effect of duplicated rows; the difference between random splits of the de-duplicated
    file and the temporal folds reflects chronology, together with the smaller training
    sets and the specific periods of those folds.

The hold-out set is not evaluated by either stage: the random splits of the second stage
are drawn from the whole file, as in the first version, and no result of this script is
used for any modelling choice. Results are written to ``results/robustness/``.

Usage::

    python scripts/run_robustness_checks.py
    python scripts/run_robustness_checks.py --stage model_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fraud_detection.config import Config, load_config
from fraud_detection.data import deduplicate, load_raw, prepare, sha256_of_file
from fraud_detection.experiment import run_metadata, write_result
from fraud_detection.features import build_features
from fraud_detection.metrics import auprc, threshold_metrics
from fraud_detection.models import build_baseline
from fraud_detection.splits import expanding_window_folds, random_stratified_split, temporal_holdout
from fraud_detection.validation import fit_on_fold

STAGES = ("model_comparison", "protocol_decomposition")
REFERENCE = "random_forest"


def model_comparison(config: Config) -> dict:
    selection, selected, ev = config.model_selection, config.selected_model, config.evaluation
    prepared = prepare(load_raw(config.data.raw_path, config.data.sha256), config.data.time_column)
    dev_idx, _ = temporal_holdout(len(prepared), config.split.test_fraction)
    development = prepared.iloc[dev_idx]
    folds = expanding_window_folds(len(development), config.split.n_cv_folds)
    X, y = build_features(
        development, config.data.target, config.data.time_column, selection.feature_set
    )
    systems = {
        "retained": (selected.family, selected.params),
        "reference": (REFERENCE, {"n_estimators": config.baseline_model.n_estimators}),
    }

    # scores[system][seed][fold]: scores on the validation transactions of the fold.
    scores = {name: [] for name in systems}
    for seed in selection.seeds:
        for name, (family, params) in systems.items():
            per_fold = []
            for train_idx, val_idx in folds:
                model = fit_on_fold(
                    family, params, X.iloc[train_idx], y.iloc[train_idx], "none", 0.0, seed
                )
                per_fold.append(model.predict_proba(X.iloc[val_idx])[:, 1])
            scores[name].append(per_fold)
        print(f"  seed {seed} fitted", flush=True)
    labels = [y.iloc[val_idx].to_numpy() for _, val_idx in folds]

    def mean_auprc(name: str, index_per_fold: list[np.ndarray]) -> float:
        return float(
            np.mean(
                [
                    auprc(labels[f][index], scores[name][s][f][index])
                    for s in range(len(selection.seeds))
                    for f, index in enumerate(index_per_fold)
                ]
            )
        )

    everything = [np.arange(len(label)) for label in labels]
    point = {name: mean_auprc(name, everything) for name in systems}

    rng = np.random.default_rng(config.seed)
    differences = []
    while len(differences) < ev.bootstrap_resamples:
        index_per_fold = [rng.integers(0, len(label), len(label)) for label in labels]
        if any(labels[f][index].sum() == 0 for f, index in enumerate(index_per_fold)):
            continue
        differences.append(
            mean_auprc("retained", index_per_fold) - mean_auprc("reference", index_per_fold)
        )
    alpha = (1.0 - ev.confidence_level) / 2.0
    low, high = np.quantile(differences, [alpha, 1.0 - alpha])

    # Wins of the retained model for every (seed of the retained model, seed of the
    # reference, period) triple: seeds of different libraries are not paired units.
    cell = {
        name: np.array(
            [
                [auprc(labels[f], scores[name][s][f]) for f in range(len(folds))]
                for s in range(len(selection.seeds))
            ]
        )
        for name in systems
    }
    wins = cell["retained"][:, None, :] > cell["reference"][None, :, :]
    return {
        "retained": {"family": selected.family, "auprc": point["retained"]},
        "reference": {"family": REFERENCE, "auprc": point["reference"]},
        "difference": {
            "value": point["retained"] - point["reference"],
            "ci": [float(low), float(high)],
            "share_positive": float(np.mean(np.array(differences) > 0)),
        },
        "difference_per_period": (
            cell["retained"].mean(axis=0) - cell["reference"].mean(axis=0)
        ).tolist(),
        "reference_seed_std_per_period": cell["reference"].std(axis=0, ddof=1).tolist(),
        "retained_seed_std_per_period": cell["retained"].std(axis=0, ddof=1).tolist(),
        "wins_over_seed_pairs_per_period": wins.sum(axis=(0, 1)).tolist(),
        "seed_pairs_per_period": int(wins.shape[0] * wins.shape[1]),
        "n_resamples": len(differences),
        "confidence_level": ev.confidence_level,
        "seeds": selection.seeds,
    }


def protocol_decomposition(config: Config) -> dict:
    data, ev, seeds = config.data, config.evaluation, config.model_selection.seeds
    raw = load_raw(data.raw_path, data.sha256)
    variants = {"raw_file": raw, "deduplicated_file": deduplicate(raw)}

    result = {}
    for name, frame in variants.items():
        X, y = build_features(frame, data.target, data.time_column, "raw_time")
        runs = []
        for seed in seeds:
            train_idx, test_idx = random_stratified_split(
                y.to_numpy(), config.split.test_fraction, seed
            )
            model = build_baseline(config.baseline_model.n_estimators, seed)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores = model.predict_proba(X.iloc[test_idx])[:, 1]
            y_test = y.iloc[test_idx].to_numpy()
            matched = frame.iloc[test_idx].merge(
                frame.iloc[train_idx].drop_duplicates(), how="left", indicator=True
            )
            shared = matched["_merge"] == "both"
            at = threshold_metrics(y_test, scores, ev.threshold)
            runs.append(
                {
                    "seed": seed,
                    "auprc": auprc(y_test, scores),
                    "precision": at["precision"],
                    "recall": at["recall"],
                    "n_test_frauds": int(y_test.sum()),
                    "test_rows_also_in_train": int(shared.sum()),
                    "test_frauds_also_in_train": int(matched.loc[shared, data.target].sum()),
                }
            )
            print(f"  {name} seed {seed}: AUPRC {runs[-1]['auprc']:.4f}", flush=True)
        result[name] = {
            **{
                f"{key}_mean": float(np.mean([run[key] for run in runs]))
                for key in ("auprc", "precision", "recall")
            },
            **{
                f"{key}_std": float(np.std([run[key] for run in runs], ddof=1))
                for key in ("auprc", "precision", "recall")
            },
            "runs": runs,
        }
    return {
        "protocol": "random stratified splits, model and features of the first version",
        "seeds": seeds,
        "variants": result,
        "duplicate_effect": {
            key: result["raw_file"][f"{key}_mean"] - result["deduplicated_file"][f"{key}_mean"]
            for key in ("auprc", "recall")
        },
    }


def write_summary(out_dir: Path, results_dir: Path) -> None:
    """Regenerate ``summary.md`` from the JSON files present in ``out_dir``."""
    lines = ["# Robustness checks", "", "Generated by `scripts/run_robustness_checks.py`.", ""]
    path = out_dir / "model_comparison.json"
    if path.exists():
        r = json.loads(path.read_text(encoding="utf-8"))
        d = r["difference"]
        per_period = ", ".join(f"{v:+.3f}" for v in r["difference_per_period"])
        wins = ", ".join(
            f"{w}/{r['seed_pairs_per_period']}" for w in r["wins_over_seed_pairs_per_period"]
        )
        lines += [
            "## Retained model against the random forest, cross-validation",
            "",
            f"- Mean AUPRC: {r['retained']['auprc']:.3f} against {r['reference']['auprc']:.3f}.",
            f"- Difference: {d['value']:+.3f}, 95% paired bootstrap interval "
            f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}], positive in {d['share_positive']:.0%} "
            "of the resamples.",
            f"- Difference per period: {per_period}.",
            f"- Wins of the retained model over pairs of seeds, per period: {wins}.",
            "",
        ]
    path = out_dir / "protocol_decomposition.json"
    if path.exists():
        r = json.loads(path.read_text(encoding="utf-8"))
        lines += [
            "## Decomposition of the protocol of the first version",
            "",
            "| Protocol | AUPRC, mean (std) | Recall at 0.5, mean (std) "
            "| Test frauds also in training |",
            "|---|---|---|---|",
        ]
        labels = {
            "raw_file": "Random splits, duplicates kept",
            "deduplicated_file": "Random splits, duplicates removed",
        }
        for name, v in r["variants"].items():
            shared = np.mean([run["test_frauds_also_in_train"] for run in v["runs"]])
            lines.append(
                f"| {labels[name]} | {v['auprc_mean']:.3f} ({v['auprc_std']:.3f}) "
                f"| {v['recall_mean']:.3f} ({v['recall_std']:.3f}) | {shared:.1f} |"
            )
        cv_path = results_dir / "baseline" / "cv_development.json"
        if cv_path.exists():
            cv = json.loads(cv_path.read_text(encoding="utf-8"))["feature_sets"]["raw_time"]
            recalls = [fold["at_threshold"]["recall"] for fold in cv["folds"]]
            lines.append(
                f"| Temporal folds, duplicates removed | {cv['auprc_mean']:.3f} "
                f"({cv['auprc_std']:.3f}) | {np.mean(recalls):.3f} "
                f"({np.std(recalls, ddof=1):.3f}) | 0 |"
            )
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written to {out_dir / 'summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stage", choices=STAGES, action="append")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(config.results_dir) / "robustness"
    runners = {
        "model_comparison": model_comparison,
        "protocol_decomposition": protocol_decomposition,
    }
    for stage in args.stage or STAGES:
        print(f"{stage} ...", flush=True)
        payload = {
            "experiment": stage,
            **runners[stage](config),
            "metadata": run_metadata(config.to_dict(), sha256_of_file(config.data.raw_path)),
        }
        print(f"  written to {write_result(out_dir / f'{stage}.json', payload)}")
    write_summary(out_dir, Path(config.results_dir))


if __name__ == "__main__":
    main()
