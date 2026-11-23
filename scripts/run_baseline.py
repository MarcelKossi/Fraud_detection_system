"""Reference experiments for the baseline model.

Three experiments, each written to ``results/baseline/<name>.json``:

``v1_replication``
    Random stratified split on the raw file, exactly as in the first version.
``v1_temporal_holdout``
    Same model and features, trained on the development period of the de-duplicated
    file and evaluated on the final temporal hold-out set.
``cv_development``
    Expanding-window cross-validation restricted to the development period, for each
    feature set. The hold-out set is not read by this experiment.

Usage::

    python scripts/run_baseline.py                      # all experiments
    python scripts/run_baseline.py --only cv_development
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fraud_detection.config import Config, load_config
from fraud_detection.data import load_raw, prepare, sha256_of_file
from fraud_detection.experiment import run_metadata, write_result
from fraud_detection.features import FEATURE_SETS, build_features
from fraud_detection.metrics import evaluate_scores
from fraud_detection.models import build_baseline
from fraud_detection.splits import (
    assert_chronological,
    assert_disjoint,
    expanding_window_folds,
    random_stratified_split,
    temporal_holdout,
)

EXPERIMENTS = ("v1_replication", "v1_temporal_holdout", "cv_development")


def _fit_and_score(config: Config, X_train, y_train, X_eval) -> np.ndarray:
    model = build_baseline(config.baseline_model.n_estimators, config.seed)
    model.fit(X_train, y_train)
    return model.predict_proba(X_eval)[:, 1]


def _report(config: Config, y_true, scores, with_ci: bool) -> dict:
    ev = config.evaluation
    return evaluate_scores(
        np.asarray(y_true),
        scores,
        threshold=ev.threshold,
        max_fpr=ev.max_fpr,
        n_resamples=ev.bootstrap_resamples if with_ci else 0,
        confidence_level=ev.confidence_level,
        seed=config.seed,
    )


def v1_replication(config: Config, raw) -> dict:
    X, y = build_features(raw, config.data.target, config.data.time_column, "raw_time")
    train_idx, test_idx = random_stratified_split(
        y.to_numpy(), config.split.test_fraction, config.seed
    )
    assert_disjoint(train_idx, test_idx)
    scores = _fit_and_score(config, X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx])

    # Test rows that are exact copies of a training row (all columns, target included).
    matched = raw.iloc[test_idx].merge(
        raw.iloc[train_idx].drop_duplicates(), how="left", indicator=True
    )
    shared = matched["_merge"] == "both"
    return {
        "protocol": "random stratified split, raw file, duplicates kept",
        "feature_set": "raw_time",
        "n_train": int(len(train_idx)),
        "test_rows_also_in_train": {
            "rows": int(shared.sum()),
            "frauds": int(matched.loc[shared, config.data.target].sum()),
        },
        "test": _report(config, y.iloc[test_idx], scores, with_ci=True),
    }


def v1_temporal_holdout(config: Config, prepared) -> dict:
    X, y = build_features(prepared, config.data.target, config.data.time_column, "raw_time")
    dev_idx, test_idx = temporal_holdout(len(prepared), config.split.test_fraction)
    assert_disjoint(dev_idx, test_idx)
    assert_chronological(prepared[config.data.time_column].to_numpy(), dev_idx, test_idx)
    scores = _fit_and_score(config, X.iloc[dev_idx], y.iloc[dev_idx], X.iloc[test_idx])
    return {
        "protocol": "temporal hold-out, de-duplicated file",
        "feature_set": "raw_time",
        "n_train": int(len(dev_idx)),
        "test": _report(config, y.iloc[test_idx], scores, with_ci=True),
    }


def cv_development(config: Config, prepared) -> dict:
    dev_idx, _ = temporal_holdout(len(prepared), config.split.test_fraction)
    development = prepared.iloc[dev_idx]
    times = development[config.data.time_column].to_numpy()
    folds = expanding_window_folds(len(development), config.split.n_cv_folds)

    by_feature_set = {}
    for feature_set in FEATURE_SETS:
        X, y = build_features(development, config.data.target, config.data.time_column, feature_set)
        fold_reports = []
        for train_idx, val_idx in folds:
            assert_disjoint(train_idx, val_idx)
            assert_chronological(times, train_idx, val_idx)
            scores = _fit_and_score(config, X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx])
            report = _report(config, y.iloc[val_idx], scores, with_ci=False)
            report["n_train"] = int(len(train_idx))
            report["n_train_positive"] = int(y.iloc[train_idx].sum())
            fold_reports.append(report)
        auprcs = np.array([r["auprc"] for r in fold_reports])
        recalls = np.array([r["recall_at_fpr"]["value"] for r in fold_reports])
        by_feature_set[feature_set] = {
            "folds": fold_reports,
            "auprc_mean": float(auprcs.mean()),
            "auprc_std": float(auprcs.std(ddof=1)),
            "recall_at_fpr_mean": float(recalls.mean()),
            "recall_at_fpr_std": float(recalls.std(ddof=1)),
        }
    return {
        "protocol": "expanding-window cross-validation on the development period",
        "n_development": int(len(development)),
        "n_development_positive": int(development[config.data.target].sum()),
        "feature_sets": by_feature_set,
    }


def write_summary(out_dir: Path) -> None:
    """Regenerate ``summary.md`` from the JSON files present in ``out_dir``."""
    lines = ["# Baseline results", "", "Generated by `scripts/run_baseline.py`; do not edit.", ""]

    holdout_rows = []
    for name in ("v1_replication", "v1_temporal_holdout"):
        path = out_dir / f"{name}.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            test, at = result["test"], result["test"]["at_threshold"]
            low, high = test["auprc_ci"]
            holdout_rows.append(
                f"| {result['protocol']} | {test['n_positive']} | {at['precision']:.3f} "
                f"| {at['recall']:.3f} | {at['f1']:.3f} "
                f"| {test['auprc']:.3f} [{low:.3f}, {high:.3f}] |"
            )
    if holdout_rows:
        lines += [
            "## Random forest of the first version, two evaluation protocols",
            "",
            "| Protocol | Frauds in test | Precision | Recall | F1 | AUPRC [95% CI] |",
            "|---|---|---|---|---|---|",
            *holdout_rows,
            "",
            "Precision, recall and F1 are computed at a threshold of 0.5.",
            "",
        ]

    path = out_dir / "cv_development.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        lines += [
            "## Expanding-window cross-validation on the development period",
            "",
            "| Feature set | AUPRC per fold | AUPRC mean (std) | Recall at FPR mean (std) |",
            "|---|---|---|---|",
        ]
        for feature_set, summary in result["feature_sets"].items():
            per_fold = ", ".join(f"{fold['auprc']:.3f}" for fold in summary["folds"])
            lines.append(
                f"| `{feature_set}` | {per_fold} "
                f"| {summary['auprc_mean']:.3f} ({summary['auprc_std']:.3f}) "
                f"| {summary['recall_at_fpr_mean']:.3f} ({summary['recall_at_fpr_std']:.3f}) |"
            )
        first = next(iter(result["feature_sets"].values()))
        positives = ", ".join(str(fold["n_positive"]) for fold in first["folds"])
        max_fpr = first["folds"][0]["recall_at_fpr"]["max_fpr"]
        lines += [
            "",
            f"Frauds per validation fold: {positives}. "
            f"Recall is reported at a false-positive rate of at most {max_fpr}.",
            "",
        ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--only", choices=EXPERIMENTS, action="append")
    args = parser.parse_args()

    config = load_config(args.config)
    selected = args.only or list(EXPERIMENTS)

    data_sha256 = sha256_of_file(config.data.raw_path)
    raw = load_raw(config.data.raw_path, config.data.sha256)
    prepared = prepare(raw, config.data.time_column)
    out_dir = Path(config.results_dir) / "baseline"

    runners = {
        "v1_replication": lambda: v1_replication(config, raw),
        "v1_temporal_holdout": lambda: v1_temporal_holdout(config, prepared),
        "cv_development": lambda: cv_development(config, prepared),
    }
    for name in selected:
        print(f"Running {name} ...", flush=True)
        payload = {
            "experiment": name,
            **runners[name](),
            "metadata": run_metadata(config.to_dict(), data_sha256),
        }
        print(f"  written to {write_result(out_dir / f'{name}.json', payload)}")
    write_summary(out_dir)
    print(f"Summary written to {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
