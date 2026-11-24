"""Check that the raw data file is present and intact, then describe it.

Usage::

    python scripts/prepare_data.py
"""

from __future__ import annotations

import argparse

from fraud_detection.config import load_config
from fraud_detection.data import load_raw, prepare
from fraud_detection.splits import expanding_window_folds, temporal_holdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    target, time_column = config.data.target, config.data.time_column

    raw = load_raw(config.data.raw_path, config.data.sha256)
    print(f"Checksum verified: {config.data.raw_path}")
    print(f"Rows: {len(raw):,}  frauds: {int(raw[target].sum()):,}")

    duplicated = raw.duplicated(keep="first")
    print(
        f"Exact duplicate rows: {int(duplicated.sum()):,} "
        f"(of which frauds: {int(raw.loc[duplicated, target].sum()):,})"
    )

    clean = prepare(raw, time_column)
    dev_idx, test_idx = temporal_holdout(len(clean), config.split.test_fraction)
    print(f"After de-duplication: {len(clean):,} rows, {int(clean[target].sum()):,} frauds")
    for name, idx in (("development", dev_idx), ("hold-out", test_idx)):
        part = clean.iloc[idx]
        hours = part[time_column] / 3600
        print(
            f"  {name:<12} {len(part):>7,} rows  {int(part[target].sum()):>4} frauds  "
            f"hours {hours.min():.1f} to {hours.max():.1f}"
        )
    development = clean.iloc[dev_idx]
    for i, (train_idx, val_idx) in enumerate(
        expanding_window_folds(len(development), config.split.n_cv_folds), start=1
    ):
        print(
            f"  fold {i}: train {len(train_idx):>7,} rows "
            f"({int(development[target].iloc[train_idx].sum())} frauds), "
            f"validation {len(val_idx):>6,} rows "
            f"({int(development[target].iloc[val_idx].sum())} frauds)"
        )


if __name__ == "__main__":
    main()
