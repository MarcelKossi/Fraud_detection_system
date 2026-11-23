"""Loading and cleaning of the raw transaction file."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DATASET_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"


def sha256_of_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: str | Path, expected_sha256: str | None = None) -> pd.DataFrame:
    """Load the raw CSV, optionally checking its integrity first.

    Raises
    ------
    FileNotFoundError
        If the file is absent; the message states where to obtain it.
    ValueError
        If the checksum differs from ``expected_sha256``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download creditcard.csv from {DATASET_URL} and place it there."
        )
    if expected_sha256 is not None:
        actual = sha256_of_file(path)
        if actual != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual}."
            )
    return pd.read_csv(path)


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are exact copies of an earlier row (all columns, target included).

    The first occurrence is kept and the original order is preserved, so that an
    identical transaction can never appear on both sides of a split.
    """
    return frame.drop_duplicates(keep="first").reset_index(drop=True)


def sort_by_time(frame: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """Order transactions chronologically with a stable sort."""
    return frame.sort_values(time_column, kind="mergesort").reset_index(drop=True)


def prepare(frame: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """De-duplicate, then order chronologically: the input expected by temporal splits."""
    return sort_by_time(deduplicate(frame), time_column)
