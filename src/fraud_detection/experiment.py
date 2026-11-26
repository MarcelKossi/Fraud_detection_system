"""Provenance and persistence of experiment results."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy
import pandas
import sklearn
import xgboost


CODE_PATHS = ("src", "scripts", "configs", "api", "pyproject.toml", "requirements.lock")


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def code_sha256(root: str | Path = ".") -> str | None:
    """SHA-256 of the source files that determine a result.

    Covers the package, the scripts and the configuration, in a fixed order. Unlike the
    commit identifier, it identifies the code even when a result is produced from a
    working tree with uncommitted changes.
    """
    root = Path(root)
    files = sorted(
        path
        for pattern in ("src/fraud_detection/*.py", "scripts/*.py", "configs/*.yaml")
        for path in root.glob(pattern)
    )
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        # Line endings are normalised so that the digest is the same on every platform.
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def run_metadata(config: dict[str, Any], data_sha256: str) -> dict[str, Any]:
    """Everything needed to trace a result back to its code, data and environment."""
    # Only the files that determine a result are considered: regenerated results,
    # documents and figures do not make the code "uncommitted".
    status = _git("status", "--porcelain", "--", *CODE_PATHS)
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_uncommitted_changes": bool(status) if status is not None else None,
        "code_sha256": code_sha256(),
        "data_sha256": data_sha256,
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }


def write_result(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a result file as indented JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Line endings are fixed so that results are byte-identical on every platform.
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


class HoldoutAlreadyEvaluated(RuntimeError):
    """Raised when an experiment that reads a hold-out set already has a stored result."""


def require_unevaluated(path: str | Path) -> None:
    """Refuse to repeat an evaluation on a hold-out set.

    A hold-out set gives an unbiased estimate only as long as no choice depends on it.
    Every experiment that reads one calls this function with the path of its result
    file, and stops if that file exists. Repeating the evaluation requires deleting the
    file deliberately, and the reason belongs in the journal.
    """
    if Path(path).exists():
        raise HoldoutAlreadyEvaluated(
            f"{path} already exists. This experiment reads a hold-out set, which is meant "
            "to be evaluated once; delete the file deliberately if a new evaluation is "
            "justified, and record why."
        )
