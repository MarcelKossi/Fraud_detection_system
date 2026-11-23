"""Provenance and persistence of experiment results."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy
import pandas
import sklearn


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def run_metadata(config: dict[str, Any], data_sha256: str) -> dict[str, Any]:
    """Everything needed to trace a result back to its code, data and environment."""
    status = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_uncommitted_changes": bool(status) if status is not None else None,
        "data_sha256": data_sha256,
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }


def write_result(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a result file as indented JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path
