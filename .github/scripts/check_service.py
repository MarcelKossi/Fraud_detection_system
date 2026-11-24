"""Smoke test of a running prediction service, using the standard library only.

Usage::

    python .github/scripts/check_service.py <base url> <examples.json>

Each example stored with the model is sent to the service; the probability and the
decision returned must match those recorded when the model was trained.
"""

from __future__ import annotations

import json
import sys
import urllib.request

TOLERANCE = 1e-5


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def main() -> int:
    base_url, examples_path = sys.argv[1].rstrip("/"), sys.argv[2]
    health = get(f"{base_url}/health")
    if health != {"status": "ok", "model_loaded": True}:
        print(f"Unexpected health report: {health}")
        return 1

    with open(examples_path, encoding="utf-8") as handle:
        examples = json.load(handle)
    failures = 0
    for example in examples:
        result = post(f"{base_url}/predict?explain=true", example["transaction"])
        expected = example["expected"]
        probability_ok = (
            abs(result["fraud_probability"] - expected["fraud_probability"]) <= TOLERANCE
        )
        alert_ok = result["decision"]["alert"] == expected["alert"]
        status = "ok" if probability_ok and alert_ok else "MISMATCH"
        failures += status != "ok"
        print(f"{status}: {example['label']} -> {result['fraud_probability']:.6f}")
    print(f"{len(examples) - failures}/{len(examples)} examples reproduced")
    return 1 if failures or not examples else 0


if __name__ == "__main__":
    sys.exit(main())
