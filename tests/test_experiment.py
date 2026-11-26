import json

import pytest

from fraud_detection.experiment import (
    HoldoutAlreadyEvaluated,
    code_sha256,
    require_unevaluated,
    write_result,
)


def test_an_existing_holdout_result_blocks_a_new_evaluation(tmp_path):
    path = tmp_path / "final" / "holdout_evaluation.json"
    require_unevaluated(path)
    write_result(path, {"value": 1})
    with pytest.raises(HoldoutAlreadyEvaluated, match="evaluated once"):
        require_unevaluated(path)


def test_results_are_written_with_unix_line_endings(tmp_path):
    path = write_result(tmp_path / "result.json", {"a": [1, 2], "b": {"c": 3}})
    content = path.read_bytes()
    assert b"\r" not in content and content.endswith(b"\n")
    assert json.loads(content) == {"a": [1, 2], "b": {"c": 3}}


def test_code_digest_ignores_line_endings_and_detects_changes(tmp_path):
    for name in ("src/fraud_detection", "scripts", "configs"):
        (tmp_path / name).mkdir(parents=True)
    module = tmp_path / "src" / "fraud_detection" / "module.py"
    module.write_bytes(b"x = 1\ny = 2\n")
    (tmp_path / "configs" / "config.yaml").write_bytes(b"seed: 1\n")
    reference = code_sha256(tmp_path)

    module.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert code_sha256(tmp_path) == reference
    module.write_bytes(b"x = 1\ny = 3\n")
    assert code_sha256(tmp_path) != reference
    assert code_sha256(tmp_path / "empty") is None
