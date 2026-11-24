from pathlib import Path

import pytest
import yaml

from fraud_detection.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def test_repository_config_is_valid():
    config = load_config(CONFIG_PATH)
    assert len(config.data.sha256) == 64
    assert config.to_dict()["split"]["n_cv_folds"] == config.split.n_cv_folds


def test_invalid_test_fraction_is_rejected(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["split"]["test_fraction"] = 1.5
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="test_fraction"):
        load_config(path)


def test_selected_model_matches_the_search_result():
    """The retained hyper-parameters are those of the best cross-validated trial."""
    import json

    from fraud_detection.models import build_model

    config = load_config(CONFIG_PATH)
    selected = config.selected_model
    path = CONFIG_PATH.parents[1] / "results" / "model_selection" / f"search_{selected.family}.json"
    best = json.loads(path.read_text(encoding="utf-8"))["best"]["params"]
    assert selected.params == pytest.approx(best)
    build_model(selected.family, selected.params, seed=config.seed)
