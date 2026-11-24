"""Typed access to the YAML configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from fraud_detection.simulation import SimulationConfig


@dataclass(frozen=True)
class DataConfig:
    raw_path: str
    sha256: str
    target: str
    time_column: str


@dataclass(frozen=True)
class SplitConfig:
    test_fraction: float
    n_cv_folds: int


@dataclass(frozen=True)
class EvaluationConfig:
    threshold: float
    max_fpr: float
    bootstrap_resamples: int
    confidence_level: float


@dataclass(frozen=True)
class BaselineModelConfig:
    n_estimators: int


@dataclass(frozen=True)
class ModelSelectionConfig:
    feature_set: str
    n_trials: dict[str, int]
    seeds: list[int]
    resampling_ratio: float
    calibration_fraction: float


@dataclass(frozen=True)
class SelectedModelConfig:
    family: str
    imbalance: str
    calibration: str
    params: dict[str, Any]


@dataclass(frozen=True)
class BehaviouralConfig:
    windows_days: list[int]
    delay_days: int
    test_fraction: float
    n_cv_folds: int
    seeds: list[int]
    feature_sets: dict[str, list[str]]


@dataclass(frozen=True)
class CostConfig:
    decision_rule: str
    alert_cost: float
    sensitivity_alert_costs: list[float]
    amount_column: str


@dataclass(frozen=True)
class Config:
    seed: int
    data: DataConfig
    split: SplitConfig
    evaluation: EvaluationConfig
    baseline_model: BaselineModelConfig
    model_selection: ModelSelectionConfig
    selected_model: SelectedModelConfig
    simulation: SimulationConfig
    behavioural: BehaviouralConfig
    cost: CostConfig
    model_dir: str
    results_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    """Read a YAML file and validate its structure and value ranges."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    config = Config(
        seed=int(raw["seed"]),
        data=DataConfig(**raw["data"]),
        split=SplitConfig(**raw["split"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        baseline_model=BaselineModelConfig(**raw["baseline_model"]),
        model_selection=ModelSelectionConfig(**raw["model_selection"]),
        selected_model=SelectedModelConfig(**raw["selected_model"]),
        simulation=SimulationConfig(**raw["simulation"]),
        behavioural=BehaviouralConfig(**raw["behavioural"]),
        cost=CostConfig(**raw["cost"]),
        model_dir=str(raw["model_dir"]),
        results_dir=str(raw["results_dir"]),
    )

    if not 0.0 < config.split.test_fraction < 1.0:
        raise ValueError("split.test_fraction must lie strictly between 0 and 1")
    if config.split.n_cv_folds < 2:
        raise ValueError("split.n_cv_folds must be at least 2")
    if config.cost.decision_rule not in ("expected_cost", "threshold"):
        raise ValueError("cost.decision_rule must be 'expected_cost' or 'threshold'")
    if not 0.0 < config.evaluation.confidence_level < 1.0:
        raise ValueError("evaluation.confidence_level must lie strictly between 0 and 1")
    return config
