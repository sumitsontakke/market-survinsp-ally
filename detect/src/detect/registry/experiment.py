"""ExperimentConfig - YAML-driven config dataclass for a training run.

Validates required fields on load; raises ``ValueError`` early so a bad
config never silently produces a degenerate model.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    seed: int

    model: dict[str, Any]
    features: dict[str, Any]
    data: dict[str, Any]
    calibration: dict[str, Any]
    split: dict[str, Any]
    loss: dict[str, Any]
    evaluation: dict[str, Any]
    training: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentConfig":
        text = Path(path).read_text(encoding="utf-8")
        payload = yaml.safe_load(text)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        required = (
            "experiment_id", "seed", "model", "features",
            "data", "calibration", "split", "loss", "evaluation",
        )
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(
                f"ExperimentConfig: missing required fields {missing}"
            )
        # Sub-section sanity.
        if "family" not in payload["model"]:
            raise ValueError("ExperimentConfig.model.family is required")
        if "policy" not in payload["split"]:
            raise ValueError("ExperimentConfig.split.policy is required")
        if "runs" not in payload["data"]:
            raise ValueError("ExperimentConfig.data.runs is required")
        return cls(
            experiment_id=str(payload["experiment_id"]),
            seed=int(payload["seed"]),
            model=dict(payload["model"]),
            features=dict(payload["features"]),
            data=dict(payload["data"]),
            calibration=dict(payload["calibration"]),
            split=dict(payload["split"]),
            loss=dict(payload["loss"]),
            evaluation=dict(payload["evaluation"]),
            training=dict(payload.get("training", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------------
    # convenience accessors
    # ------------------------------------------------------------------
    @property
    def model_family(self) -> str:
        return str(self.model["family"])

    @property
    def split_policy(self) -> str:
        return str(self.split["policy"])

    @property
    def runs_cohort(self) -> str:
        return str(self.data["runs"])

    @property
    def calibration_apply(self) -> bool:
        return bool(self.calibration.get("apply", False))
