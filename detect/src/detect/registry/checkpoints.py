"""Checkpoint registry - file-based, no external DB.

Each saved experiment is a directory under ``model_registry/<experiment_id>__<timestamp>/``
containing:
  - config.yaml         the ExperimentConfig used
  - metrics.json        the MetricBundle
  - model.joblib        the model object (pickled by joblib for sklearn / xgboost,
                        torch.save for nn.Module - handled by caller)
  - notes.md            optional free-text from the trainer

Reference
---------
Cont, R. (2001). Empirical properties of asset returns. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from detect.evaluate.metrics import MetricBundle
from detect.registry.experiment import ExperimentConfig

REGISTRY_ROOT = Path(os.environ.get("MODEL_REGISTRY_DIR", "/data/model_registry"))


def _now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_experiment(
    config: ExperimentConfig,
    metrics: MetricBundle,
    model_writer=None,
    notes: str = "",
    root: Path | None = None,
) -> Path:
    """Persist a finished experiment.

    ``model_writer`` is an optional callable taking a target Path. The
    caller writes the model in whatever format suits (joblib for sklearn /
    xgboost, torch.save for torch.nn.Module).
    """
    target_root = root if root is not None else REGISTRY_ROOT
    folder = target_root / f"{config.experiment_id}__{_now_token()}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    metrics.write_json(folder / "metrics.json")
    if notes:
        (folder / "notes.md").write_text(notes, encoding="utf-8")
    if model_writer is not None:
        model_writer(folder / "model.bin")
    return folder


def list_experiments(root: Path | None = None) -> list[dict[str, Any]]:
    """Inventory of saved experiments. Cheap; no model files loaded."""
    target_root = root if root is not None else REGISTRY_ROOT
    if not target_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for folder in sorted(target_root.iterdir()):
        if not folder.is_dir():
            continue
        m_path = folder / "metrics.json"
        c_path = folder / "config.yaml"
        if not m_path.exists() or not c_path.exists():
            continue
        try:
            metrics = MetricBundle.read_json(m_path)
            config_payload = yaml.safe_load(c_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        rows.append(
            {
                "folder": str(folder),
                "experiment_id": config_payload.get("experiment_id"),
                "model_family": config_payload.get("model", {}).get("family"),
                "seed": config_payload.get("seed"),
                "cv_f1": metrics.cv_f1,
                "locked_ring_recall": metrics.locked_ring_recall,
                "locked_mixed_recall": metrics.locked_mixed_recall,
                "locked_benign_alarm": metrics.locked_benign_alarm,
                "n_train_runs": metrics.n_train_runs,
                "n_eval_runs": metrics.n_eval_runs,
            }
        )
    return rows


def load_experiment(folder: Path | str) -> tuple[ExperimentConfig, MetricBundle]:
    folder_path = Path(folder)
    config = ExperimentConfig.from_yaml(folder_path / "config.yaml")
    metrics = MetricBundle.read_json(folder_path / "metrics.json")
    return config, metrics
