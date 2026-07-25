"""Rung 3 XGBoost wrapper.

Hyperparameters default to the proven Phase 1 v1 setup
(``n_estimators=500, max_depth=7, lr=0.05`` per
``scripts/run_edge_level_experiment.py``) so a future Rung 3 retraining
on the new (calibrated) cohort uses the same model class.

This wrapper is callable but DEFERRED in M1 - the user has chosen to
focus on GNN. Rung 3 retraining is a later, optional pass once the
calibrated synthetic cohort is regenerated.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class EdgeXGBoostModel:
    """Sklearn-style wrapper around xgboost for edge classification."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 7,
        learning_rate: float = 0.05,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        random_state: int = 42,
        device: str = "cpu",
    ) -> None:
        try:
            import xgboost as xgb
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "xgboost is required for the Rung 3 wrapper. "
                "Install in the trainer image."
            ) from exc

        self._xgb = xgb
        self._model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            device=device,
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        )
        self.feature_names: list[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "EdgeXGBoostModel":
        self._model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    @property
    def model(self):
        return self._model
