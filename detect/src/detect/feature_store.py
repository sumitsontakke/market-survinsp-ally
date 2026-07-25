"""feature_store.py — Load and serve the ML feature matrix from disk."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must NOT be used as features
_NON_FEATURE_ROLES = {"metadata", "label", "target"}

# Known label columns in the dataset
LABEL_COLUMNS = {
    "label_loose",
    "label_strict",
    "label_high_confidence",
    "label_partial",
    "label_positive",
}


class FeatureStore:
    """Loads the group-level ML dataset and provides the feature matrix.

    Parameters
    ----------
    dataset_path:
        Path to ``ml_dataset_groups_final.csv`` (or equivalent).
    schema_path:
        Path to ``feature_schema_final.json`` containing role metadata.
    """

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        schema_path: str | Path | None = None,
    ) -> None:
        repo_root = self._find_repo_root()
        self.dataset_path: Path = Path(dataset_path) if dataset_path else repo_root / "ml_dataset_groups_final.csv"
        self.schema_path: Path = Path(schema_path) if schema_path else repo_root / "feature_schema_final.json"
        self._df: pd.DataFrame | None = None
        self._schema: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_feature_matrix(
        self,
        label_col: str = "label_loose",
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Return ``(X, y, feature_names)`` ready for sklearn estimators.

        Parameters
        ----------
        label_col:
            One of ``"label_loose"``, ``"label_strict"``, or
            ``"label_high_confidence"``.

        Returns
        -------
        X:
            DataFrame of numeric feature columns (NaN-filled with 0).
        y:
            Integer label Series (0 / 1).
        feature_names:
            Ordered list of feature column names (same as ``X.columns``).
        """
        df = self._load_dataset()
        feature_names = self._get_feature_names(df)

        if label_col not in df.columns:
            available = [c for c in df.columns if c.startswith("label")]
            raise ValueError(
                f"Label column '{label_col}' not found. Available: {available}"
            )

        X = df[feature_names].fillna(0.0).astype(float)
        y = df[label_col].astype(int)
        return X, y, feature_names

    def get_dataframe(self) -> pd.DataFrame:
        """Return the raw dataset DataFrame."""
        return self._load_dataset()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_dataset(self) -> pd.DataFrame:
        if self._df is None:
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            self._df = pd.read_csv(self.dataset_path)
            logger.info("Loaded dataset %s — %d rows", self.dataset_path.name, len(self._df))
        return self._df

    def _get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Derive feature column names from schema or by exclusion."""
        schema = self._load_schema()
        if schema:
            dataset_key = "ml_dataset_groups_final.csv"
            entries = schema.get("datasets", {}).get(dataset_key, [])
            schema_features = [
                e["name"]
                for e in entries
                if e.get("role") == "feature" and e["name"] in df.columns
            ]
            if schema_features:
                return schema_features

        # Fallback: exclude known non-feature columns
        exclude = LABEL_COLUMNS | {"run_id", "scenario_family", "group_id", "group_type", "trader_ids",
                                    "window_start", "window_end", "generated_at", "dataset_version"}
        return [c for c in df.columns if c not in exclude and df[c].dtype.kind in "iufcb"]

    def _load_schema(self) -> dict[str, Any] | None:
        if self._schema is None:
            if self.schema_path.exists():
                try:
                    self._schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Failed to parse schema %s: %s", self.schema_path, exc)
                    self._schema = {}
        return self._schema

    @staticmethod
    def _find_repo_root() -> Path:
        """Walk up from this file to find the repo root (pyproject.toml)."""
        here = Path(__file__).resolve()
        for parent in [here, *here.parents]:
            if (parent / "pyproject.toml").exists():
                return parent
        # Fallback: two levels up from src/surveillance_ml/
        return here.parent.parent.parent
