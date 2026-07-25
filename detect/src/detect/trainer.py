"""trainer.py — Train the best model on the full dataset and save for production."""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from detect.feature_store import FeatureStore
from detect.models import MODEL_REGISTRY, NEEDS_SCALING

logger = logging.getLogger(__name__)

_MODELS_DIR_DEFAULT = "app_data_v2/models"
_BASELINE_DIR_DEFAULT = "outputs/ml_baseline"


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parent.parent.parent


def _extract_feature_importance(model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    """Extract feature importance or coefficients from any sklearn-compatible model."""
    estimator = model
    # Unwrap Pipeline
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("clf", model)

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        importance = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
    else:
        return []

    pairs = sorted(zip(feature_names, importance.tolist()), key=lambda x: -x[1])
    return [{"feature": f, "importance": round(float(v), 8)} for f, v in pairs]


def train_and_save_best(
    label_col: str = "label_loose",
    *,
    baseline_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    feature_store: FeatureStore | None = None,
) -> dict[str, Any]:
    """Train the best model (from baseline evaluation) on the full dataset.

    Reads the best model name from the evaluation summary JSON, trains it on
    the full dataset, and writes the following artefacts:

    - ``app_data_v2/models/tabular_group_scorer.joblib``
    - ``app_data_v2/models/tabular_group_scorer.json``  (metadata)
    - ``app_data_v2/models/feature_importance_{label_col}.csv``

    Parameters
    ----------
    label_col:
        Label column that was used during evaluation.
    baseline_dir:
        Directory where evaluation results live.
    models_dir:
        Directory to write trained model artefacts.
    feature_store:
        Optional pre-built :class:`FeatureStore`.

    Returns
    -------
    Metadata dict written to ``tabular_group_scorer.json``.
    """
    repo_root = _find_repo_root()
    baseline_dir = Path(baseline_dir) if baseline_dir else repo_root / _BASELINE_DIR_DEFAULT
    models_dir = Path(models_dir) if models_dir else repo_root / _MODELS_DIR_DEFAULT
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation summary to find best model name
    summary_path = baseline_dir / f"summary_{label_col}.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Evaluation summary not found at {summary_path}. "
            "Run evaluator.run_full_evaluation() first."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_model_name: str = summary["best_model"]
    logger.info("Best model identified: %s (F1=%.4f)", best_model_name, summary.get("best_f1_mean", 0.0))

    if best_model_name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{best_model_name}' not found in MODEL_REGISTRY.")

    # Load data
    store = feature_store or FeatureStore()
    X_df, y, feature_names = store.get_feature_matrix(label_col=label_col)
    X = X_df.values
    logger.info("Training '%s' on %d samples, %d features …", best_model_name, len(y), len(feature_names))

    # Build pipeline
    estimator = copy.deepcopy(MODEL_REGISTRY[best_model_name])
    if best_model_name in NEEDS_SCALING:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    else:
        pipeline = estimator

    pipeline.fit(X, y.values)
    logger.info("Training complete.")

    # Save model
    joblib.dump(pipeline, models_dir / "tabular_group_scorer.joblib")
    logger.info("Saved model → %s", models_dir / "tabular_group_scorer.joblib")

    # Build metadata
    from detect.dashboard.contracts import GRAPH_PACKAGE_SCHEMA_VERSION, SCORER_OUTPUT_SCHEMA_VERSION
    metadata: dict[str, Any] = {
        "schema_version": SCORER_OUTPUT_SCHEMA_VERSION,
        "model_name": best_model_name,
        "model_version": f"{best_model_name}_v1",
        "label_col": label_col,
        "expected_graph_schema_version": GRAPH_PACKAGE_SCHEMA_VERSION,
        "feature_columns": feature_names,
        "target_entity_type": "group",
        "threshold": 0.50,
        "eval_f1_mean": round(float(summary.get("best_f1_mean", 0.0)), 6),
        "eval_roc_auc_mean": round(float(summary.get("best_roc_auc_mean", 0.0)), 6),
        "n_training_samples": int(len(y)),
    }
    meta_path = models_dir / "tabular_group_scorer.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved metadata → %s", meta_path)

    # Save feature importance
    importance_rows = _extract_feature_importance(pipeline, feature_names)
    if importance_rows:
        fi_df = pd.DataFrame(importance_rows)
        fi_path = models_dir / f"feature_importance_{label_col}.csv"
        fi_df.to_csv(fi_path, index=False)
        logger.info("Saved feature importance → %s", fi_path)

    print(f"\n[trainer] Best model '{best_model_name}' trained and saved to {models_dir}")
    print(f"  F1 (CV): {summary.get('best_f1_mean', 0.0):.4f} | ROC-AUC (CV): {summary.get('best_roc_auc_mean', 0.0):.4f}")
    return metadata
