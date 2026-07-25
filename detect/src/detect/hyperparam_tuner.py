"""hyperparam_tuner.py — RandomizedSearchCV on the top-N models from evaluation."""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from detect.feature_store import FeatureStore
from detect.models import MODEL_REGISTRY, NEEDS_SCALING

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter grids per model type
# ---------------------------------------------------------------------------
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "random_forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": ["balanced", "balanced_subsample"],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "min_samples_leaf": [1, 2, 4],
    },
    "extra_trees": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    "logistic_regression": {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs", "saga"],
    },
    "svm_rbf": {
        "clf__C": [0.1, 1.0, 10.0, 100.0],
        "clf__gamma": ["scale", "auto", 0.01, 0.001],
    },
    "svm_linear": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    },
    "mlp_small": {
        "clf__hidden_layer_sizes": [(64, 32), (128, 64), (64,)],
        "clf__alpha": [0.0001, 0.001, 0.01],
        "clf__learning_rate_init": [0.001, 0.01],
    },
    "mlp_medium": {
        "clf__hidden_layer_sizes": [(128, 64, 32), (256, 128, 64), (128, 64)],
        "clf__alpha": [0.0001, 0.001, 0.01],
        "clf__learning_rate_init": [0.001, 0.005],
    },
    "knn_5": {
        "clf__n_neighbors": [3, 5, 7, 9, 11],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["minkowski", "euclidean"],
    },
    "knn_11": {
        "clf__n_neighbors": [7, 11, 15, 21],
        "clf__weights": ["uniform", "distance"],
    },
    "xgboost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    "lightgbm": {
        "n_estimators": [100, 200, 300],
        "max_depth": [-1, 5, 10],
        "learning_rate": [0.05, 0.1, 0.2],
        "num_leaves": [31, 63, 127],
    },
    "adaboost": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.5, 1.0, 1.5],
    },
    "decision_tree": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    },
    "lda": {
        "clf__solver": ["svd", "lsqr"],
        "clf__shrinkage": [None, "auto", 0.1, 0.5],
    },
    "sgd": {
        "clf__alpha": [0.0001, 0.001, 0.01],
        "clf__l1_ratio": [0.0, 0.15, 0.5, 1.0],
        "clf__penalty": ["l2", "elasticnet"],
    },
}


def _build_param_grid_for_pipeline(model_name: str, in_pipeline: bool) -> dict[str, list[Any]]:
    """Prefix param grid keys with 'clf__' if model is in a Pipeline."""
    grid = PARAM_GRIDS.get(model_name, {})
    if not in_pipeline:
        # Strip any clf__ prefixes (for non-pipeline models)
        return {k.replace("clf__", ""): v for k, v in grid.items()}
    # Add clf__ prefix to keys that don't already have it
    prefixed = {}
    for k, v in grid.items():
        prefixed[k if k.startswith("clf__") else f"clf__{k}"] = v
    return prefixed


def tune_top_models(
    label_col: str = "label_loose",
    top_n: int = 3,
    n_iter: int = 20,
    cv_folds: int = 5,
    *,
    baseline_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    feature_store: FeatureStore | None = None,
) -> list[dict[str, Any]]:
    """Run RandomizedSearchCV on the top-N models from baseline evaluation.

    Parameters
    ----------
    label_col:
        Label column used during evaluation.
    top_n:
        Number of top models to tune (by F1).
    n_iter:
        Number of parameter combinations to try per model.
    cv_folds:
        Folds for inner CV during tuning.
    baseline_dir:
        Directory containing ``results_{label_col}.csv``.
    output_dir:
        Directory to save tuned model artefacts.
    feature_store:
        Optional pre-built :class:`FeatureStore`.

    Returns
    -------
    List of dicts with tuning results for each model.
    """
    import pandas as pd

    repo_root = _find_repo_root()
    baseline_dir = Path(baseline_dir) if baseline_dir else repo_root / "outputs/ml_baseline"
    output_dir = Path(output_dir) if output_dir else repo_root / "outputs/ml_baseline/tuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = baseline_dir / f"results_{label_col}.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found at {results_path}. Run evaluator first.")

    leaderboard = pd.read_csv(results_path)
    top_models = leaderboard.dropna(subset=["f1_mean"]).head(top_n)["model"].tolist()
    logger.info("Tuning top-%d models: %s", top_n, top_models)

    store = feature_store or FeatureStore()
    X_df, y, feature_names = store.get_feature_matrix(label_col=label_col)
    X = X_df.values
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    tuning_results: list[dict[str, Any]] = []

    for model_name in top_models:
        if model_name not in MODEL_REGISTRY:
            logger.warning("Model '%s' not in registry — skipping tuning", model_name)
            continue

        logger.info("Tuning '%s' with %d iterations …", model_name, n_iter)
        estimator = copy.deepcopy(MODEL_REGISTRY[model_name])
        in_pipeline = model_name in NEEDS_SCALING

        if in_pipeline:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
        else:
            pipe = estimator

        param_grid = _build_param_grid_for_pipeline(model_name, in_pipeline)
        if not param_grid:
            logger.warning("No param grid for '%s' — skipping tuning", model_name)
            continue

        try:
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring="f1",
                cv=cv,
                random_state=42,
                n_jobs=-1,
                verbose=0,
                refit=True,
            )
            search.fit(X, y.values)

            result = {
                "model": model_name,
                "best_f1": float(search.best_score_),
                "best_params": search.best_params_,
            }
            tuning_results.append(result)
            logger.info("  %s best F1=%.4f  params=%s", model_name, result["best_f1"], result["best_params"])

            # Save tuned model
            model_path = output_dir / f"tuned_{model_name}_{label_col}.joblib"
            joblib.dump(
                {"model": search.best_estimator_, "feature_names": feature_names, "model_name": model_name},
                model_path,
            )
            logger.info("  Saved tuned model → %s", model_path)

        except Exception as exc:
            logger.error("  Tuning '%s' failed: %s", model_name, exc, exc_info=True)
            tuning_results.append({"model": model_name, "error": str(exc)})

    # Save summary
    summary_path = output_dir / f"tuning_summary_{label_col}.json"
    summary_path.write_text(json.dumps(tuning_results, indent=2, default=str), encoding="utf-8")
    logger.info("Tuning summary saved → %s", summary_path)
    return tuning_results


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parent.parent.parent
