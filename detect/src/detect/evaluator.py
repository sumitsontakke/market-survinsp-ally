"""evaluator.py — Cross-validated evaluation of all registered ML models."""
from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from detect.feature_store import FeatureStore
from detect.models import MODEL_REGISTRY, NEEDS_SCALING

logger = logging.getLogger(__name__)


def _make_pipeline(name: str, estimator: Any) -> Any:
    """Wrap estimator in a StandardScaler pipeline if the model needs it."""
    if name in NEEDS_SCALING:
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    return estimator


def _evaluate_model(
    name: str,
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> dict[str, Any]:
    """Run stratified K-fold CV for one model; return metrics dict."""
    pipe = _make_pipeline(name, copy.deepcopy(estimator))
    metrics_per_fold: list[dict[str, float]] = []
    confusion_agg = np.zeros((2, 2), dtype=int)
    fit_times: list[float] = []
    predict_times: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Skip fold if only one class in train or test
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logger.debug("  fold %d skipped: only one class present", fold_idx + 1)
            continue

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        y_pred = pipe.predict(X_test)
        predict_times.append(time.perf_counter() - t0)

        # Probability for AUC (fall back to decision_function or binary pred)
        try:
            proba = pipe.predict_proba(X_test)
            # Handle edge case where only one class seen at fit time
            if proba.shape[1] < 2:
                y_prob = proba[:, 0]
            else:
                y_prob = proba[:, 1]
        except AttributeError:
            try:
                y_prob = pipe.decision_function(X_test)
            except AttributeError:
                y_prob = y_pred.astype(float)

        fold_metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        # ROC-AUC and PR-AUC require at least 2 classes in the test fold
        if len(np.unique(y_test)) > 1:
            fold_metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
            fold_metrics["pr_auc"] = float(average_precision_score(y_test, y_prob))
        else:
            fold_metrics["roc_auc"] = float("nan")
            fold_metrics["pr_auc"] = float("nan")

        metrics_per_fold.append(fold_metrics)
        confusion_agg += confusion_matrix(y_test, y_pred, labels=[0, 1])

        logger.debug("  fold %d: F1=%.4f ROC-AUC=%.4f", fold_idx + 1, fold_metrics["f1"], fold_metrics["roc_auc"])

    # If all folds were skipped, return NaN row
    if not metrics_per_fold:
        return {
            "model": name,
            "f1_mean": float("nan"),
            "f1_std": float("nan"),
            "roc_auc_mean": float("nan"),
            "roc_auc_std": float("nan"),
            "pr_auc_mean": float("nan"),
            "pr_auc_std": float("nan"),
            "accuracy_mean": float("nan"),
            "accuracy_std": float("nan"),
            "precision_mean": float("nan"),
            "precision_std": float("nan"),
            "recall_mean": float("nan"),
            "recall_std": float("nan"),
            "fit_time_mean": float("nan"),
            "predict_time_mean": float("nan"),
            "confusion_matrix": [[0, 0], [0, 0]],
            "n_folds": cv.n_splits,
            "error": "all_folds_skipped_single_class",
        }

    # Aggregate
    metric_names = list(metrics_per_fold[0].keys())
    agg: dict[str, float] = {}
    for m in metric_names:
        vals = [fm[m] for fm in metrics_per_fold if not np.isnan(fm[m])]
        agg[f"{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{m}_std"] = float(np.std(vals)) if vals else float("nan")

    return {
        "model": name,
        **agg,
        "fit_time_mean": float(np.mean(fit_times)) if fit_times else float("nan"),
        "predict_time_mean": float(np.mean(predict_times)) if predict_times else float("nan"),
        "confusion_matrix": confusion_agg.tolist(),
        "n_folds": cv.n_splits,
    }


def run_full_evaluation(
    label_col: str = "label_loose",
    cv_folds: int = 5,
    output_dir: str | Path = "outputs/ml_baseline",
    *,
    feature_store: FeatureStore | None = None,
) -> pd.DataFrame:
    """Evaluate all registered models via stratified K-fold CV.

    Parameters
    ----------
    label_col:
        Target label column name.
    cv_folds:
        Number of folds for StratifiedKFold.
    output_dir:
        Directory to save CSV results, JSON summary, and best model.
    feature_store:
        Optional pre-built :class:`FeatureStore`; constructed from defaults if omitted.

    Returns
    -------
    Leaderboard ``DataFrame`` sorted by ``f1_mean`` descending.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    store = feature_store or FeatureStore()
    logger.info("Loading feature matrix (label=%s) …", label_col)
    X_df, y, feature_names = store.get_feature_matrix(label_col=label_col)
    X = X_df.values
    n_positive = int(y.sum())
    n_negative = int((y == 0).sum())
    logger.info("Dataset: %d samples, %d features, positives=%d (%.1f%%)",
                len(y), len(feature_names), n_positive, 100.0 * y.mean())

    # Degenerate dataset check: need at least 2 classes to run CV
    if n_positive == 0 or n_negative == 0:
        logger.warning(
            "Label '%s' has only one class (positives=%d, negatives=%d) — "
            "skipping evaluation (cannot run stratified CV).",
            label_col, n_positive, n_negative,
        )
        empty_df = pd.DataFrame(columns=["model", "f1_mean", "roc_auc_mean"])
        csv_path = output_dir / f"results_{label_col}.csv"
        empty_df.to_csv(csv_path, index=False)
        summary = {
            "label_col": label_col,
            "cv_folds": cv_folds,
            "n_samples": int(len(y)),
            "n_features": int(len(feature_names)),
            "n_positive": n_positive,
            "positive_rate": float(y.mean()),
            "models_evaluated": 0,
            "best_model": "",
            "best_f1_mean": 0.0,
            "best_roc_auc_mean": 0.0,
            "feature_names": feature_names,
            "skipped": True,
            "skip_reason": "degenerate_label_single_class",
        }
        json_path = output_dir / f"summary_{label_col}.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[evaluator] Skipped {label_col}: only one class present (positives={n_positive}).")
        return empty_df

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results: list[dict[str, Any]] = []

    for name, estimator in MODEL_REGISTRY.items():
        logger.info("Evaluating %s …", name)
        try:
            row = _evaluate_model(name, estimator, X, y.values, cv)
            results.append(row)
            logger.info(
                "  %s — F1=%.4f±%.4f  ROC-AUC=%.4f±%.4f  fit=%.2fs",
                name,
                row["f1_mean"],
                row["f1_std"],
                row["roc_auc_mean"],
                row["roc_auc_std"],
                row["fit_time_mean"],
            )
        except Exception as exc:
            logger.error("  %s FAILED: %s", name, exc, exc_info=True)
            results.append({"model": name, "error": str(exc)})

    leaderboard = (
        pd.DataFrame(results)
        .sort_values("f1_mean", ascending=False)
        .reset_index(drop=True)
    )

    csv_path = output_dir / f"results_{label_col}.csv"
    leaderboard.to_csv(csv_path, index=False)
    logger.info("Saved results → %s", csv_path)

    # ---- Save summary JSON ----
    valid_rows = leaderboard.dropna(subset=["f1_mean"])
    best_row = valid_rows.iloc[0].to_dict() if not valid_rows.empty else {}
    summary = {
        "label_col": label_col,
        "cv_folds": cv_folds,
        "n_samples": int(len(y)),
        "n_features": int(len(feature_names)),
        "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()),
        "models_evaluated": len(results),
        "best_model": best_row.get("model", ""),
        "best_f1_mean": float(best_row.get("f1_mean", 0.0)),
        "best_roc_auc_mean": float(best_row.get("roc_auc_mean", 0.0)),
        "feature_names": feature_names,
    }
    json_path = output_dir / f"summary_{label_col}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved summary → %s", json_path)

    # ---- Save best model ----
    if best_row:
        best_name = best_row["model"]
        best_estimator = copy.deepcopy(MODEL_REGISTRY[best_name])
        best_pipe = _make_pipeline(best_name, best_estimator)
        logger.info("Training best model '%s' on full dataset …", best_name)
        best_pipe.fit(X, y.values)
        model_path = output_dir / f"best_model_{label_col}.joblib"
        joblib.dump({"model": best_pipe, "feature_names": feature_names, "model_name": best_name}, model_path)
        logger.info("Saved best model → %s", model_path)

    # ---- Print leaderboard ----
    display_cols = [
        c for c in ("model", "f1_mean", "roc_auc_mean", "pr_auc_mean",
                     "accuracy_mean", "precision_mean", "recall_mean", "fit_time_mean")
        if c in leaderboard.columns
    ]
    print(f"\n{'='*70}")
    print(f"ML LEADERBOARD  |  label={label_col}  |  {cv_folds}-fold CV")
    print(f"{'='*70}")
    print(leaderboard[display_cols].head(20).to_string(index=True, float_format=lambda x: f"{x:.4f}"))
    print(f"{'='*70}\n")

    return leaderboard
