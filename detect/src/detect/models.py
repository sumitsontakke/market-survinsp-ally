"""models.py — Registry of all ML classifiers to evaluate."""
from __future__ import annotations

import logging

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, object] = {
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200,
        random_state=42,
    ),
    "extra_trees": ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "adaboost": AdaBoostClassifier(
        n_estimators=200,
        random_state=42,
    ),
    "bagging": BaggingClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    ),
    "logistic_regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    ),
    "svm_rbf": SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42,
    ),
    "svm_linear": SVC(
        kernel="linear",
        class_weight="balanced",
        probability=True,
        random_state=42,
    ),
    "knn_5": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "knn_11": KNeighborsClassifier(n_neighbors=11, n_jobs=-1),
    "decision_tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
    ),
    "naive_bayes": GaussianNB(),
    "mlp_small": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42,
        early_stopping=True,
    ),
    "mlp_medium": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=500,
        random_state=42,
        early_stopping=True,
    ),
    "lda": LinearDiscriminantAnalysis(),
    "qda": QuadraticDiscriminantAnalysis(),
    "sgd": SGDClassifier(
        class_weight="balanced",
        random_state=42,
        loss="log_loss",
        max_iter=1000,
    ),
}

# Models that require StandardScaler (non-tree, distance/margin based)
NEEDS_SCALING: frozenset[str] = frozenset({
    "logistic_regression",
    "svm_rbf",
    "svm_linear",
    "knn_5",
    "knn_11",
    "mlp_small",
    "mlp_medium",
    "lda",
    "qda",
    "sgd",
})

# ---------------------------------------------------------------------------
# Optional: XGBoost
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier  # type: ignore[import-untyped]

    MODEL_REGISTRY["xgboost"] = XGBClassifier(
        n_estimators=300,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=1,  # will be overridden per-dataset
    )
    logger.info("XGBoost available — added to MODEL_REGISTRY")
except ImportError:
    logger.info("XGBoost not installed — skipping")

# ---------------------------------------------------------------------------
# Optional: LightGBM
# ---------------------------------------------------------------------------
try:
    from lightgbm import LGBMClassifier  # type: ignore[import-untyped]

    MODEL_REGISTRY["lightgbm"] = LGBMClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    logger.info("LightGBM available — added to MODEL_REGISTRY")
except ImportError:
    logger.info("LightGBM not installed — skipping")
