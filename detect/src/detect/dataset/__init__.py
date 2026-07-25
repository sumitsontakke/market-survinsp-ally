"""Dataset layer - cohort resolution, loading, splitting.

Cont, R. (2001). Empirical properties of asset returns. Quantitative Finance, 1(2), 223-236.
"""
from detect.dataset.loader import Run, load_run, list_runs_in_cohort  # noqa: F401
from detect.dataset.splitter import iter_run_holdout_split, iter_grouped_kfold  # noqa: F401
from detect.dataset.universe import infer_run_family, RUN_FAMILY_BENIGN  # noqa: F401

__all__ = [
    "Run", "load_run", "list_runs_in_cohort",
    "iter_run_holdout_split", "iter_grouped_kfold",
    "infer_run_family", "RUN_FAMILY_BENIGN",
]
