"""Split policies for the unified harness.

Two policies, used at different stages:

  * ``run_holdout``        - hold out specific run_ids for locked stress.
                             Critical: the holdout set is identical for
                             Rung 3 and Rung 4 reports.
  * ``scenario_stratified`` - StratifiedGroupKFold with run_id groups for
                             in-distribution CV F1.

Phase 1 already converged on this split structure (see
``scripts/run_edge_level_experiment.py``); we replicate it so the
"same dataset" contract holds across rungs.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def iter_run_holdout_split(
    run_ids: Sequence[str],
    holdout_runs: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Return (train_run_ids, eval_run_ids).

    Any holdout id that doesn't appear in ``run_ids`` is silently dropped
    so configs survive cohort changes (e.g. a regenerated R09 still
    appears under the same name).
    """
    holdout = set(holdout_runs)
    train = [r for r in run_ids if r not in holdout]
    evald = [r for r in run_ids if r in holdout]
    if not train or not evald:
        raise ValueError(
            "iter_run_holdout_split: empty train or eval set. "
            f"got train={len(train)}, eval={len(evald)}, "
            f"holdout cohort={holdout}"
        )
    return train, evald


def iter_grouped_kfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """StratifiedGroupKFold with deterministic shuffle.

    Yields ``(train_idx, val_idx)`` per fold.

    Note: ``groups`` should be the ``run_id`` of each row, so the same
    run never appears in both train and val within one fold.
    """
    unique_groups = np.unique(groups)
    eff_splits = min(int(n_splits), len(unique_groups))
    if eff_splits < 2:
        raise ValueError(
            f"iter_grouped_kfold: only {len(unique_groups)} groups, need >= 2 for CV"
        )
    cv = StratifiedGroupKFold(n_splits=eff_splits, shuffle=True, random_state=int(seed))
    yield from cv.split(X, y, groups)
