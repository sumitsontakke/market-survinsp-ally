"""MetricBundle - the single dataclass returned by every evaluator.

Every rung's experiment writes a MetricBundle JSON to the registry. This
makes "Rung 3 vs Rung 4" a ``pd.DataFrame.merge`` in the comparison page,
not a parsing exercise.

A sentinel value of ``-1.0`` is used for any metric that could not be
computed (e.g. holdout family had no positive examples). Never silently
default to 0.0 - that would falsely register as "perfect alarm-rate" or
"zero recall achieved".

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# A sentinel for metrics that couldn't be computed honestly.
NOT_COMPUTED = -1.0


@dataclass(frozen=True)
class MetricBundle:
    """Standardized metrics for a single experiment run.

    The contract is read by ``training/registry/checkpoints.py`` and
    consumed by the webapp's Compare page.
    """

    # ---- in-distribution (cross-validation on training cohort) -------
    cv_f1: float
    cv_auc: float
    cv_precision: float
    cv_recall: float
    cv_threshold: float

    # ---- locked stress (heldout runs) --------------------------------
    locked_clique_recall: float
    locked_ring_recall: float
    locked_mixed_recall: float
    locked_benign_alarm: float
    locked_per_run: dict[str, float] = field(default_factory=dict)

    # ---- locked stress: precision-side (purity = TP/(TP+FP)) ----------
    # Added Phase F: the surveillance auditor's question is "of the
    # traders you flagged, how many were actually manipulating?"
    # Recall alone overstates model usefulness on imbalanced cohorts.
    # Default to NOT_COMPUTED so older registry entries deserialize.
    locked_clique_purity: float = NOT_COMPUTED
    locked_ring_purity: float = NOT_COMPUTED
    locked_mixed_purity: float = NOT_COMPUTED

    # ---- provenance ---------------------------------------------------
    n_train_runs: int = 0
    n_eval_runs: int = 0
    n_edges_train: int = 0
    n_pos_edges_train: int = 0
    seed: int = 42

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetricBundle":
        d = dict(payload)
        if isinstance(d.get("locked_per_run"), str):
            d["locked_per_run"] = json.loads(d["locked_per_run"])
        # Forward compatibility: drop unknown keys so the registry can read
        # entries from older / newer schema versions without error.
        import dataclasses as _dc
        known = {f.name for f in _dc.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)

    def write_json(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def read_json(cls, path: Path | str) -> "MetricBundle":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    # convenience for MetricBundle math
    # ------------------------------------------------------------------
    def diff(self, other: "MetricBundle") -> dict[str, float]:
        """Per-field delta (self - other) for the comparison page."""
        out: dict[str, float] = {}
        for field_name in (
            "cv_f1", "cv_auc", "cv_precision", "cv_recall", "cv_threshold",
            "locked_clique_recall", "locked_ring_recall",
            "locked_mixed_recall", "locked_benign_alarm",
        ):
            a = getattr(self, field_name)
            b = getattr(other, field_name)
            out[field_name] = float(a - b) if (a >= 0 and b >= 0) else NOT_COMPUTED
        return out

    def summary_line(self) -> str:
        """Single-line representation for log streaming."""
        return (
            f"cv_f1={self.cv_f1:.3f} auc={self.cv_auc:.3f} | "
            f"locked: ring={self.locked_ring_recall:.3f} "
            f"mixed={self.locked_mixed_recall:.3f} "
            f"clique={self.locked_clique_recall:.3f} "
            f"benign_alarm={self.locked_benign_alarm:.3f} | "
            f"purity: clique={self.locked_clique_purity:.3f} "
            f"ring={self.locked_ring_purity:.3f} "
            f"mixed={self.locked_mixed_purity:.3f} | "
            f"runs={self.n_train_runs}/{self.n_eval_runs} seed={self.seed}"
        )

    @classmethod
    def empty(cls, seed: int = 42) -> "MetricBundle":
        """All-sentinel instance for failed runs."""
        return cls(
            cv_f1=NOT_COMPUTED, cv_auc=NOT_COMPUTED,
            cv_precision=NOT_COMPUTED, cv_recall=NOT_COMPUTED,
            cv_threshold=NOT_COMPUTED,
            locked_clique_recall=NOT_COMPUTED,
            locked_ring_recall=NOT_COMPUTED,
            locked_mixed_recall=NOT_COMPUTED,
            locked_benign_alarm=NOT_COMPUTED,
            locked_clique_purity=NOT_COMPUTED,
            locked_ring_purity=NOT_COMPUTED,
            locked_mixed_purity=NOT_COMPUTED,
            seed=seed,
        )
