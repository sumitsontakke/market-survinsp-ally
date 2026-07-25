"""Run-family inference and universe helpers.

Mirrors ``scripts/run_edge_level_experiment.py:_infer_family`` so the
new harness applies the same family labels and the locked-stress
evaluator can group runs identically.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

RUN_FAMILY_BENIGN = "benign"
RUN_FAMILY_CLIQUE = "clique"
RUN_FAMILY_RING = "ring"
RUN_FAMILY_MIXED = "mixed"
RUN_FAMILY_OTHER = "other"


def infer_run_family(run_name: str, scenario_summary: str = "") -> str:
    """Map a run identifier into one of the five families.

    Logic ports directly from the original Phase 1 inference (see
    ``scripts/run_edge_level_experiment.py:_infer_family``) so locked-
    stress family aggregations remain comparable.
    """
    text = f"{run_name} {scenario_summary}".lower()
    if "benign" in text or "generic" in text:
        return RUN_FAMILY_BENIGN
    if "mixed" in text:
        return RUN_FAMILY_MIXED
    if "ring" in text:
        return RUN_FAMILY_RING
    if "clique" in text:
        return RUN_FAMILY_CLIQUE
    return RUN_FAMILY_OTHER
