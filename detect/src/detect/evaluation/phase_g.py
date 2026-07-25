"""Phase G — evaluate the final continual-learning checkpoint on the OOD test cohort.

Loads the last day's checkpoint from outputs/phase_g_state/, builds graph
arrays for every run in outputs/phase_g_test_ood/, projects edge probs to
trader scores via the same 0.7*max + 0.3*top3 projection, and reports:

  * cv_auc (edge-level on the test cohort, threshold not tuned)
  * per-family trader recall
  * per-family trader purity  (precision-side, added Phase F)
  * benign alarm rate (false-positive rate on non-manipulator traders)

Also registers the final model + numbers in /data/model_registry so the
Metric Timeline page picks it up.

Run inside the trainer-gpu container:
    docker compose run --rm trainer-gpu \\
        python -u /app/training/phase_g_eval.py
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

_APP_ROOT = os.environ.get("MSA_APP_ROOT", "/app")
sys.path.insert(0, _APP_ROOT)

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from detect.dataset.loader import load_run  # noqa: E402
from detect.evaluate.locked_stress import (  # noqa: E402
    evaluate_locked_stress, _graphs_to_edge_frame,
)
from detect.evaluate.metrics import MetricBundle, NOT_COMPUTED  # noqa: E402
from detect.features.pyg_builder import (  # noqa: E402
    AUGMENTED_NODE_FEATURE_NAMES,
    DEFAULT_NODE_FEATURE_NAMES,
    build_graph_arrays,
)
from detect.models.gnn_graphsage import (  # noqa: E402
    EdgeGraphSAGEModel, GraphSAGEConfig,
)
from detect.registry import ExperimentConfig, save_experiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Variant matches phase_g_continual.py — defaults to v2 (second pilot).
_VARIANT        = os.environ.get("MSA_PHASE_G_VARIANT", "v2")
# Phase J: same MSA_PHASE_G_HOLDOUT_FAMILY env var as the trainer. State
# dir picks up _no_{family} suffix so we load the matching checkpoint.
HOLDOUT_FAMILY  = os.environ.get("MSA_PHASE_G_HOLDOUT_FAMILY", "")
_HOLDOUT_SUFFIX = f"_no_{HOLDOUT_FAMILY}" if HOLDOUT_FAMILY else ""
STATE_ROOT      = Path(_APP_ROOT) / f"outputs/phase_g_state_{_VARIANT}{_HOLDOUT_SUFFIX}"

# Phase I (cross-generator test): cohort-configurable. Defaults preserve
# the Phase G OOD eval behaviour; set MSA_EVAL_COHORT / RUN_PREFIX /
# COHORT_TAG to evaluate any cohort whose runs share the standard schema
# (orders.csv, trades.csv, scenarios.csv) — e.g. the ABIDES pilot cohort.
OOD_COHORT_ROOT = Path(_APP_ROOT) / os.environ.get(
    "MSA_EVAL_COHORT", "outputs/phase_g_test_ood")
RUN_PREFIX      = os.environ.get("MSA_EVAL_RUN_PREFIX", "OOD_RUN")
COHORT_TAG      = os.environ.get("MSA_EVAL_COHORT_TAG", "")
_TAG_SUFFIX     = f"_{COHORT_TAG}" if COHORT_TAG else ""
OUT_JSON        = Path(_APP_ROOT) / (
    f"outputs/_phase_g_eval_results_{_VARIANT}{_HOLDOUT_SUFFIX}{_TAG_SUFFIX}.json")

# Path A2 (Phase H): v4 is the feature-augmented variant. Its checkpoint
# expects 8-dim node features, so the eval graphs must be built with the
# augmented feature set too. The test-graph cache is keyed by feature set
# AND cohort so 2-dim/8-dim and OOD/ABIDES graphs never collide.
_AUGMENTED = (_VARIANT == "v4")
NODE_FEATURE_NAMES = (AUGMENTED_NODE_FEATURE_NAMES if _AUGMENTED
                      else DEFAULT_NODE_FEATURE_NAMES)
_CACHE_BASE     = "outputs/_phase_g_test_graph_cache"
_CACHE_SUFFIX   = ("_aug" if _AUGMENTED else "") + _TAG_SUFFIX
GRAPH_CACHE     = Path(_APP_ROOT) / (_CACHE_BASE + _CACHE_SUFFIX)
SEED = 42


def _log(msg: str) -> None:
    print(msg, flush=True)


def _list_ood_runs() -> list[Path]:
    if not OOD_COHORT_ROOT.is_dir():
        return []
    # Require both orders.csv AND scenarios.csv. The ABIDES cohort's
    # *_abides_raw/ sub-dirs have orders.csv (raw ABIDES dump) but no
    # scenarios.csv and zero is_manipulative labels - including them
    # corrupted the false-positive count on the first cross-generator run.
    return sorted(d for d in OOD_COHORT_ROOT.iterdir()
                  if d.is_dir() and d.name.startswith(RUN_PREFIX)
                  and (d / "orders.csv").is_file()
                  and (d / "scenarios.csv").is_file())


def _load_or_build_graph(run_dir: Path):
    GRAPH_CACHE.mkdir(parents=True, exist_ok=True)
    pkl = GRAPH_CACHE / f"{run_dir.name}.pkl"
    if pkl.exists():
        with pkl.open("rb") as fh:
            return pickle.load(fh)
    run = load_run(run_dir)
    g = build_graph_arrays(run, sparsification="knn", k=12,
                           node_feature_names=NODE_FEATURE_NAMES)
    with pkl.open("wb") as fh:
        pickle.dump(g, fh)
    return g


def _find_final_checkpoint() -> Optional[Path]:
    if not STATE_ROOT.is_dir():
        return None
    ckpts = sorted(STATE_ROOT.glob("day_*_checkpoint.pt"))
    return ckpts[-1] if ckpts else None


def _rebuild_model(ckpt_path: Path, device: str, f_node: int,
                   f_edge: int) -> EdgeGraphSAGEModel:
    import torch
    blob = torch.load(str(ckpt_path), map_location=device)
    cfg_d = blob["config"]
    cfg = GraphSAGEConfig(
        node_in_dim=int(cfg_d["node_in_dim"]),
        edge_in_dim=int(cfg_d["edge_in_dim"]),
        hidden_dims=tuple(cfg_d["hidden_dims"]),
        dropout=float(cfg_d["dropout"]),
        aggregator=str(cfg_d["aggregator"]),
        head_hidden=int(cfg_d["head_hidden"]),
        head_dropout=float(cfg_d["head_dropout"]),
    )
    model = EdgeGraphSAGEModel(cfg, seed=int(blob.get("seed", SEED)),
                               device=device)
    model._build_module()
    model._module.load_state_dict(blob["state_dict"])
    model._module.eval()
    return model


def main() -> int:
    _log("=" * 78)
    _log("PHASE G — OOD test evaluation")
    _log("=" * 78)

    ckpt = _find_final_checkpoint()
    if ckpt is None:
        _log(f"FATAL: no checkpoint in {STATE_ROOT}")
        _log("  Run training/phase_g_continual.py first.")
        return 2
    _log(f"  final ckpt  : {ckpt.name}")

    ood_runs = _list_ood_runs()
    if not ood_runs:
        _log(f"FATAL: no OOD runs in {OOD_COHORT_ROOT}")
        _log("  Run scripts/regen_phase_g_test_ood.py + synthesize first.")
        return 3
    _log(f"  ood runs    : {len(ood_runs)}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"  device      : {device}")

    eval_graphs = []
    for rd in ood_runs:
        try:
            g = _load_or_build_graph(rd)
            eval_graphs.append(g)
        except Exception as exc:  # noqa: BLE001
            _log(f"  graph build failed for {rd.name}: {exc!r}")
    if not eval_graphs:
        _log("FATAL: no OOD graphs loaded")
        return 4

    f_node = eval_graphs[0].x.shape[1]
    f_edge = eval_graphs[0].edge_attr.shape[1]
    _log(f"  features    : node_in={f_node}  edge_in={f_edge}")

    model = _rebuild_model(ckpt, device, f_node, f_edge)
    _log(f"  model       : reloaded GraphSAGE max-aggregator from {ckpt.name}")

    t0 = time.perf_counter()
    eval_probs = [model.predict_proba(g) for g in eval_graphs]
    pred_secs = time.perf_counter() - t0

    # Edge-level AUC on the OOD cohort (informational; can be poor by design).
    pooled_y = np.concatenate([g.y.astype(int) for g in eval_graphs])
    pooled_p = np.concatenate(eval_probs)
    cv_auc = (float(roc_auc_score(pooled_y, pooled_p))
              if len(np.unique(pooled_y)) > 1 else NOT_COMPUTED)

    # Use the evaluator with the same cv_probs (= pooled_p) and labels (= y).
    metrics = evaluate_locked_stress(
        train_graphs=eval_graphs,
        eval_graphs=eval_graphs,
        cv_probs=pooled_p,
        cv_labels=pooled_y,
        eval_probs_per_graph=eval_probs,
        cv_auc=cv_auc,
        n_train_runs=0,
        n_eval_runs=len(eval_graphs),
        seed=SEED,
    )
    _log(f"  predicted   : {pred_secs:.1f}s for {len(eval_graphs)} runs")
    _log("")
    _log(f"  cv_auc                  : {cv_auc:.4f}")
    _log(f"  locked_clique_recall    : {metrics.locked_clique_recall:.3f}")
    _log(f"  locked_ring_recall      : {metrics.locked_ring_recall:.3f}")
    _log(f"  locked_mixed_recall     : {metrics.locked_mixed_recall:.3f}")
    _log(f"  locked_clique_purity    : {metrics.locked_clique_purity:.3f}")
    _log(f"  locked_ring_purity      : {metrics.locked_ring_purity:.3f}")
    _log(f"  locked_mixed_purity     : {metrics.locked_mixed_purity:.3f}")
    _log(f"  locked_benign_alarm     : {metrics.locked_benign_alarm:.4f}")
    _log("")
    _log("Comparison anchors (in-distribution):")
    _log("  M3+ champion clique=0.956 ring=0.896 mixed=0.905 (PHASE1_R01_R24)")
    _log("")
    _log("This Phase G evaluation is on the OOD cohort. If recalls hold near "
         "the M3+ headline, the model generalizes. If they collapse, the "
         "model was memorizing the training-cohort distribution.")

    # Persist eval JSON.
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "schema_version":   "1",
        "ckpt":             ckpt.name,
        "n_test_runs":      len(eval_graphs),
        "cv_auc":           cv_auc,
        "metrics":          metrics.to_dict(),
        "predicted_in_sec": pred_secs,
    }, indent=2), encoding="utf-8")
    _log(f"  results JSON : {OUT_JSON}")

    # Register final model.
    exp_id = f"phase_g_final__{ckpt.stem}"
    try:
        cfg = ExperimentConfig(
            experiment_id=exp_id,
            seed=SEED,
            model={
                "family":      "gnn",
                "kind":        "edge_graphsage",
                "hidden_dims": list(model.config.hidden_dims),
                "dropout":     float(model.config.dropout),
                "aggregator":  str(model.config.aggregator),
            },
            features={"sparsification": "knn", "k": 12},
            data={
                "cohort": "PHASE_G_TEST_OOD",
                "runs":   [d.name for d in ood_runs],
                "n_traders_per_run": "5000 (or as generated)",
                "test_cohort_dir": str(OOD_COHORT_ROOT),
            },
            calibration={"alpha_source": "calibrated_nse"},
            split={"policy": "ood_test", "n_test": len(eval_graphs)},
            loss={"type": "focal", "alpha": 0.85, "gamma": 2.0},
            evaluation={
                "projection": "0.7*max + 0.3*top3_mean",
                "early_stop": "val_trader_recall (Phase G)",
            },
            training={"protocol": "continual_warm_start"},
        )
        folder = save_experiment(
            cfg, metrics,
            notes=(f"Phase G final continual-learning model. "
                   f"Evaluated on OOD test cohort with held-out seeds AND "
                   f"held-out parameter ranges (duration 60-90, ratio 2-5%, "
                   f"core 16-25). Source ckpt: {ckpt.name}"),
        )
        _log(f"  registered  : {folder.name}")
    except Exception as exc:  # noqa: BLE001
        _log(f"  register failed: {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
