# 49 — Phase H / Path A2: Feature-Augmented GraphSAGE (v4)

> The Phase H headline. Pairs with [[Research Notes/48 Tier-2 Classifier Results|48]]
> (the bolt-on stack this is the end-to-end alternative to) and
> [[Research Notes/45 Phase G Generalization Pipeline|45]] (the v1 model
> lineage v4 retrains from).

## TL;DR

v4 is the v1 GraphSAGE architecture retrained from scratch with the six
engineered manipulation-signature features injected as GNN **node**
features — an 8-dim node vector (2 topology + 6 engineered) instead of
v1's 2-dim. Training knobs are identical to v1, so the only variable is
the feature set.

On the 50-run Phase G OOD cohort, v4 transforms generalization:

| Metric (OOD, 50 runs)  | v1 (2-feat) | v4 (8-feat) |   delta   |
|------------------------|-------------|-------------|-----------|
| edge AUC (cv_auc)      | 0.638       | **0.984**   | +0.346    |
| clique trader recall   | 0.406       | **0.810**   | +0.404    |
| ring trader recall     | 0.480       | **0.928**   | +0.448    |
| mixed trader recall    | 0.408       | **0.745**   | +0.337    |
| clique trader purity   | 0.113       | **0.774**   | +0.661    |
| ring trader purity     | 0.136       | **0.729**   | +0.593    |
| mixed trader purity    | 0.094       | **0.705**   | +0.611    |

Averaged across families, v4 catches **~83% of manipulators at ~74%
purity**; v1 caught ~43% at ~11% purity — i.e. v1 on the OOD cohort was
effectively unusable, with nine of every ten flags false. Same
architecture, same training protocol: the six engineered node features
did all of this.

## What changed vs v1

Exactly one thing — the node feature vector.

  v1 node features (2):  trader_total_volume, trader_unique_counterparties
  v4 node features (8):  the v1 two, plus burst_concentration,
                         side_entropy_in_burst, counterparty_hhi_burst,
                         order_qty_cov, top_partner_trade_share,
                         co_active_top_count

Everything else is held constant: GraphSAGE 2-layer max-aggregator,
hidden dims (256, 128, 64), focal loss (alpha 0.85, gamma 2.0),
val-trader-recall early stop (patience 6), 5-day continual warm-start,
seed 42. v4 trains **from scratch** — it cannot warm-start v1 because the
first SAGEConv weight changes shape (2 -> 8 inputs).

Wired as `MSA_PHASE_G_VARIANT=v4` in `phase_g_continual.py` /
`phase_g_eval.py`. The feature computation is shared with the tier-2
study via `training/features/engineered_core.py`, so v4's GNN inputs and
note 48's analysed features are provably the same code.

## v4 vs the tier-2 bolt-on stack

Note 48's tier-2 GBM (v1 score + 6 features) reached pooled trader-AUC
0.968 under leave-one-run-out CV. v4 puts the same six features *inside*
the GNN and reaches edge-AUC 0.984 on the OOD cohort, with trader-level
per-family recall 0.74-0.93 at purity 0.70-0.77.

The two are not a strict head-to-head — tier-2 used leave-one-run-out CV
with a held-out decision threshold; v4's eval trains on the separate
`phase_g_cohort` and tests on the OOD cohort, but tunes its locked
threshold in-sample on that OOD set. Directionally, though, Phase H
confirms note 48's hypothesis: **end-to-end feature learning matches or
beats the bolt-on stack, and does it as a single model** — no
second-stage classifier to train, persist or maintain. A GNN can exploit
feature-graph interactions (a low-HHI trader *connected to* other
low-HHI traders) that a trader-marginal GBM structurally cannot see.

## Honest caveats

- **Clique recall is bimodal.** v4's 0.810 clique average hides a split:
  13 of 18 clique runs score 0.65-1.0 (mostly above 0.85), but 5 clique
  runs are near-misses — RUN011 0.03, RUN030 0.09, RUN012 0.10,
  RUN018 0.12, RUN032 0.17. Some clique configuration evades v4 entirely.
  This deserves a drill-down before claiming uniform clique coverage.

- **In-sample threshold.** The locked decision threshold (0.6) is tuned
  on the OOD eval set — the same harness every v1/v2/v3 eval used, so the
  v4-vs-v1 comparison is fair, but the absolute operating point is mildly
  optimistic.

- **Family-uniform.** Like all Phase G work, clique/ring/mixed appear in
  both training and test cohorts. Family-disjoint generalization is still
  open.

- **Synthetic only.** Calibrated synthesizer + ABIDES substrate; no real
  NSE data yet.

## How to reproduce

```
cd calibration_service
docker compose run --rm -e MSA_PHASE_G_VARIANT=v4 trainer-gpu \
    python -u /app/training/phase_g_continual.py
docker compose run --rm -e MSA_PHASE_G_VARIANT=v4 trainer-gpu \
    python -u /app/training/phase_g_eval.py
```

Training: ~172 min on GPU (5-day continual warm-start), plus a one-time
augmented graph build cached to `outputs/_phase_g_graph_cache_aug/`.
Outputs:

  outputs/phase_g_state_v4/day_*_checkpoint.pt   per-day checkpoints
  outputs/_phase_g_eval_results_v4.json          OOD eval metrics

## What this means for the dissertation

Phase G's honest read (chapter 6, note 40) was that v1 generalized
poorly to the OOD cohort — edge-AUC 0.638, trader purity ~0.11. Phase H
closes that gap from the model side, not with a bolt-on:

> On a 50-run out-of-distribution cohort with parameter-disjoint
> manipulation scenarios, a feature-augmented GraphSAGE GNN — the v1
> architecture retrained with six explicit manipulation-signature node
> features — catches ~83% of manipulators at ~74% precision, versus the
> 2-feature v1's ~43% at ~11%.

The dissertation now has two converging Phase 3 results: the tier-2
stacked classifier (note 48) and the feature-augmented GNN (this note).
Both say the same thing — the engineered features carry the
generalization — and v4 is the cleaner single-model expression of it.

## See also

- [[Research Notes/48 Tier-2 Classifier Results|48]] — the bolt-on alternative
- [[Research Notes/45 Phase G Generalization Pipeline|45]] — the v1 model lineage
- [[Research Notes/46 Engineered Features + Investigation Page|46]] — feature methodology
- [[Research Notes/40 LIMITATIONS Chapter 6|40]] — the honest read this result lifts
- `training/features/engineered_core.py` — shared feature core
- `training/features/node_engineered.py` — GNN node-feature adapter
- `training/phase_g_continual.py` / `training/phase_g_eval.py` — the v4 variant
- `outputs/_phase_g_eval_results_v4.json` — the measured numbers
