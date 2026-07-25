# 50 — Phase I / ABIDES Cross-Generator Test

> The truly-unseen-distribution test, now at statistical power.
> Pairs with [[Research Notes/49 Phase H Path A2 Feature-Augmented GraphSAGE|49]]
> (the v4 model evaluated here) and [[Research Notes/43 ABIDES Integration|43]]
> (the cross-generator substrate). Updated 30 May 2026 with the
> expanded_v1 cohort (n=15) replacing the pilot directional finding (n=3).

## TL;DR

The Phase G OOD eval (notes 45, 49) tested generalisation across *seeds
and parameters*, but every run came from the same Phase G synthesiser.
Phase I tests generalisation across **data-generating processes**:
evaluate v1 (2-feature) and v4 (8-feature, Phase H winner) on cohorts
whose manipulators come from the **ABIDES discrete-event simulator** with
our CollusiveCliqueAgent / RingTraderAgent / FrontAccountAgent (note 43),
*not* the Phase G synth.

Headline result on the expanded cohort (n=15, five seeds × three
families × one calibration date, 2026-04-15):

| Metric (ABIDES expanded_v1, n=15) | v1 (2-feat)         | v4 (8-feat)        |
|------------------------------------|---------------------|---------------------|
| edge AUC (cv_auc)                  | 0.518 (chance)      | **0.842**           |
| clique trader recall (mean ± sd)   | 0.867 ± 0.157       | **0.905 ± 0.113**   |
| ring trader recall                 | 0.864 ± 0.025       | **1.000 ± 0.000**   |
| mixed trader recall                | 0.837 ± 0.071       | **0.983 ± 0.015**   |
| clique trader purity (aggregate)   | 0.096               | 0.094               |
| ring trader purity                 | 0.276               | 0.293               |
| mixed trader purity                | 0.249               | 0.269               |

Full four-cell comparison, Phase G OOD vs ABIDES expanded_v1:

| Metric                 | v1 OOD (n=50) | v4 OOD (n=50) | v1 ABIDES (n=15) | v4 ABIDES (n=15) |
|------------------------|---------------|---------------|------------------|------------------|
| edge AUC               | 0.638         | **0.984**     | 0.518            | **0.842**        |
| clique recall          | 0.406         | 0.810         | 0.860            | 0.898            |
| ring recall            | 0.480         | 0.928         | 0.865            | **1.000**        |
| mixed recall           | 0.408         | 0.745         | 0.836            | 0.982            |
| clique purity          | 0.113         | 0.774         | 0.096            | 0.094            |
| ring purity            | 0.136         | 0.729         | 0.276            | 0.293            |
| mixed purity           | 0.094         | 0.705         | 0.249            | 0.269            |

The pilot (n=3) directional finding holds with statistical power: **v1
collapses to chance on ABIDES; v4 retains real discriminative power.**
A +0.324 AUC-point gap across 15 independent runs is not noise — and
v4's AUC actually rose vs the pilot (0.795 → 0.842), so the small
cohort was if anything pessimistic.

## What the numbers say

**Strong claims, now backed by statistical power:**

- **The features carry signal across generators.** v4 edge AUC 0.842 on
  ABIDES vs v1's 0.518. v4 is doing real work; v1 is essentially random.
  The pilot showed this; expansion confirms it.

- **Ring manipulation is perfectly caught.** v4 recall on the five ring
  runs is 1.000 ± 0.000 — zero variance. Every ring manipulator in
  every ABIDES ring run was flagged.

- **Mixed is nearly as clean.** Mean v4 mixed recall is 0.983 ± 0.015
  across five runs (range 0.96-1.00). Tight, high.

- **Clique is the soft family.** Mean v4 clique recall 0.905 ± 0.113
  (range 0.69-1.00). Still strong on average; one of the five clique
  runs landed at 0.69, pulling the mean down. Worth a per-run drill-down
  if Dr Milan asks.

- **The decision threshold does not transfer.** v4 purity on ABIDES is
  0.09 - 0.29 — much lower than the 0.70 - 0.77 on Phase G OOD.
  Importantly, v1's ABIDES purity is essentially identical (0.10 - 0.28),
  which confirms this is a *cohort* property, not a v4 weakness — both
  models flag too many ABIDES benigns at their locked thresholds. Cross-
  generator deployment would need per-generator threshold recalibration.

## Honest caveats

- **In-sample threshold.** Same caveat as note 49: the locked threshold
  is tuned on the same cohort it is evaluated on. The cross-MODEL
  comparison (v1 vs v4) is fair; absolute purity numbers are mildly
  optimistic — which makes the cross-generator purity gap look larger
  rather than smaller.

- **ABIDES manipulators are not real NSE manipulators.** ABIDES is one
  step closer to a real distribution than the Phase G synthesiser
  (different microstructure, real order-book dynamics) but it is still
  synthetic. The next test would be a labelled real-data slice if one
  becomes available.

- **Family-uniform.** Like all Phase G work, clique/ring/mixed appear in
  both training and test. Family-disjoint generalisation is still open.

- **Clique recall variance is real.** 0.905 ± 0.113 across 5 runs has
  meaningful spread; one ABIDES clique configuration (seed 41) landed
  at 0.69. The 0.905 mean is robust but it isn't a tight result like
  ring or mixed.

## What this means for the dissertation

The chapter-6 generalisation claim now has two layers, both at
statistical power:

> **Conservative defensible claim:** on a 50-run out-of-distribution
> cohort with parameter-disjoint manipulation scenarios, the
> feature-augmented GraphSAGE detector (v4) catches ~83% of manipulators
> at ~74% purity (note 49). On a separate **15-run cross-generator
> cohort** whose manipulators come from a different agent-based simulator
> (ABIDES) entirely, the same model retains substantive discriminative
> power (**edge AUC 0.842 vs 0.518 for the 2-feature baseline**),
> demonstrating that the engineered features generalise across
> data-generating processes — though the decision threshold requires
> per-generator recalibration.

This addresses the strongest reviewer critique of the Phase G OOD
result — that it tests only seeds and parameters within one generator.
Phase I closes that loop with an honest result: *generalisation of
signal, not of operating point*, now confirmed across 15 cross-generator
runs.

## Methodology — pilot → expansion

The original Phase I pilot used 3 runs (one per family, single seed
2026-03-21) as a directional check (see prior version of this note).
The expansion adds 15 runs (5 seeds × 3 families × 1 calibration date,
2026-04-15) — deliberately disjoint from the pilot's seed and date to
guarantee independent draws. Same per-run config as the pilot
(num_traders=500, manipulators_per_run=6).

The pilot direction (v4 AUC 0.795 vs v1 0.504) is fully reproduced and
strengthened by the expansion (0.842 vs 0.518). For dissertation reporting,
the n=15 expanded numbers are the citable result; the pilot serves as
a methodology footnote showing the direction was clear from the start.

## How to reproduce

```
# 1. Generate the 15-run expanded cohort
docker compose -f docker-compose.abides.yml run --rm abides-synth \
    src/run_cohort.py \
    --cohort-name expanded_v1 \
    --families clique ring mixed \
    --seeds 23 29 31 37 41 \
    --calibration-dates 2026-04-15 \
    --num-traders 500 \
    --manipulators-per-run 6 \
    --out-root /srv/output

# 2. Run v4 + v1 eval on the expanded cohort
docker compose -f calibration_service/docker-compose.yml run --rm \
    -e MSA_PHASE_G_VARIANT=v4 \
    -e MSA_EVAL_COHORT=outputs/abides_runs/expanded_v1 \
    -e MSA_EVAL_RUN_PREFIX=R \
    -e MSA_EVAL_COHORT_TAG=abides_expanded_v1 \
    trainer-gpu python -u /app/training/phase_g_eval.py

# Same again with MSA_PHASE_G_VARIANT=v1 for the contrast.
```

Outputs:

  outputs/_phase_g_eval_results_v4_abides_expanded_v1.json
  outputs/_phase_g_eval_results_v1_abides_expanded_v1.json

phase_g_eval.py is cohort-configurable (env vars MSA_EVAL_COHORT /
RUN_PREFIX / COHORT_TAG) — the same script evaluates any cohort whose
runs share the standard schema (orders.csv + trades.csv + scenarios.csv).
Each ABIDES sim run produces both an adapted dir and an `*_abides_raw/`
sub-dir; the eval filter requires scenarios.csv so only the 15 adapted
dirs enter the metrics.

## What this does NOT settle

- **Family-disjoint generalisation.** Train clique+mixed, test ring
  (and reverse) is still open. Different question from cross-generator,
  flagged in notes 48 and 49.

- **Tier-2 stack on ABIDES.** The bolt-on GBM (note 48) was not
  evaluated on ABIDES here. Would need v1 trader scores pre-computed
  on the ABIDES cohort then a refit. Worth doing if the v4-vs-tier-2
  head-to-head on a new generator becomes a specific question.

- **Real NSE data.** Still the ultimate generalisation test.

## See also

- [[Research Notes/49 Phase H Path A2 Feature-Augmented GraphSAGE|49]] — the v4 model
- [[Research Notes/48 Tier-2 Classifier Results|48]] — the bolt-on alternative
- [[Research Notes/45 Phase G Generalization Pipeline|45]] — v1 lineage and OOD cohort
- [[Research Notes/43 ABIDES Integration|43]] — the cross-generator substrate
- [[Research Notes/40 LIMITATIONS Chapter 6|40]] — honest-read context
- `training/phase_g_eval.py` — the cohort-configurable evaluator
- `services/abides-synth/src/run_cohort.py` — the ABIDES cohort generator
- `outputs/abides_runs/pilot_v1/` — the 3-run pilot cohort
- `outputs/abides_runs/expanded_v1/` — the 15-run expanded cohort
- `outputs/_phase_g_eval_results_v4_abides_expanded_v1.json` — v4 numbers (the headline)
- `outputs/_phase_g_eval_results_v1_abides_expanded_v1.json` — v1 numbers
