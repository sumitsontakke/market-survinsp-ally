# 51 · Phase J — Family-Disjoint Results

> Leave-one-manipulation-family-out cross-validation of the tier-2 GBM (bolt-on) and v4 GraphSAGE (end-to-end) detectors. The strictest generalisation test in the project. Answers the question: *does the trained detector transfer to manipulation families it has never seen?*

**Status:** complete.
**Related notes:** 45 (Phase G OOD baseline) · 46 (six engineered features) · 48 (tier-2 GBM) · 49 (v4 GraphSAGE) · 50 (Phase I ABIDES cross-generator).
**Related tasks:** #147 (tier-2 CPU pass), #148 (`MSA_PHASE_G_HOLDOUT_FAMILY` env var wiring), #149 (v4 retrains × 3), #150 (this note).

---

## Motivation

Phase I established that engineered features transfer across *data-generating processes* (v4 held AUC 0.842 on 15-run ABIDES cohort vs 0.518 for the two-feature baseline). Phase J asks a strictly harder question: do they transfer across *manipulation families themselves*?

Rationale: real surveillance environments do not see next quarter's manipulation strategies in this quarter's training data. A detector that generalises across parameter shift and across simulators may still fail when a new family emerges. The test protocol simulates that exact deployment risk by holding out one entire manipulation family at training time.

## Protocol

- Training cohort restricted to two of the three families {clique, ring, mixed}. Held-out family fully excluded from both train and validation splits.
- Test cohort restricted to the held-out family only.
- Both detectors evaluated on the same held-out cohort. No hyperparameter re-tuning per hold-out.
- Held-out family cycled through all three positions → three complete experiments.
- Tier-2 GBM: refit per hold-out over available runs (LOO-CV within the two-family training pool). Fast; each fit finishes in seconds on CPU.
- v4 GraphSAGE: retrained from scratch per hold-out. Three separate GPU trainer-gpu jobs on RTX 4090. Focal loss (γ=2, α=0.75), val-trader-recall early stopping (patience 6), 30-epoch budget.
- Wiring: `MSA_PHASE_G_HOLDOUT_FAMILY` environment variable filters the cohort loader (task #148). Same evaluator (`phase_g_eval.py`) as within-generator and ABIDES tests — no evaluation code changes.

## Results

| Held-out family | Tier-2 GBM (bolt-on) | v4 GraphSAGE (end-to-end) |
|---|---:|---:|
| Ring | **0.994** | 0.480 |
| Clique | **0.975** | 0.533 |
| Mixed | **0.906** | 0.374 |
| **Mean** | **0.958** | **0.462** |

Tier-2 holds AUC 0.91–0.99 across all three hold-outs. v4 collapses to 0.37–0.53 — indistinguishable from chance ranking, and in the mixed hold-out below it.

## Interpretation

The contrast is not marginal — it is roughly half an AUC. The mechanism is architectural, not a hyperparameter accident:

- **Tier-2 is trader-marginal.** The GBM scores each trader independently on the six engineered features. The features themselves are family-agnostic: counterparty concentration (φ₃, φ₅) and co-active count (φ₆) are elevated in *all three* families. When one family is held out, the GBM has already learnt that these features are broadly predictive on the remaining two, and applies that judgement uniformly at test time.
- **v4 does message-passing.** During training on two families the SAGEConv weights inevitably learn family-specific structural priors. For clique training data the prior is "high triangle density in the trader's local neighbourhood". For ring training data the prior is "the trader lies on a short cycle". These priors compose with the engineered features in the learnt representation.
- **When the held-out family is a ring**, the ring-specific structural prior never appears in the training data. The network's representation of a ring trader at test time is qualitatively different from anything it built representations for at training. Score distribution collapses toward the null.

This is not an overfitting story fixable by more regularisation. It is the price the network pays for composing engineered features with graph structure inside the message-passing layers. The bolt-on GBM does not compose — that is precisely why it generalises.

Retraining v4 with a wider validation pool or looser early-stopping tolerance (single training configuration reported here, patience 6 on a 3-graph val split) *might* recover partial family generalisation. That is speculation, not evidence; listed in dissertation Chapter 7 future work.

## Deployment Implication

For surveillance settings where the manipulation strategies observed in production may include patterns not seen at training time — which is the default rather than the exception in a real regulatory environment — the bolt-on stack is the more robust deployment choice. This flips the conventional graph-learning intuition that end-to-end always dominates.

The recommended hybrid pattern:
- **Tier-2 as primary detector.** Robust across novel families. Explainable (feature importances). Trader-marginal scoring is a match for investigator workload.
- **v4 as within-family confirmation layer.** Higher recall than tier-2 on known families (v4: 83% recall @ 74% purity vs tier-2: 62% recall @ 99% purity on within-generator OOD). Route a flagged trader to v4 only when the manipulation family is already suspected from tier-2's flag.

This split is described qualitatively in dissertation Chapter 6.6 and conference paper v1 Section 8. It has not been prototyped as a two-model workflow — that is dissertation Chapter 7 future work item four.

## Reproducibility

- Model registry entries: `v4_family_disjoint_ring`, `v4_family_disjoint_clique`, `v4_family_disjoint_mixed`. Each carries training config hash, git SHA, best-epoch, and per-epoch validation trajectory.
- Tier-2 runs: reproduce by setting `MSA_PHASE_G_HOLDOUT_FAMILY` and calling `phase_g_eval.py` with the tier-2 checkpoint. No retraining needed for the ring / clique holdouts on the fast CPU LOO path.
- v4 retrains: three invocations of `run_m3_boosted.py` with `MSA_PHASE_G_HOLDOUT_FAMILY` set. ~1.6 seconds per training day on the 4090 (patience 6 fires around epoch 8 due to small val cohort).
- Docker: `calibration_service/docker-compose.yml` → `trainer-gpu` service with `MSA_PHASE_G_HOLDOUT_FAMILY` in `environment`. `phase_g_eval.py` handles the `_list_ood_runs()` filter that requires `orders.csv` + `scenarios.csv` presence.

## Limitations

- Single v4 training configuration per hold-out. A more forgiving early-stop or a larger val split might tell a different story on family-disjoint. Reported result is one point in that space, not a full sweep.
- Three families only (clique, ring, mixed). Cannot claim behaviour across the full manipulation taxonomy (wash trades between two accounts, spoofing, layering, marking-the-close remain out of scope).
- Calibrated synthetic substrate only. No labelled real NSE data was used at any stage of Phase J. The architectural contrast is a claim about behaviour on synthetic substrates; whether it holds on production data is an open empirical question (dissertation Chapter 7 future work item one).

## Where This Lands in the Broader Project

- Deck: `outputs/final_submission_drmilan_deck.pptx` slide 6 shows the grouped-bar chart.
- Dissertation: Chapter 6 Section 6.4 (table 6.4) and Section 6.6 (discussion). Figure 6.2 renders the same chart as the deck.
- Papers: paper v1 Section 7 (family-disjoint) + Section 8 (interpretation); paper v2 Section 6 (empirical validation) + Section 8 (why bolt-on beats end-to-end on family shift).
- Streamlit dashboard: `Metric_Timeline` page — the "Family-disjoint" panel and the milestone card for Tier-2/v4.

---

*Note closed. Task #150 complete. Phase J complete.*
