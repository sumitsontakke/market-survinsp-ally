# market-survinsp-ally · Data Boundary Schema

> The on-disk contract between the `synth/` and `detect/` modules. Both modules read and write this format. Both the NSE-calibrated synthesizer and the ABIDES adapter emit conformant runs today.

**Schema version:** `0.1.0` (matches `manifest.json#schema_version` in the current codebase).
**Status:** stable for v1 release; roadmap for v0.2 → v1.0 at the bottom.
**Breaking-change policy:** minor bumps (`0.1.x` → `0.2.0`) for additive fields or reserved-field promotions; major bumps for renames, removals, or semantics changes.
**Reference implementation:** `services/abides-synth/src/adapters/exchange_log_to_msa.py` (current path — will move to `synth/src/synth/generator/exporter/` in the restructure).

---

## Cohort filesystem layout

A cohort is a directory. Each run is its own subdirectory of the cohort root, plus one cohort-level manifest:

```
<cohort_root>/
├── cohort_manifest.json                  (cohort-level spec + run inventory)
└── R<NN>_<generator>_<family>_s<seed>_<yyyymmdd>/
    ├── manifest.json                     (per-run manifest)
    ├── orders.csv                        (order stream, one row per order)
    ├── trades.csv                        (matched trades, one row per fill)
    ├── traders.csv                       (trader entity table)
    ├── accounts.csv                      (account entity table)
    ├── beneficial_owners.csv             (owner entity table)
    ├── brokers.csv                       (broker entity table)
    ├── instruments.csv                   (instrument reference data)
    ├── sessions.csv                      (trading-session reference data)
    └── scenarios.csv                     (scenario / manipulation-episode registry — this is the "labels" table)
```

Run-directory naming convention (recommended, not enforced by the schema):

```
R<NN>_<generator>_<family>_s<seed>_<yyyymmdd>
    │      │           │         │      │
    │      │           │         │      └─ Simulated trading date (YYYYMMDD)
    │      │           │         └─── RNG seed (matches manifest.json#config_hash context)
    │      │           └─────────────── Manipulation family (clique | ring | mixed | benign)
    │      └───────────────────────── Generator identifier (abides | msa | ext_*)
    └────────────────────────────────── Zero-padded run index inside the cohort
```

Consumers should treat the directory name as opaque and rely on `manifest.json#run_label`.

The `synth/` module's ABIDES pipeline additionally leaves a `<run_label>_abides_raw/` sibling directory (raw ABIDES exchange log) and a `<run_label>_manipulator.json` sidecar (the injection config that produced the run). These are **not** part of the boundary schema; they are ABIDES-specific artefacts the adapter keeps for provenance. Detectors do not read them.

---

## Cohort manifest (`cohort_manifest.json`)

Top-level cohort metadata. One file per cohort root.

| Field | Type | Required | Description |
|---|---|---|---|
| `spec.cohort_name` | string | yes | Human-readable cohort identifier (e.g. `"pilot_v1"`). |
| `spec.families` | array of string | yes | Union of manipulation families present. Subset of § Manipulation families vocabulary. |
| `spec.seeds` | array of integer | yes | RNG seeds used across the cohort. |
| `spec.calibration_dates` | array of ISO date | yes | Trading dates each run is calibrated to. |
| `spec.num_traders` | integer | yes | Target trader-population size per run. |
| `spec.manipulators_per_run` | integer | yes | Target count of manipulator agents per non-benign run. |
| `spec.out_root` | string | no | Original output root at generation time. Diagnostic only. |
| `runs` | array of string | yes | List of run subdirectory names present in the cohort. Consumers may cross-check against `os.listdir`. |

Example (from `outputs/abides_runs/pilot_v1/cohort_manifest.json`):

```json
{
  "spec": {
    "cohort_name": "pilot_v1",
    "families": ["clique", "ring", "mixed"],
    "seeds": [11],
    "calibration_dates": ["2026-03-21"],
    "num_traders": 500,
    "manipulators_per_run": 6,
    "out_root": "/srv/output/pilot_v1"
  },
  "runs": [
    "R01_abides_clique_s11_20260321",
    "R02_abides_ring_s11_20260321",
    "R03_abides_mixed_s11_20260321"
  ]
}
```

---

## Per-run manifest (`manifest.json`)

Per-run provenance and counts. One file per run.

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | string | yes | This schema's version. Currently `"0.1.0"`. |
| `generator_version` | string | yes | Version of the generator or adapter that produced this run. E.g., `"abides-adapter-0.1.0-phase-a"` or `"msa-synth-0.4.2"`. |
| `package_version` | string | no | Semver of the emitting package. Consumers should not depend on it beyond diagnostic display. |
| `config_hash` | string (hex) | yes | Deterministic hash of the full run configuration (generator params + agent population + injection params + seed). Two runs with the same `config_hash` must be bit-identical. |
| `generated_at` | string (ISO 8601 UTC) | yes | Wall-clock time the run was generated. |
| `run_label` | string | yes | Matches the containing directory name. |
| `data_source` | enum `{"abides", "msa", "ext_*"}` | yes | Which generator emitted this run. Determines expected `generator_version` shape. |
| `counts.brokers` | integer | yes | Row count in `brokers.csv`. |
| `counts.beneficial_owners` | integer | yes | Row count in `beneficial_owners.csv`. |
| `counts.accounts` | integer | yes | Row count in `accounts.csv`. |
| `counts.traders` | integer | yes | Row count in `traders.csv`. |
| `counts.instruments` | integer | yes | Row count in `instruments.csv`. |
| `counts.sessions` | integer | yes | Row count in `sessions.csv`. |
| `counts.orders` | integer | yes | Row count in `orders.csv`. |
| `counts.trades` | integer | yes | Row count in `trades.csv`. |
| `counts.scenarios` | integer | yes | Row count in `scenarios.csv`. |
| `scenario_types` | array of string | yes | Union of `scenario_type` values across `scenarios.csv`. Values from § Scenario-type vocabulary. |
| `scenario_ids` | array of string | yes | Union of `scenario_id` values. |
| `manipulative_order_count` | integer | yes | Count of `orders.csv` rows with `is_manipulative == True`. |
| `manipulative_trade_count` | integer | yes | Count of `trades.csv` rows with `is_manipulative == True`. |
| `entity_relationships` | object | no | Documentation of the entity hierarchy. Diagnostic; consumers may ignore. Currently: `{"beneficial_owner_to_account": "1..n", "account_to_trader": "1..n", "trader_to_order": "1..n", "order_to_trade": "0..n"}`. |
| `label_definitions` | object | no | Human-readable definitions of label values. Diagnostic. |

---

## Entity tables

The schema separates the entity hierarchy so surveillance features can join on realistic surveillance keys (broker overlap, beneficial-owner concentration, etc). One trader has one account, one account has one beneficial owner, one trader routes through one broker.

### `traders.csv`

The list of participants in the run.

| Column | Type | Required | Description |
|---|---|---|---|
| `trader_id` | string | yes | Stable within-run identifier (e.g., `"trader_00015"`). |
| `account_id` | string | yes | FK → `accounts.csv#account_id`. |
| `beneficial_owner_id` | string | yes | FK → `beneficial_owners.csv#owner_id`. Denormalised for convenient joins. |
| `broker_id` | string | yes | FK → `brokers.csv#broker_id`. |
| `trader_profile_id` | string | yes | Categorical profile tag emitted by the generator (e.g., `"abides_background"`, `"clique_alpha"`, `"noise_trader"`). |
| `risk_tier` | enum `{"low", "medium", "high"}` | yes | Onboarding risk tier. Currently `"medium"` for ABIDES background; `"high"` for injected manipulators. |
| `region` | string | yes | ISO 3166-1 alpha-2 country code. `"IN"` for NSE-calibrated runs. |
| `created_at` | string (ISO 8601) | yes | Trader account creation timestamp within the simulation clock. |
| `status` | enum `{"active", "suspended", "closed"}` | yes | Currently always `"active"` in v0.1.0. |

### `accounts.csv`

Trading accounts. A beneficial owner may hold multiple accounts (this is what enables front-account manipulation detection).

| Column | Type | Required | Description |
|---|---|---|---|
| `account_id` | string | yes | Stable identifier (e.g., `"account_00015"`). |
| `beneficial_owner_id` | string | yes | FK → `beneficial_owners.csv#owner_id`. |
| `opened_at` | string (ISO 8601) | yes | Account opening timestamp. |
| `status` | enum `{"active", "suspended", "closed"}` | yes | Currently always `"active"`. |

### `beneficial_owners.csv`

Underlying real-world entities. In v0.1.0 the synthesizer keeps this 1:1 with accounts; for front-account scenarios the generator sets multiple accounts to share an `owner_id`.

| Column | Type | Required | Description |
|---|---|---|---|
| `owner_id` | string | yes | Stable identifier (e.g., `"owner_00015"`). |
| `name` | string | yes | Synthetic name. Not PII. |
| `kyc_status` | enum `{"verified", "pending", "flagged"}` | yes | KYC state. Currently `"verified"` for background traders. |
| `region` | string | yes | ISO 3166-1 alpha-2. |
| `created_at` | string (ISO 8601) | yes | Owner registration timestamp. |

### `brokers.csv`

Executing brokers. In v0.1.0 the ABIDES adapter synthesises 20 brokers by default and round-robins traders across them.

| Column | Type | Required | Description |
|---|---|---|---|
| `broker_id` | string | yes | Stable identifier (e.g., `"broker_00019"`). |
| `name` | string | yes | Synthetic broker name. |
| `region` | string | yes | ISO 3166-1 alpha-2. |
| `registered_at` | string (ISO 8601) | yes | Broker registration timestamp. |
| `status` | enum `{"active", "suspended"}` | yes | Currently always `"active"`. |

### `instruments.csv`

The universe of traded instruments in this run. Currently always exactly one instrument.

| Column | Type | Required | Description |
|---|---|---|---|
| `instrument_id` | string | yes | Stable identifier (e.g., `"instrument_00001"`). |
| `symbol` | string | yes | Ticker symbol. |
| `asset_class` | string | yes | `"equity"` in v0.1.0. |
| `listing_venue` | string | yes | `"NSE"` for NSE-calibrated, `"ABIDES-SIM"` for ABIDES runs. |
| `currency` | string | yes | ISO 4217 (e.g., `"INR"`). |

### `sessions.csv`

Trading sessions inside the run. Currently always exactly one session (one trading day).

| Column | Type | Required | Description |
|---|---|---|---|
| `session_id` | string | yes | Stable identifier. |
| `instrument_id` | string | yes | FK → `instruments.csv#instrument_id`. |
| `session_date` | string (ISO 8601 date) | yes | Trading date. |
| `open_ts` | string (ISO 8601 datetime) | yes | Market open timestamp. |
| `close_ts` | string (ISO 8601 datetime) | yes | Market close timestamp. |

---

## Order & trade tables

### `orders.csv`

The order stream. Rows are sorted by `timestamp` ascending in emitting code, but consumers should not rely on order.

| Column | Type | Required | Description |
|---|---|---|---|
| `order_id` | integer | yes | Stable within-run identifier. |
| `timestamp` | string (ISO 8601 datetime) | yes | Order arrival time in simulation clock. |
| `trader_id` | string | yes | FK → `traders.csv#trader_id`. |
| `account_id` | string | yes | FK → `accounts.csv#account_id`. Denormalised. |
| `broker_id` | string | yes | FK → `brokers.csv#broker_id`. Denormalised. |
| `instrument_id` | string | yes | FK → `instruments.csv#instrument_id`. |
| `side` | enum `{"buy", "sell"}` | yes | Order side. |
| `order_type` | enum `{"limit", "market"}` | yes | Order type. |
| `price` | number | yes | Limit price. Present even for market orders (equal to the reference price at arrival). |
| `quantity` | integer | yes | Order quantity. Positive. |
| `time_in_force` | enum `{"day", "ioc", "gtc"}` | yes | Time-in-force. Currently only `"day"` in v0.1.0. |
| `scenario_id` | string | yes | FK → `scenarios.csv#scenario_id`. Every order is attributed to a scenario, benign or manipulative. |
| `scenario_label` | string | yes | Denormalised `scenarios.csv#scenario_label`. |
| `scenario_type` | string | yes | Denormalised `scenarios.csv#scenario_type`. |
| `is_manipulative` | boolean | yes | `True` iff the order was emitted by an injected manipulator agent. |
| `parent_order_id` | integer or empty | no | For split / iceberg parent-child order chains. Empty in v0.1.0. |
| `remaining_quantity` | integer | yes | Unfilled quantity when the order was cancelled or the day ended. |

### `trades.csv`

Matched trades (fills). One row per execution.

| Column | Type | Required | Description |
|---|---|---|---|
| `trade_id` | integer | yes | Stable within-run identifier. |
| `timestamp` | string (ISO 8601 datetime) | yes | Match time. |
| `buy_order_id` | integer | yes | FK → `orders.csv#order_id` of the buy leg. |
| `sell_order_id` | integer | yes | FK → `orders.csv#order_id` of the sell leg. |
| `buy_trader_id` | string | yes | FK → `traders.csv#trader_id`. Denormalised. |
| `sell_trader_id` | string | yes | FK → `traders.csv#trader_id`. Denormalised. |
| `instrument_id` | string | yes | FK → `instruments.csv#instrument_id`. |
| `price` | number | yes | Execution price. |
| `quantity` | integer | yes | Executed quantity. |
| `scenario_id` | string | yes | Attributed to the scenario of the initiating (aggressor) order. |
| `scenario_label` | string | yes | Denormalised. |
| `scenario_type` | string | yes | Denormalised. |
| `is_manipulative` | boolean | yes | `True` iff either leg is manipulative. |

---

## `scenarios.csv` — the labels table

Every run partitions its activity into one or more *scenarios*. There is always at least a `"normal"` scenario for background activity; manipulator injections add one scenario per injected pattern. **This table is the ground-truth "labels" surface for the detectors.**

| Column | Type | Required | Description |
|---|---|---|---|
| `scenario_id` | string | yes | Stable identifier (e.g., `"normal"`, `"scenario_clique_001"`, `"scenario_ring_002"`). |
| `scenario_label` | string | yes | Human-readable label (e.g., `"normal"`, `"clique_alpha_0"`, `"ring_beta_2"`). |
| `scenario_type` | string | yes | Value from § Scenario-type vocabulary. |
| `start_ts` | string (ISO 8601) | yes | Scenario window start. |
| `end_ts` | string (ISO 8601) | yes | Scenario window end. |
| `manipulator_count` | integer | yes | Number of trader-agents participating in this scenario. `0` for the background scenario. |

Example:

```csv
scenario_id,scenario_label,scenario_type,start_ts,end_ts,manipulator_count
normal,normal,generic_background,2026-03-21T09:30:00,2026-03-21T10:30:00,0
scenario_clique_001,clique_alpha_0,collusive_clique,2026-03-21T09:30:00,2026-03-21T10:30:00,6
```

### Deriving per-trader labels

Detectors typically want a per-trader boolean `is_manipulator`. Derive it as:

```python
manipulator_trader_ids = (
    orders.loc[orders["is_manipulative"], "trader_id"].unique()
)
labels = traders.assign(
    is_manipulator = traders["trader_id"].isin(manipulator_trader_ids)
)
```

There is no separate `labels.csv` — the per-trader label falls out of `orders.csv#is_manipulative` joined back through `trader_id`. This is deliberate: it keeps the ground truth authoritative in one place (the order stream) and avoids the possibility of a labels file drifting out of sync.

The manipulation *family* for a labelled trader is `scenario_type` on any of their manipulative orders (all their manipulative orders will share the same scenario in v0.1.0).

---

## Scenario-type vocabulary

| Value | Meaning |
|---|---|
| `"generic_background"` | Benign activity from noise, value, momentum, and market-maker agents. Always present. `manipulator_count == 0`. |
| `"collusive_clique"` | A closed group of accounts trades disproportionately among themselves. Emitted by `CollusiveCliqueAgent`. |
| `"ring_trader"` | A closed cycle of accounts rotates ownership around the cycle without net position change. Emitted by `RingTraderAgent`. |
| `"front_account"` | One controller directs multiple nominally-independent accounts. Emitted by `FrontAccountAgent`. Multiple accounts share a `beneficial_owner_id`. |
| `"mixed"` | Two or more of the above families are present in the same run, potentially with overlapping traders. |

The high-level "manipulation family" used in cohort specs and paper results maps as follows:

| High-level family | Scenario types present |
|---|---|
| `clique` | `collusive_clique` (+ always `generic_background`) |
| `ring` | `ring_trader` |
| `mixed` | at least two of `collusive_clique`, `ring_trader`, `front_account` |
| `front` | `front_account` (currently emitted, not exercised in reported results) |
| `benign` | only `generic_background` |

---

## Manipulator sidecar (`<run_label>_manipulator.json`)

The generator writes the exact injection configuration alongside the run directory. This is **provenance, not schema** — detectors do not read this file. It is retained so that the injection can be replayed identically or audited.

Example (from `outputs/abides_runs/pilot_v1/R01_abides_clique_s11_20260321_manipulator.json`):

```json
{
  "cliques": [
    {"size": 6, "target_pct_move": 0.005, "num_actions": 3}
  ],
  "rings": [],
  "fronts": []
}
```

Fields are family-specific; the schema is documented in `synth/docs/manipulator_config.md` (to be written during the restructure).

---

## Validation

The synth module ships a `synth.validate` submodule that walks a cohort root and asserts conformance:

```
python -m synth.validate <cohort_root>
```

Checks include:

- `cohort_manifest.json` present and parses.
- Every directory listed in `cohort_manifest.runs` exists.
- Every run has all 10 canonical files and a well-formed `manifest.json`.
- Foreign-key integrity across the entity tables (every `trader_id` in `orders.csv` has a row in `traders.csv`, etc.).
- Count consistency (`manifest.counts.orders == len(orders.csv)`, etc.).
- `is_manipulative` on orders is consistent with `scenarios.csv#scenario_type`.
- No orphan scenarios (every scenario has at least one order).

CI runs the validator against a checked-in mini-cohort on every push.

---

## Roadmap: 0.1.x → 1.0.0

Fields reserved for future minor versions. Consumers should ignore unknown fields at their schema minor.

- **`0.2.0`** — Add `orders.csv#execution_ts` for latency modelling. Add `traders.csv#tags` (array-of-string) for downstream feature hooks. Add `manifest.json#git_sha` alongside `config_hash` for exact source-of-emitter provenance.
- **`0.3.0`** — Multi-instrument support: relax `instruments.csv` and `sessions.csv` to `1..n` rows, add per-instrument matching in `orders.csv`.
- **`1.0.0`** — Freeze the current column set as long-term-stable. Any future field addition is minor; any rename or removal ships `2.0.0`.

Contributors adding a schema field should:

1. Add the field to the appropriate emitter (`synth/src/synth/generator/exporter/`).
2. Update the validator (`synth/src/synth/validate/`).
3. Bump `manifest.json#schema_version` at the emitter.
4. Update this document.
5. Update the mini-cohort fixture in `tests/fixtures/`.

---

## Rationale for this shape

Two design goals drive the schema:

**1. The entity hierarchy is expressive enough for real surveillance features.** Front-account manipulation is detectable *only* if we can join accounts back to beneficial owners. Broker-collusion signals require the broker dimension. Squashing the model to trader+order (as many academic datasets do) would preclude those features.

**2. The ground truth lives on the order stream.** `orders.csv#is_manipulative` + `scenarios.csv` is the source of truth. Per-trader labels are derived, not stored. This makes it impossible for label metadata to drift out of sync with the order stream — a class of bug that plagues surveillance datasets in the wild.

Not covered by this schema (deliberate):

- **The internal representation the generator uses during simulation** (ABIDES `Kernel`, MSA `SimulationOrchestrator`, etc.). Private to the emitter.
- **The graph object detectors build during feature extraction** (PyG `Data`, NetworkX graph). Private to `detect/`.
- **The model registry format** on the detector side. Private to `detect/registry/`.
- **Detector output** (per-trader scores, flagged sets). Detectors are free to write to `<run_dir>/detect_<model_tag>/…` with their own conventions.
