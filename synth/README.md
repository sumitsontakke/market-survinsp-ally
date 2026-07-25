# synth — NSE-calibrated market synthesizer with ABIDES integration

**Companion module to [`detect/`](../detect).** Emits schema-conformant cohorts of synthetic trading days with configurable manipulation-family injection (clique / ring / mixed / front-account). Consumers include the sibling `detect/` module and any third-party surveillance research code that honours [`SCHEMA.md`](../SCHEMA.md).

## What this module produces

Per-cohort output tree:

```
runs/
├── R01_msa_clique_s42_20260321/
│   ├── orders.csv     traders.csv         accounts.csv       beneficial_owners.csv
│   ├── brokers.csv    instruments.csv     sessions.csv       scenarios.csv
│   ├── trades.csv     manifest.json
├── R02_msa_ring_s42_20260321/
│   └── ...
└── cohort_manifest.json
```

Ten canonical files per run plus a cohort-level manifest, exactly as documented in [`SCHEMA.md`](../SCHEMA.md).

## Install

```bash
pip install -e .
```

## Quick start

```bash
# Calibrate marginal distributions from NSE bhavcopy data (one-time per stock)
python -m synth.calibration.calibrate --stock RELIANCE --lookback-days 240

# Generate a demo cohort (3 runs, ~5 min on CPU)
python -m synth.cli generate --config ../configs/synth/demo_cohort.yaml --out ../cohorts/demo

# Generate a cross-generator ABIDES cohort
python -m synth.cli generate --config ../configs/synth/abides_cohort.yaml --out ../cohorts/abides

# Validate that a cohort is SCHEMA.md-conformant
python -m synth.validate ../cohorts/demo
```

## Structure

```
synth/src/synth/
├── __init__.py
├── constants.py              shared constants (schema version, defaults)
├── generator/                the NSE-calibrated MSA synthesizer
│   ├── behaviors/            per-agent order-generation strategies
│   ├── domain/               entities: Trader, Account, Order, Scenario, ...
│   ├── market/               matching-engine + orderbook simulation
│   ├── simulation/           run orchestrator
│   ├── exporters/            SCHEMA.md-conformant CSV/JSON writers
│   ├── analysis/             calibration diagnostics
│   ├── registry/             stock + scenario registries
│   ├── utils/                small helpers
│   └── wrappers/             high-level entry-point helpers
├── abides/                   ABIDES cross-generator integration
│   ├── agents/               CollusiveCliqueAgent, RingTraderAgent, FrontAccountAgent
│   ├── adapters/             ABIDES exchange-log → SCHEMA.md adapter
│   └── run_cohort.py         orchestrate an ABIDES cohort
├── calibration/              NSE bhavcopy fetch + marginal-distribution fit
│   ├── fetcher/              bhavcopy fetch
│   ├── calibrator/           marginal distribution fit
│   └── core/                 shared calibration data models
├── validate/                 SCHEMA.md conformance checker
└── cli.py                    unified `python -m synth {generate,validate,fetch,calibrate}`
```

## Dependencies

Standard scientific Python (numpy, pandas, scipy, pyyaml). The ABIDES sub-module depends on the vendored [ABIDES](https://github.com/abides-sim/abides) simulator (installed as a submodule at first `make install`). The calibration sub-module additionally uses `requests` for bhavcopy fetching.

Full pin list in [`pyproject.toml`](pyproject.toml).

## Testing

```bash
pip install -e ".[test]"
pytest tests/
```

Ships with a mini fixture cohort under `tests/fixtures/` for validator round-trip tests.

## Contributing

See the repository-level [`../docs/DEVELOPING.md`](../docs/DEVELOPING.md). Contributions that add a new manipulation-family agent should include a fixture run under `tests/fixtures/` and a schema-conformance test.

## License

MIT. See [`../LICENSE`](../LICENSE).
