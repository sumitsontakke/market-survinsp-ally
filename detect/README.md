# detect — graph-neural-network manipulation detectors + investigator dashboard

**Companion module to [`synth/`](../synth).** Trains and evaluates three detector variants over any cohort that honours [`SCHEMA.md`](../SCHEMA.md): a baseline 2-feature GraphSAGE (v1), a Gradient-Boosting-Machine bolt-on over six engineered features (tier-2), and a feature-augmented 8-feature GraphSAGE (v4). Ships an investigator Streamlit dashboard with per-trader deep-dive and LLM-based justifications.

## What this module produces

- **Per-trader manipulation scores** written back to the cohort under `<run_dir>/detect_<model_tag>/scores.csv`.
- **Model-registry entries** at `<registry_root>/<experiment_id>/{config.yaml, metrics.json, checkpoint.pt}` per training run.
- **Evaluation reports** at `<registry_root>/<experiment_id>/eval_<regime>.json` per (checkpoint × regime) pair.
- **Investigator dashboard** served at `http://localhost:8505/` with pages for cohort selection, per-trader deep-dive, model timeline, and case export.

## Install

```bash
pip install -e .
```

For GPU training (v4 GraphSAGE retrain):

```bash
pip install -e ".[gpu]"
```

## Quick start

```bash
# Compute the six engineered features for every run in a cohort
python -m detect.cli features --cohort ../cohorts/demo

# Train the v1 baseline GraphSAGE
python -m detect.cli train --cohort ../cohorts/demo --model v1

# Train the tier-2 bolt-on GBM (leave-one-run-out CV)
python -m detect.cli train --cohort ../cohorts/demo --model tier2

# Retrain v4 (GraphSAGE on 8 features)
python -m detect.cli train --cohort ../cohorts/demo --model v4 --gpu

# Evaluate any trained checkpoint across all three OOD regimes
python -m detect.cli evaluate --checkpoint <path> --cohort ../cohorts/demo

# Launch the investigator dashboard
docker compose up webapp
open http://localhost:8505/Metric_Timeline
```

## Structure

```
detect/src/detect/
├── __init__.py
├── constants.py                 shared constants (feature names, palette)
├── features/
│   ├── engineered.py            six features (φ₁–φ₆)
│   └── projection.py            trader projection: 0.7·max + 0.3·top3
├── models/
│   ├── graphsage.py             v1 and v4 GraphSAGE architecture
│   └── tier2_gbm.py             sklearn GradientBoostingClassifier wrapper
├── training/
│   ├── driver.py                training loop entry point
│   ├── loops.py                 epoch loops with focal loss + val-trader-recall early stopping
│   └── hyperparam.py            sweep utilities
├── evaluation/
│   ├── metrics.py               AUC, recall, purity, coverage
│   ├── phase_g.py               within-generator OOD (n=50 protocol)
│   ├── abides.py                cross-generator ABIDES protocol
│   └── family_disjoint.py       leave-one-family-out protocol
├── registry/
│   ├── writer.py                model-registry writer
│   ├── reader.py                model-registry reader
│   └── download.py              pull checkpoints from Zenodo
├── dashboard/                   Streamlit investigator workbench
│   ├── app.py
│   ├── pages/
│   │   ├── 1_Data_Inventory.py
│   │   ├── 5_Synthetic_Runs.py
│   │   ├── 7_Demo_Flow.py
│   │   ├── 8_Compare.py
│   │   ├── 9_Demo_Review.py
│   │   ├── 10_Metric_Timeline.py
│   │   └── Phase_G_Investigation.py
│   └── static/
└── cli.py                       unified `python -m detect {features,train,evaluate,serve}`
```

## Dependencies

Core stack: PyTorch 2.x + PyTorch Geometric, scikit-learn, pandas, numpy, matplotlib, Streamlit, Plotly (interactive 3D subgraph view). The optional Ollama integration for LLM justifications gracefully degrades to a template when the server is not reachable.

Full pin list in [`pyproject.toml`](pyproject.toml). GPU stack (PyTorch with `sm_120` wheels for the Ada Lovelace architecture) documented separately in [`docker/trainer-gpu/Dockerfile`](docker/trainer-gpu/Dockerfile).

## Testing

```bash
pip install -e ".[test]"
pytest tests/
```

Ships with a mini fixture cohort under `tests/fixtures/` for feature-extraction and detector round-trip tests.

## Contributing

See the repository-level [`../docs/DEVELOPING.md`](../docs/DEVELOPING.md). Contributions that add a new detector variant or a new evaluation regime should include a fixture-based test and a docs update.

## License

MIT. See [`../LICENSE`](../LICENSE).
