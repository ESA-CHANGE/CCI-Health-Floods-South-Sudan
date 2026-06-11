# BayFloodGEN modular workflow

This package contains a modular implementation of the BayFloodGEN flood-scenario workflow for South Sudan health facilities.

## Module structure

- `bayflood_config.py`
  - Centralized configuration constants (sampling, scenarios, backend preferences).
- `bayflood_runtime.py`
  - Runtime/backend helpers (JAX device detection and sampler selection).
- `bayflood_data.py`
  - Data parsing and expansion utilities.
  - Converts input facility time-series into daily long-format records.
- `bayflood_model.py`
  - Bayesian model class (`PatchGENBayGEN`).
  - Model building, posterior sampling, diagnostics, and scenario generation.
- `bayflood_validation.py`
  - Validation plots for observed vs synthetic behavior.

## Recommended usage

Use the orchestrator entrypoint:

```bash
python /mnt/staas/CLICHE/01_SRC/bayflood_orchestrator.py
```

It will execute the full pipeline:
1. Load and parse input data.
2. Expand to daily records.
3. Build and sample the Bayesian model.
4. Generate synthetic scenarios.
5. Save outputs.
6. Produce validation plots.

## Expected outputs

By default, outputs are written under:

`/mnt/staas/CLICHE/00_DATA/HF_daily_flood_series_complete`

Main artifacts include:
- NetCDF posterior trace.
- CSV with observed + synthetic records.
- Validation plot images.

## Notes

- Environment variables for JAX/GPU are configured by the orchestrator.
- Sampling backend is selected automatically based on availability and preferences in `bayflood_config.py`.
- Most behavior can be tuned by editing constants in `bayflood_config.py`.
