# BayFloodGEN modular workflow

This package contains the modular BayFloodGEN workflow used by `src/model/bayflood_orchestrator.py`.

## Module structure

- `bayflood_config.py`
  - Centralized configuration constants (sampling, scenarios, backend preferences, validation sample size).
- `bayflood_runtime.py`
  - Runtime/backend helpers (JAX device detection and sampler selection).
- `bayflood_data.py`
  - Data parsing and expansion utilities.
  - Converts facility time-series into daily long-format records.
- `bayflood_model.py`
  - Bayesian model class `PatchGENBayGEN`.
  - Model building, posterior sampling, diagnostics.
  - Scenario generation helpers, including streamed batch writing and streamed output concatenation.
- `bayflood_validation.py`
  - Validation plots for observed vs synthetic behavior.

## Recommended usage

From the repository root:

```bash
python src/model/bayflood_orchestrator.py
```

Pipeline steps:
1. Load and parse EO pool input.
2. Expand to daily records.
3. Build and sample Bayesian model.
4. Generate synthetic scenario batches.
5. Stream-concatenate observed + synthetic output CSV.
6. Run validation plots.

## Branch updates vs `main`

This branch includes model/scenario updates compared to older `main` behavior:

- **Optional breakpoint block**
  - Controlled with `INCLUDE_BREAKPOINT`.
  - If disabled, breakpoint terms are not sampled and are not required during scenario simulation.

- **Inference stability/memory updates**
  - Non-centered facility-level parameterization.
  - Configuration-driven `kappa` prior (`KAPPA_ALPHA`, `KAPPA_BETA`).
  - Raw non-centered variables are dropped from trace after sampling to reduce memory footprint.

- **Scenario generation is streamed to disk**
  - Synthetic output is written as `scenario_batch_*.csv` files in an output batch directory.
  - Final combined output is created with chunked append (`stream_concat_csv`) instead of full in-memory concatenation.

- **Validation reads a synthetic sample by default**
  - Validation uses `load_scenarios_sample(...)` with `max_files=N_BATCH_FILES_FOR_VALIDATION`.
  - Default is a bounded subset (currently `4` batch files).

## Validation scope: subset vs all synthetic

Configuration key in `bayflood_config.py`:

- `N_BATCH_FILES_FOR_VALIDATION = 4` (default)
  - Loads only part of synthetic data for validation plots.
  - Recommended on shared/limited-memory environments.

- `N_BATCH_FILES_FOR_VALIDATION = None`
  - Loads all synthetic batch files for validation.
  - Use only if enough RAM/time is available.

## Expected output artifacts

Depending on orchestrator paths, typical model outputs include:

- Posterior trace (`.nc`).
- Scenario batch files (`scenario_batch_*.csv`).
- Streamed combined output CSV (observed + synthetic).
- Validation plots (`val_*_v2_4.png`).

## Notes

- GPU/JAX environment variables are set in orchestrator.
- Sampler backend is auto-selected from `bayflood_config.py` preferences and installed packages.
- Most behavior can be tuned through `bayflood_config.py`.
