# impact_model package

Internal modular package for impact assessment.

This README focuses on module responsibilities and API boundaries.
Execution order and user workflow are documented in the parent folder README.

## Module map

- `config.py`
	- Constants and defaults (paths, thresholds, colors, rule toggles, weights).

- `engine.py`
	- Core vulnerability logic.
	- Main classes/functions:
		- `VulnerabilityConfig`
		- `VulnerabilityEngine`
		- `_build_threshold_arrays()`
		- `_apply_thresholds_vec()`

- `processing.py`
	- Facility-level processing and aggregation from partitioned parquet input.
	- Main functions:
		- `process_by_facility()`
		- `_aggregate_series()`

- `analysis.py`
	- Statistical/risk summaries from aggregated outputs.
	- Main functions:
		- `facility_ranking()`
		- `exceedance_table()`
		- `category_comparison()`
		- `seasonal_risk()`
		- `temporal_trend()`
		- `dunn_test()`

- `plotting.py`
	- Plot generation for ranking, exceedance, category, seasonal, temporal, and sensitivity outputs.

- `sensitivity.py`
	- Threshold sensitivity experiments.
	- Main function:
		- `sensitivity_r1(df_pool_raw, threshold_sets, labels)`

- `calibration.py`
	- Empirical threshold calibration utilities.
	- Main functions:
		- `compute_spells()`
		- `calibrate_thresholds(df_obs)`
		- `plot_threshold_calibration(calib, output_dir)`
		- `compute_empirical_thresholds(df_pool_raw, synthetic_only=True)`
		- `build_config_from_calibration(calib, base_cfg=None)`
		- `save_calibration_report(calibration, output_path)`

- `io.py`
	- Data input/output helpers.
	- Main functions:
		- `ensure_output_dirs()`
		- `load_pool_raw()`
		- `save_outputs()`

- `pipeline.py`
	- High-level composition of processing, analysis, saving, plotting, and sensitivity.
	- Main function:
		- `run_all(parts_dir=..., cfg=None)`

## Design notes

- `pipeline.run_all()` accepts optional `cfg` so calibrated thresholds can be applied without changing module defaults.
- Calibration utilities return structured dictionaries and optional plots/reports to support reproducible threshold selection.
- `__init__.py` re-exports public symbols for convenient package-level imports.
