# CCI Health Floods — `src` General Pipeline Guide

This README documents the full processing pipeline in `src`, the purpose of each module, the exact execution order (1→10), and the main generated files.

## `src` folder structure

- `flood_maps/`: flood map preprocessing, hazard map generation, and EDA.
- `eo_pool/`: health-facility filtering and EO pool extraction/reclassification.
- `model/`: BayFloodGEN training + synthetic scenario generation.
- `uncertainty/`: uncertainty diagnostics from the trained model trace/output.
- `impact_assessment/`: threshold calibration + calibrated impact analysis.

---

## End-to-end execution order (recommended)

## 1) `flood_maps/binarize_maps.py`
Purpose:
- Binarizes VIIRS flood maps and standardizes AOI crop.

Run:
- `python flood_maps/binarize_maps.py`

Main inputs:
- `../data/flood_maps/viirs_maps/*.tif`

Main outputs:
- `../data/flood_maps/binary_maps/*.tif`
- `../data/flood_maps/binary_maps/maps2review.txt` (only if problematic maps are detected)

---

## 2) `eo_pool/get_valid_facilities.ipynb`
Purpose:
- Builds/exports validated health facility coordinates used by EO extraction.

Run:
- Open notebook and run all cells.

Main output:
- `../data/valid_facilities.csv`

---

## 3) `flood_maps/hazard_maps.py`
Purpose:
- Generates flood hazard products from binary daily maps.

Run (recommended for this pipeline):
- `python flood_maps/hazard_maps.py --map_type annual_3`

Main input:
- `../data/flood_maps/binary_maps/*.tif`

Main outputs (annual_3 mode):
- `../data/hazard_maps/annual_3/floods_annual_<YEAR>.tif` (4 bands: frequency, duration, valid obs, max consecutive)

Note:
- Other map types exist (`annual`, `seasonal`, `seasonal_2`, `annual_2`) but the downstream EDA script in this repository is prepared for `annual_3` filenames.

---

## 4) `flood_maps/eda_maps.py`
Purpose:
- Runs exploratory analysis of annual hazard products and generates QA/diagnostic plots.

Run:
- `python flood_maps/eda_maps.py`

Main inputs:
- `../data/hazard_maps/annual_3/floods_annual_<YEAR>.tif`
- `../data/valid_facilities.csv`

Main outputs:
- Plot and analysis artifacts in `../data/EDA/`
- Facility-zone table: `../data/EDA/hf_areas_persistence_variability.csv`

---

## 5) `eo_pool/create_eo_pool_daily.py`
Purpose:
- Extracts daily flood metrics around each facility (binary occurrence, % flooded pixels, min distance).

Run:
- `python eo_pool/create_eo_pool_daily.py`

Main inputs:
- `../data/valid_facilities.csv`
- `../data/flood_maps/binary_maps/*.tif`

Main outputs:
- `../data/eo_pool/eo_pool.parquet`
- `../data/eo_pool/summary_hf_years.csv`
- `../data/eo_pool/metadata.json`
- Optional NPZ per HF-year (if NPZ saving mode is used)

---

## 6) `eo_pool/check_facilities_categories.ipynb`
Purpose:
- Checks/adjusts facility categories and exports model-ready EO pool.

Run:
- Open notebook and run all cells.

Main output:
- `../data/eo_pool/eo_pool_reclassified.csv`

---

## 7) `model/bayflood_orchestrator.py`
Purpose:
- Trains BayFloodGEN and generates synthetic daily flood scenarios.
- In this branch, scenario generation is memory-safe (streamed batches to disk).

Run:
- `python model/bayflood_orchestrator.py`

Main input:
- `../data/eo_pool/eo_pool_reclassified.csv`

Main outputs:
- `../data/model_output/bayfloodgen_trace.nc`
- `../data/model_output/synthetic_scenarios/scenario_batch_*.csv`
- `../data/model_output/bayfloodgen_output.csv` (stream-concatenated observed + synthetic)
- Validation plots in `../data/model_output/validation_plots/`

Critical behavior:
- Validation does **not** load all synthetic rows by default.
- It uses a bounded number of batch files controlled by `N_BATCH_FILES_FOR_VALIDATION`
  in `model/model_code/bayflood_config.py` (default: `4`).
- To validate with all synthetic data, set `N_BATCH_FILES_FOR_VALIDATION = None`.
- Using all synthetic data may require substantially more RAM/time.

---

## 8) `uncertainty/trace_analysis.py`
Purpose:
- Computes uncertainty diagnostics from model output + posterior trace.

Run:
- `python uncertainty/trace_analysis.py`

Main inputs:
- `../data/model_output/bayfloodgen_output.csv`
- `../data/model_output/bayfloodgen_trace.nc`

Main outputs:
- `../data/uncertainty/trace_uncertainty/A_scenario_stability.png`
- `../data/uncertainty/trace_uncertainty/B_facility_credible_intervals.png`
- `../data/uncertainty/trace_uncertainty/C_t_year_sensitivity.png`
- `../data/uncertainty/trace_uncertainty/D1_global_parameter_posteriors.png`
- `../data/uncertainty/trace_uncertainty/D2_delta_per_facility.png`
- `../data/uncertainty/trace_uncertainty/D3_convergence_diagnostics.png`
- `../data/uncertainty/trace_uncertainty/F_monthly_fan_chart.png`
- `../data/uncertainty/trace_uncertainty/trace_uncertainty_summary_table.csv`

---

## 9) `impact_assessment/impact_threshold_calibration.py`
Purpose:
- Calibrates impact thresholds from observed data and writes calibration artifacts.

Run:
- `python impact_assessment/impact_threshold_calibration.py`

Main input:
- `../data/model_output/bayfloodgen_output.csv`

Main outputs:
- `../data/impact_outputs/threshold_calibration_report.json`
- `../data/impact_outputs/impact_plots/threshold_calibration.png`

---
## 10) `impact_assessment/divide_csv_per_facility.py` 
Purpose:
- Divide the whole EO+synthethic pool in different CSV files per facilities for performance.

Run:
- `python impact_assessment/divide_csv_per_facilitiy.py`

Input:
- `../data/model_output/bayfloodgen_output.csv`

Output:
- `../data/model_output/by_facility/*.parquet`

## 11) `impact_assessment/impact_assessment_calibrated_orchestrator.py`
Purpose:
- Runs full impact assessment using calibrated thresholds.

Run:
- `python impact_assessment/impact_assessment_calibrated_orchestrator.py`

Main input:
- `../data/model_output/by_facility/` (facility partitions)
- `../data/model_output/bayfloodgen_output.csv` (for sensitivity and calibration base)

Main outputs:
- `../data/impact_outputs/scen_stats.parquet`
- `../data/impact_outputs/obs_stats.parquet`
- `../data/impact_outputs/facility_ranking.csv`
- `../data/impact_outputs/exceedance_table.csv`
- `../data/impact_outputs/category_stats.csv`
- `../data/impact_outputs/category_dunn.csv`
- `../data/impact_outputs/seasonal_risk.csv`
- `../data/impact_outputs/temporal_trend.csv` (if trend data exists)
- Plots in `../data/impact_outputs/impact_plots/`:
  - `facility_ranking.png`
  - `exceedance_heatmap.png`
  - `category_comparison.png`
  - `seasonal_risk.png`
  - `temporal_trend.png`
  - `sensitivity_r1.png`

---

## Synthetic scenario recreation shortcut

If the goal is only to recreate synthetic scenarios and downstream uncertainty/impact products (without rebuilding flood maps and EO extraction), execute only:

- **Step 7** → model generation
- **Step 8** → uncertainty analysis
- **Step 9** → threshold calibration
- **Setp 10** → divide pool into facilities CSV
- **Step 11** → calibrated impact assessment

This assumes prerequisite model input data already exists (especially `../data/eo_pool/eo_pool_reclassified.csv`).

## Branch-specific model updates vs `main`

For the model module (`src/model`), this branch introduces:

- Optional structural breakpoint block via `INCLUDE_BREAKPOINT`.
- Non-centered model parameterization and tighter configuration for stable inference.
- Streamed scenario generation/output assembly to reduce memory pressure.
- Validation-over-sample-by-default behavior with explicit control to use all synthetic data.

---

## Practical dependency summary

- Steps 1→6 prepare the EO/model input.
- Step 7 produces synthetic scenarios (`bayfloodgen_output.csv`).
- Step 8 quantifies model uncertainty.
- Steps 9→11 produce calibrated impact metrics and rankings.

--
## More information

For further details on impact assessment workflow, please refer to the ```./impact_assessment```folder and its respective README file.