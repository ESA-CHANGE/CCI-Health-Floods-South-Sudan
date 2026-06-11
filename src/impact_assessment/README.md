# Impact Assessment Workflow

This folder contains two valid ways to run impact assessment.

## Files and purpose

- `divide_csv_per_facility.py`
    - Split whole CSV into different CSV per facility.

- `impact_threshold_calibration.py`
  - Computes empirical thresholds from **observed** data.
  - Saves calibration outputs (report + calibration plot).

- `impact_assessment_calibrated_orchestrator.py`
  - Runs the **full pipeline with calibrated thresholds**.
  - This is the recommended run for data-driven thresholding.

- `impact_assessment_orchestrator.py`
  - Runs the **full pipeline with default/fixed thresholds** from config.
  - Use this for baseline or reproducible fixed-threshold comparisons.

- `impact_model/`
  - Modular implementation (`config`, `engine`, `processing`, `analysis`, `plotting`, `sensitivity`, `calibration`, `pipeline`).

---

## Recommended beginner workflow (clear order)

### Setp 0 - Divide CSV into facility-csv if not done already
run:
`python divide_csv_per_facility.py`

### Step 1 — Calibrate thresholds first
Run:

`python impact_threshold_calibration.py`

What it does:
- Loads observed daily series.
- Builds wet spells.
- Computes empirical percentiles for duration, intensity, and distance.
- Produces suggested threshold values.
- Saves calibration artifacts.

### Step 2 — Run full analysis with calibrated thresholds
Run:

`python impact_assessment_calibrated_orchestrator.py`

What it does:
- Recomputes calibration from observed data.
- Builds calibrated vulnerability config.
- Runs full impact assessment pipeline using those calibrated thresholds.

✅ Yes: in this calibrated orchestrator flow, thresholds are automatically incorporated.

---

## If you want fixed thresholds (no calibration)
Run:

`python impact_assessment_orchestrator.py`

This uses thresholds from `impact_model/config.py` as-is.

---

## Practical decision guide

- If you are new and want thresholds derived from your data:
  1) `impact_threshold_calibration.py`
  2) `impact_assessment_calibrated_orchestrator.py`

- If you want a stable baseline with fixed thresholds:
  - `impact_assessment_orchestrator.py`


