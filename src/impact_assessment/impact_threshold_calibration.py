# -*- coding: utf-8 -*-

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
Stage 1: threshold calibration runner.

This script extracts empirical thresholds from observed daily data and writes:
- a JSON report
- a calibration plot panel

Run this first, review suggestions, then use calibrated configuration in the
main pipeline.
"""

import os

from impact_model.calibration import (
    calibrate_thresholds,
    plot_threshold_calibration,
    print_suggested_thresholds,
    save_calibration_report,
)
from impact_model.config import OUTPUT_DIR, PLOT_DIR, SYNTHETIC_CSV
from impact_model.io import load_pool_raw


if __name__ == "__main__":
    df_pool_raw = load_pool_raw(SYNTHETIC_CSV)

    # Calibration is based on observed data.
    if "is_synthetic" in df_pool_raw.columns:
        df_obs = df_pool_raw[df_pool_raw["is_synthetic"] == 0].copy()
    else:
        df_obs = df_pool_raw.copy()

    calib = calibrate_thresholds(df_obs)
    if calib:
        print_suggested_thresholds(calib)

        report_path = os.path.join(OUTPUT_DIR, "threshold_calibration_report.json")
        save_calibration_report(calib, report_path)
        print(f"Saved: {report_path}")

        plot_threshold_calibration(calib, PLOT_DIR)
