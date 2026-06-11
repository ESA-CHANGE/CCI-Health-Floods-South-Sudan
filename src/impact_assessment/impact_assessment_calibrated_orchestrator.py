# -*- coding: utf-8 -*-

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
Stage 2: calibrated pipeline runner.

This script computes empirical thresholds from observed data and applies them
immediately to run the full impact assessment pipeline.
"""

from impact_model.calibration import calibrate_thresholds, build_config_from_calibration
from impact_model.config import SYNTHETIC_CSV
from impact_model.io import load_pool_raw
from impact_model.pipeline import run_all


if __name__ == "__main__":
    df_pool_raw = load_pool_raw(SYNTHETIC_CSV)
    if "is_synthetic" in df_pool_raw.columns:
        df_obs = df_pool_raw[df_pool_raw["is_synthetic"] == 0].copy()
    else:
        df_obs = df_pool_raw.copy()

    calib = calibrate_thresholds(df_obs)
    cfg = build_config_from_calibration(calib) if calib else None

    run_all(parts_dir="../data/model_output/by_facility", cfg=cfg)
