# -*- coding: utf-8 -*-
__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

import os
import pandas as pd

from impact_model.analysis import (
    category_comparison,
    exceedance_table,
    facility_ranking,
    seasonal_risk,
    temporal_trend,
)
from impact_model.config import OBS_CUTOFF, OUTPUT_DIR, PLOT_DIR, SYNTHETIC_CSV
from impact_model.io import ensure_output_dirs, load_pool_raw, save_outputs
from impact_model.plotting import (
    plot_category_comparison,
    plot_exceedance_heatmap,
    plot_ranking,
    plot_seasonal_risk,
    plot_sensitivity,
    plot_temporal_trend,
)
from impact_model.processing import process_by_facility
from impact_model.sensitivity import sensitivity_r1
from impact_model.engine import VulnerabilityConfig


def run_all(
    parts_dir: str = "../data/model_output/by_facility",
    cfg: VulnerabilityConfig | None = None,
) -> None:
    r"""Execute complete impact assessment workflow with preserved behavior.

    Parameters
    ----------
    parts_dir : str
        Directory with facility-level parquet partitions.
    cfg : VulnerabilityConfig | None
        Optional vulnerability config. If provided, calibrated thresholds are
        applied in processing while preserving the rest of pipeline behavior.
    """

    ensure_output_dirs(OUTPUT_DIR, PLOT_DIR)

    df_scen, df_obs_stats = process_by_facility(parts_dir, cfg=cfg)

    print(f"scen_stats: {len(df_scen):,} | obs_stats: {len(df_obs_stats):,}")

    # ──────── Risk analysis ────────
    ranking = facility_ranking(df_scen)
    df_exc = exceedance_table(df_scen)
    df_desc, df_dunn = category_comparison(df_scen)
    seasonal = seasonal_risk(df_scen)

    df_obs_stats_filt = df_obs_stats[df_obs_stats.get("year", pd.Series(dtype=int)).le(OBS_CUTOFF)] \
        if "year" in df_obs_stats.columns else df_obs_stats
    trend = temporal_trend(df_obs_stats_filt)

    # ──────── Save results ────────
    save_outputs(
        output_dir=OUTPUT_DIR,
        df_scen=df_scen,
        df_obs_stats=df_obs_stats,
        ranking=ranking,
        df_exc=df_exc,
        df_desc=df_desc,
        df_dunn=df_dunn,
        seasonal=seasonal,
        trend=trend,
    )

    # ──────── Plots ────────
    plot_ranking(ranking, PLOT_DIR)
    plot_exceedance_heatmap(df_exc, PLOT_DIR)
    plot_category_comparison(df_desc, df_dunn, df_scen, PLOT_DIR)
    plot_seasonal_risk(seasonal, PLOT_DIR)
    plot_temporal_trend(trend, PLOT_DIR)

    # ──────── Sensitivity (re-process only synthetic, fast) ─────────────────
    # Fix: sensitivity_r1 requires raw pool dataframe as input.
    df_pool_raw = load_pool_raw(SYNTHETIC_CSV)
    sens = sensitivity_r1(df_pool_raw)
    plot_sensitivity(sens, PLOT_DIR)

    print("\n=== Analysis completed ===")
    print(f"  Outputs: {OUTPUT_DIR}")
    print(f"  Plots:   {PLOT_DIR}")
