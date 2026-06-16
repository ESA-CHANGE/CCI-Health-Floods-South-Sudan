# -*- coding: utf-8 -*-

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
CHANGE: BayFloodGEN Flood Generator v2 — South Sudan Health Facilities (Orchestrator)
===============================================================================

This script orchestrates the execution of the BayFloodGEN Flood Generator 
v2 for South Sudan Health Facilities. It leverages modularized code under 
`model_code/` to perform data loading, model building, scenario generation,
and validation.
"""

import os

# ── Keep original runtime behavior for JAX/GPU env configuration ──
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # to avoid OOM on GPU

import arviz as az
import pandas as pd

from model_code.bayflood_config import N_JOBS_SCENARIOS
from model_code.bayflood_data import expand_series_to_df, parse_series_columns
from model_code.bayflood_model import PatchGENBayGEN
from model_code.bayflood_runtime import detect_jax_devices
from model_code.bayflood_validation import validate_scenarios


if __name__ == "__main__":
    print("=== BayFloodGEN v2 paralellized (orchestrator) ===")
    has_gpu = detect_jax_devices()

    CSV_PATH = "../data/eo_pool/eo_pool_reclassified.csv"
    OUTPUT_DIR = "../data/model_output/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    PLOT_DIR = os.path.join(OUTPUT_DIR, "validation_plots")
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ──────── Load data ────────
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} facilities.")

    # Parse input series columns with original behavior
    df = parse_series_columns(df)

    print(f"Type series_binary: {type(df['series_binary'].iloc[0])}")
    print(f"Example dates[0]: {df['dates'].iloc[0][:3]}")

    # ───────────── Daily expansion ─────────────
    print("\nExpanding to daily format...")
    df_daily = expand_series_to_df(df)
    print(f"Daily records: {len(df_daily):,}")

    df_daily['scenario_id'] = -1
    df_daily['is_synthetic'] = 0

    # ───────────── Model ─────────────
    print("\Building model...")
    model = PatchGENBayGEN(df_daily)
    model.build_model()

    print("\nSampling posterior...")
    model.sample()

    trace_path = os.path.join(OUTPUT_DIR, "bayfloodgen_trace.nc")
    model.trace.to_netcdf(trace_path)
    print(f"Trace guardado en {trace_path}")

    """
    # ───────────── Load saved trace (comment previous block to re-sample) ─────────────
    trace_path = os.path.join(OUTPUT_DIR, "bayfloodgen_trace.nc")
    model.trace = az.from_netcdf(trace_path)
    print(f"Trace loaded from {trace_path}")
    """

    # ───────────── Scenario generation ─────────────
    print("\nGenerating synthetic scenarios...")
    df_syn = model.generate_scenarios(
        n_scenarios=50,
        days=365,
        start_date="2026-01-01",
        t_year_sim=1.0,
        n_jobs=N_JOBS_SCENARIOS,
    )

    # ───────────── Save ─────────────
    df_final = pd.concat([df_daily, df_syn], ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, "bayfloodgen_output.csv")
    df_final.to_csv(out_path, index=False)
    print(f"Output saved: {out_path}")

    # ───────────── Validation ─────────────
    df_obs_post = df_daily[df_daily['year'] > 2021]
    validate_scenarios(df_obs=df_obs_post, df_syn=df_syn, output_dir=PLOT_DIR)
