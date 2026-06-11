# -*- coding: utf-8 -*-
__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

import os
import pandas as pd


def ensure_output_dirs(output_dir: str, plot_dir: str) -> None:
    r"""Ensure output directories exist.
    
    Parameters
    ----------
    output_dir : str
        Directory where output files will be saved.
    plot_dir : str
        Directory where plot files will be saved.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)


def load_pool_raw(csv_path: str) -> pd.DataFrame:
    r"""Load full daily pool for sensitivity analysis.

    sensitivity_r1() requires raw day-level rows with `is_synthetic`, `date`,
    `hf_id`, `hf_category`, `scenario_id`, `occurrence`, `pct_flooded`,
    `min_distance`.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the raw pool data.
    
    Returns
    -------
    df_pool_raw : pd.DataFrame
        DataFrame containing the raw pool data, with `date` column parsed as datetime.
    """
    
    df_pool_raw = pd.read_csv(csv_path)
    if "date" in df_pool_raw.columns:
        df_pool_raw["date"] = pd.to_datetime(df_pool_raw["date"])
    return df_pool_raw


def save_outputs(
    output_dir: str,
    df_scen: pd.DataFrame,
    df_obs_stats: pd.DataFrame,
    ranking: pd.DataFrame,
    df_exc: pd.DataFrame,
    df_desc: pd.DataFrame,
    df_dunn: pd.DataFrame,
    seasonal: pd.DataFrame,
    trend: pd.DataFrame,
) -> None:
    r"""Function to save all output dataframes to CSVs in the specified output directory.
    This centralizes the saving logic and ensures consistent file naming and formats.
    
    Parameters
    ----------
    output_dir : str
        Directory where output CSVs will be saved.
    df_scen : pd.DataFrame
        DataFrame containing scenario statistics.
    df_obs_stats : pd.DataFrame
        DataFrame containing observed statistics.
    ranking : pd.DataFrame
        DataFrame containing facility ranking.
    df_exc : pd.DataFrame
        DataFrame containing exceedance table.
    df_desc : pd.DataFrame
        DataFrame containing category statistics.
    df_dunn : pd.DataFrame
        DataFrame containing Dunn test results.
    seasonal : pd.DataFrame
        DataFrame containing seasonal risk analysis.
    trend : pd.DataFrame
        DataFrame containing temporal trend analysis.
    """

    df_scen.to_parquet(os.path.join(output_dir, "scen_stats.parquet"), index=False)
    df_obs_stats.to_parquet(os.path.join(output_dir, "obs_stats.parquet"), index=False)

    ranking.to_csv(os.path.join(output_dir, "facility_ranking.csv"))
    df_exc.to_csv(os.path.join(output_dir, "exceedance_table.csv"), index=False)
    df_desc.to_csv(os.path.join(output_dir, "category_stats.csv"), index=False)
    df_dunn.to_csv(os.path.join(output_dir, "category_dunn.csv"), index=False)
    seasonal.to_csv(os.path.join(output_dir, "seasonal_risk.csv"), index=False)
    if not trend.empty:
        trend.to_csv(os.path.join(output_dir, "temporal_trend.csv"), index=False)
