# -*- coding: utf-8 -*-
__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

import gc
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from impact_model.engine import VulnerabilityConfig, VulnerabilityEngine


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation ON-THE-FLY —  without saving daily details
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_series(df: pd.DataFrame, is_syn: bool) -> dict:
    r"""It reduces a processed series to a row of annual metrics, including
    mean and max loss, number of days above certain loss thresholds, number
    of long spells, and monthly loss patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the processed daily series for a single facility and scenario/year.
    is_syn : bool
        Flag indicating whether the series is synthetic (True) or observed (False), which determines
        whether to include the year in the output record.
    
    Returns
    -------
    rec : dict
        A dictionary containing the aggregated metrics for the facility and scenario/year.
    """

    cl = df["combined_loss"].values
    oc = df["occurrence"].values
    sd = df["spell_duration"].values
    rec = {
        "hf_id": df["hf_id"].iloc[0],
        "hf_category": str(df["hf_category"].iloc[0]),
        "scenario_id": int(df["scenario_id"].iloc[0]),
        "days_flooded": int(oc.sum()),
        "annual_mean_loss": float(cl.mean()),
        "annual_max_loss": float(cl.max()),
        "days_loss_gt50": int((cl >= 0.50).sum()),
        "days_loss_gt75": int((cl >= 0.75).sum()),
        "days_loss_100": int((cl >= 0.999).sum()),
        "n_spells_15plus": int(len(np.unique(
            df["spell_id"].values[sd >= 15][sd[sd >= 15] > 0]
        ))) if (sd >= 15).any() else 0,
        # For stationarity: save mean monthly loss
        "monthly_loss": (
            df.assign(month=df["date"].dt.month)
            .groupby("month")["combined_loss"].mean()
            .reindex(range(1, 13), fill_value=0.0)
            .values.tolist()
        ),
    }
    if not is_syn:
        rec["year"] = int(df["_year"].iloc[0])
    return rec


def process_by_facility(parts_dir: str,
                        cfg: VulnerabilityConfig = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""This function processes the partitioned Parquet files for each
    facility, applies the vulnerability engine to compute the combined
    loss for each daily series, and aggregates the results into summary
    records for both synthetic scenarios and observed series.
    
    Parameters
    ----------
    parts_dir : str
        Directory containing the partitioned Parquet files for each facility.
    cfg : VulnerabilityConfig, optional
        Configuration object for the vulnerability engine. If None, a default
        configuration is used.
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two DataFrames: one for synthetic scenarios and one
        for observed series.
    """

    cfg = cfg or VulnerabilityConfig()
    engine = VulnerabilityEngine(cfg)

    files = sorted([f for f in os.listdir(parts_dir) if f.endswith(".parquet")])
    syn_recs = []
    obs_recs = []

    for fname in tqdm(files, desc="Facilities", unit="facility"):
        print("Processing facility: ", fname)
        df_fac = pd.read_parquet(os.path.join(parts_dir, fname))

        # Synthetic
        syn = df_fac[df_fac["is_synthetic"] == 1]
        print(f"  Synthetic: {len(syn)} rows, {syn['scenario_id'].nunique()} scenarios")
        for sc, grp in syn.groupby("scenario_id", sort=False):
            print(f"    Scenario {sc}: {len(grp)} rows")
            res = engine.apply(grp.reset_index(drop=True))
            print(f"    → apply OK")
            syn_recs.append(_aggregate_series(res, is_syn=True))
            print(f"    → aggregate OK")

        # Observed, grouped by year
        obs = df_fac[df_fac["is_synthetic"] == 0]
        for yr, grp in obs.groupby("_year", sort=False):
            print(f"  Year {yr}: {len(grp)} rows")
            if len(grp) > 366:
                grp = grp.sort_values("date").head(366)
            res = engine.apply(grp.reset_index(drop=True))
            obs_recs.append(_aggregate_series(res, is_syn=False))

        del df_fac, syn, obs
        gc.collect()

    return pd.DataFrame(syn_recs), pd.DataFrame(obs_recs)
