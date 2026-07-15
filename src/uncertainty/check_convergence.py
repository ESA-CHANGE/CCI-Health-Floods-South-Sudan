# -*- coding: utf-8 -*-

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
Direct verification before uncertainty analysis.

This script answers two practical questions directly from model outputs:

Q1) Are there facilities in synthetic scenarios with impossible annual
    flood-day counts, or are high CV values mostly due to low means?

Q2) What is the exact composition of convergence tails:
    - parameters with R-hat > 1.05
    - parameters with ESS_bulk < 400

It prints explicit tables (no plots) to keep diagnostics unambiguous.
"""

import os

import arviz as az
import numpy as np
import pandas as pd


BASE_DIR = "../data/model_output"
CSV_PATH = os.path.join(BASE_DIR, "bayfloodgen_output.csv")
TRACE_PATH = os.path.join(BASE_DIR, "bayfloodgen_trace.nc")


def _base_varname(index_label: str) -> str:
    r"""Extract base variable name from ArviZ summary row index.

    Parameters
    ----------
    index_label : str
        ArviZ parameter label, e.g. "delta[hf_001]".

    Returns
    -------
    str
        Base variable name, e.g. "delta".
    """

    return index_label.split("[")[0]


def run_direct_verification(csv_path: str, trace_path: str) -> None:
    r"""Run direct scenario-range and trace-tail checks.

    Parameters
    ----------
    csv_path : str
        Path to combined observed+synthetic output CSV.
    trace_path : str
        Path to ArviZ NetCDF trace.
    """

    print("=" * 70)
    print("Q1: Facility-level annual flood-day ranges in synthetic scenarios")
    print("=" * 70)

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df_syn = df[df["is_synthetic"] == 1].copy()

    fac_scen_days = (
        df_syn.groupby(["hf_id", "scenario_id"], observed=True)["occurrence"]
        .sum()
        .reset_index()
        .rename(columns={"occurrence": "annual_days"})
    )

    n_days_simulated = int(df_syn.groupby("scenario_id")["date"].nunique().iloc[0])
    print(f"\nDays simulated per scenario: {n_days_simulated}")

    impossible = fac_scen_days[fac_scen_days["annual_days"] > n_days_simulated]
    print(f"Rows with annual_days > {n_days_simulated} (impossible): {len(impossible)}")
    if not impossible.empty:
        print(impossible.head(20).to_string(index=False))
    else:
        print("  -> None. No facility/scenario exceeds the simulated period.")

    fac_stats = (
        fac_scen_days.groupby("hf_id", observed=True)["annual_days"]
        .agg(mean="mean", std="std", max="max", min="min")
        .reset_index()
    )
    fac_stats["cv"] = fac_stats["std"] / fac_stats["mean"].replace(0, np.nan)

    print(f"\nFacility-level annual_days summary across {fac_stats['hf_id'].nunique()} facilities:")
    print(fac_stats[["mean", "std", "max", "min", "cv"]].describe().round(2).to_string())

    print("\nTop 10 facilities by CV (annual_days):")
    print(fac_stats.sort_values("cv", ascending=False).head(10).round(2).to_string(index=False))

    print("\nTop 10 facilities by MAX annual_days in any scenario:")
    print(fac_stats.sort_values("max", ascending=False).head(10).round(2).to_string(index=False))

    n_low_mean_high_cv = int(((fac_stats["mean"] < 10) & (fac_stats["cv"] > 1.0)).sum())
    n_high_mean_high_cv = int(((fac_stats["mean"] >= 10) & (fac_stats["cv"] > 1.0)).sum())
    print(f"\nFacilities with mean < 10 days AND cv > 1.0: {n_low_mean_high_cv}")
    print(f"Facilities with mean >= 10 days AND cv > 1.0: {n_high_mean_high_cv}")

    print("\n" + "=" * 70)
    print("Q2: R-hat and ESS tail composition (exact values)")
    print("=" * 70)

    trace = az.from_netcdf(trace_path)
    full_summary = az.summary(trace)

    print(f"\nTotal parameters summarized: {len(full_summary)}")

    bad_rhat = full_summary[full_summary["r_hat"] > 1.05]
    print(f"\nParameters with R-hat > 1.05: {len(bad_rhat)}")
    if not bad_rhat.empty:
        print(bad_rhat[["r_hat", "ess_bulk"]].sort_values("r_hat", ascending=False).head(20).to_string())
    else:
        print(f"  -> Confirmed: NONE. Max R-hat in trace = {full_summary['r_hat'].max():.4f}")

    bad_ess = full_summary[full_summary["ess_bulk"] < 400]
    print(
        f"\nParameters with ESS_bulk < 400: {len(bad_ess)} "
        f"({100 * len(bad_ess) / len(full_summary):.1f}% of all parameters)"
    )

    bad_ess_grouped = bad_ess.copy()
    bad_ess_grouped["base_var"] = bad_ess_grouped.index.map(_base_varname)
    print("\nLow-ESS parameters by base variable name:")
    print(bad_ess_grouped["base_var"].value_counts().to_string())

    all_grouped = full_summary.copy()
    all_grouped["base_var"] = all_grouped.index.map(_base_varname)
    print("\nTOTAL parameter count by base variable name:")
    print(all_grouped["base_var"].value_counts().to_string())

    print("\nLow-ESS as % of each variable group:")
    pct_by_var = (
        bad_ess_grouped["base_var"].value_counts()
        / all_grouped["base_var"].value_counts()
        * 100
    ).dropna().sort_values(ascending=False)
    print(pct_by_var.round(1).to_string())

    print(
        f"""
{'=' * 70}
SUMMARY
{'=' * 70}
Q1: {'Implausible values found (see table above)' if len(impossible) > 0 else 'No impossible flood-day counts found.'}
Q2: Max R-hat = {full_summary['r_hat'].max():.4f} ({'exceeds' if full_summary['r_hat'].max() > 1.05 else 'does NOT exceed'} 1.05).
    {len(bad_ess)} of {len(full_summary)} parameters have ESS < 400.
"""
    )


if __name__ == "__main__":
    run_direct_verification(CSV_PATH, TRACE_PATH)
