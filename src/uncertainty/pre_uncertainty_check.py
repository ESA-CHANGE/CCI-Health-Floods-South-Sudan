# -*- coding: utf-8 -*-

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
Pre-uncertainty consistency checks for observed/synthetic pool.

This script checks:
- category distributions in real vs synthetic rows,
- facility-level mean flood rates by category,
- category consistency between real and synthetic metadata,
- scenario-mean behaviour used by uncertainty workflows.
"""

import numpy as np
import pandas as pd


POOL_CSV = "../data/model_output/bayfloodgen_output.csv"


def _print_category_stats(fac_df: pd.DataFrame, label: str) -> None:
    r"""Print per-category flood-rate stats from facility aggregation.

    Parameters
    ----------
    fac_df : pd.DataFrame
        Facility-level table with columns `hf_category`, `mean_occ`.
    label : str
        Label shown in printed section title.
    """

    print(f"\nPer facility mean flood rate by category ({label}):")
    for cat in ["SAFE", "CHRONIC", "UNSTABLE", "CRITICAL"]:
        sub = fac_df[fac_df["hf_category"] == cat]
        if sub.empty:
            print(f"  {cat}: NO FACILITIES FOUND")
        else:
            print(
                f" {cat}: n={len(sub)}, mean={sub['mean_occ'].mean():.4f},"
                f" max={sub['mean_occ'].max():.4f},"
                f" p95={sub['mean_occ'].quantile(0.95):.4f}"
            )


def run_pool_checks(pool_csv: str) -> None:
    r"""Run consistency and distribution checks over combined pool CSV.

    Parameters
    ----------
    pool_csv : str
        Path to combined observed+synthetic output CSV.
    """

    print("Loading pool CSV ...")
    df = pd.read_csv(pool_csv)
    print(f" total rows: {len(df):,}")
    print(f" columns: {df.columns.tolist()}")
    print(f" is_synthetic values: {df['is_synthetic'].value_counts().to_dict()}")

    real = df[df["is_synthetic"] == 0].copy()
    print(f"\nReal rows: {len(real):,}")
    print(f" hf_category unique values: {sorted(real['hf_category'].unique().tolist())}")
    print(f" hf_category dtype: {real['hf_category'].dtype}")
    print("\nCategory distribution (real rows):")
    print(real["hf_category"].value_counts().to_string())

    fac_real = (
        real.groupby("hf_id", observed=True)
        .agg(
            hf_category=("hf_category", "first"),
            mean_occ=("occurrence", "mean"),
            n_rows=("occurrence", "count"),
        )
        .reset_index()
    )
    _print_category_stats(fac_real, "real rows, all years")

    safe = fac_real[fac_real["hf_category"] == "SAFE"]
    safe_high = safe[safe["mean_occ"] > 0.05]
    print(f"\nSAFE facilities with mean flood rate > 0.05: {len(safe_high)}")
    if not safe_high.empty:
        print(safe_high[["hf_id", "mean_occ", "n_rows"]].to_string(index=False))

    syn = df[df["is_synthetic"] == 1].copy()
    print(f"\nSynthetic rows: {len(syn):,}")
    print("hf_category distribution in synthetic:")
    print(syn["hf_category"].value_counts().to_string())

    fac_syn = (
        syn.groupby("hf_id", observed=True)
        .agg(
            hf_category=("hf_category", "first"),
            mean_occ=("occurrence", "mean"),
            n_rows=("occurrence", "count"),
        )
        .reset_index()
    )
    _print_category_stats(fac_syn, "synthetic rows, all scenarios")

    meta_real = real.groupby("hf_id", observed=True)["hf_category"].first()
    meta_syn = syn.groupby("hf_id", observed=True)["hf_category"].first()
    merged = meta_real.rename("cat_real").to_frame().join(meta_syn.rename("cat_syn"), how="inner")
    mismatch = merged[merged["cat_real"] != merged["cat_syn"]]
    print(f"\nFacilities where hf_category differs between real/synthetic: {len(mismatch)}")
    if not mismatch.empty:
        print(mismatch.head(20).to_string())

    print("\nCompute scen_means (exact, no chunking)")
    scen_means = (
        syn.groupby(["hf_id", "scenario_id"], observed=True)["occurrence"]
        .mean()
        .unstack(level="scenario_id")
    )
    print(f" scen_means shape: {scen_means.shape}")

    meta = (
        real[["hf_id", "hf_category"]]
        .drop_duplicates(subset="hf_id")
        .set_index("hf_id")
    )
    scen_means = scen_means.join(meta, how="left")
    missing_cat = int(scen_means["hf_category"].isna().sum())
    print(f" missing hf_category after join: {missing_cat}")

    print("\nPer category mean of per-scenario means (synthetic):")
    for cat in ["SAFE", "CHRONIC", "UNSTABLE", "CRITICAL"]:
        sub = scen_means[scen_means["hf_category"] == cat].drop(columns="hf_category")
        if sub.empty:
            print(f"  {cat}: NO FACILITIES FOUND")
            continue
        vals = sub.values.flatten()
        vals = vals[~np.isnan(vals)]
        print(
            f" {cat}: n_fac={len(sub)}, mean={vals.mean():.4f},"
            f" max={vals.max():.4f}, p95={np.percentile(vals, 95):.4f}"
        )

    safe_syn = scen_means[scen_means["hf_category"] == "SAFE"].drop(columns="hf_category")
    safe_means = safe_syn.mean(axis=1)
    safe_high_syn = safe_means[safe_means > 0.10]
    print(f"\nSAFE facilities with mean synthetic flood rate > 0.10: {len(safe_high_syn)}")
    if not safe_high_syn.empty:
        print(safe_high_syn.to_string())

    safe_fac_syn = fac_syn[fac_syn["hf_category"] == "SAFE"]
    safe_high_fac_syn = safe_fac_syn[safe_fac_syn["mean_occ"] > 0.10]
    print(
        "\nSAFE facilities with mean synthetic flood rate > 0.10 "
        f"(per facility aggregation): {len(safe_high_fac_syn)}"
    )
    if not safe_high_fac_syn.empty:
        print(safe_high_fac_syn[["hf_id", "mean_occ", "n_rows"]].to_string(index=False))


if __name__ == "__main__":
    run_pool_checks(POOL_CSV)
