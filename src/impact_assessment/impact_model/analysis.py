# -*- coding: utf-8 -*-
__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau

from impact_model.config import CAT_ORDER, EXCEEDANCE_THRESHOLDS


# ─────────────────────────────────────────────────────────────────────────────
# Risk assessment and statistical analysis
# ─────────────────────────────────────────────────────────────────────────────

def facility_ranking(df_scen: pd.DataFrame) -> pd.DataFrame:
    r"""This function computes a risk ranking of facilities based on the
    median annual loss across all synthetic scenarios.
    """

    print("\n─── 1. FACILITY RANKING ───")
    ranking = (
        df_scen.groupby(["hf_id", "hf_category"])
        .agg(
            loss_p50=("annual_mean_loss", "median"),
            loss_p05=("annual_mean_loss", lambda x: x.quantile(0.05)),
            loss_p95=("annual_mean_loss", lambda x: x.quantile(0.95)),
            days_gt50=("days_loss_gt50", "median"),
            days_gt75=("days_loss_gt75", "median"),
            p_total=("days_loss_100", lambda x: (x > 0).mean()),
        )
        .reset_index()
        .sort_values("loss_p50", ascending=False)
        .reset_index(drop=True)
    )
    ranking.index += 1
    ranking.index.name = "rank"
    print(f"  {len(ranking)} facilities | Top 5:")
    print(ranking.head(5)[["hf_id", "hf_category", "loss_p50", "loss_p95", "p_total"]].to_string())
    return ranking


def exceedance_table(df_scen: pd.DataFrame,
                     thresholds=EXCEEDANCE_THRESHOLDS) -> pd.DataFrame:
    r"""This function computes an exceedance table for the synthetic
    scenarios, showing the proportion of scenarios where the annual mean
    loss exceeds specified thresholds.
    """

    print("\n─── 2. EXCEEDANCE TABLE ───")
    rows = []
    for (fid, cat), grp in df_scen.groupby(["hf_id", "hf_category"]):
        losses = grp["annual_mean_loss"].values
        row = {"hf_id": fid, "hf_category": cat}
        for thr in thresholds:
            row[f"p_exceed_{int(thr * 100):02d}pct"] = float((losses >= thr).mean())
        rows.append(row)
    df_exc = pd.DataFrame(rows).sort_values("p_exceed_10pct", ascending=False).reset_index(drop=True)
    print(f"  {len(df_exc)} facilities × {len(thresholds)} umbrales")
    return df_exc


def dunn_test(groups, labels):
    r"""Performs Dunn's test for multiple comparisons following a
    Kruskal-Wallis test.
    """

    all_data = np.concatenate(groups)
    all_ranks = stats.rankdata(all_data)
    n_total = len(all_data)
    sizes, rank_sums = [], []
    idx = 0
    for g in groups:
        sizes.append(len(g))
        rank_sums.append(all_ranks[idx:idx + len(g)].sum())
        idx += len(g)
    unique, counts = np.unique(all_data, return_counts=True)
    tie_corr = 1 - np.sum(counts ** 3 - counts) / (n_total ** 3 - n_total)
    if tie_corr == 0:
        tie_corr = 1.0
    rows = []
    pairs = list(combinations(range(len(groups)), 2))
    for i, j in pairs:
        ni, nj = sizes[i], sizes[j]
        if ni == 0 or nj == 0:
            continue
        se = np.sqrt(
            (n_total * (n_total + 1) / 12.0 - np.sum(counts ** 3 - counts) / (12.0 * (n_total - 1)))
            * (1 / ni + 1 / nj)
        )
        if se == 0:
            continue
        z = (rank_sums[i] / ni - rank_sums[j] / nj) / se
        p = 2 * stats.norm.sf(abs(z))
        rows.append({
            "group_1": labels[i], "group_2": labels[j],
            "z_stat": round(z, 3), "p_raw": p,
            "p_bonf": min(p * len(pairs), 1.0),
            "sig": "***" if p * len(pairs) < 0.001 else "**" if p * len(pairs) < 0.01
            else "*" if p * len(pairs) < 0.05 else "ns",
        })
    return pd.DataFrame(rows)


def category_comparison(df_scen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""This function performs a category comparison of the annual mean
    loss across the different facility categories using the Kruskal-Wallis
    test, followed by Dunn's test for multiple comparisons.
    """

    print("\n─── 3. CATEGORY COMPARISON ───")
    groups = [df_scen[df_scen["hf_category"] == c]["annual_mean_loss"].values for c in CAT_ORDER]
    nonempty = [(c, g) for c, g in zip(CAT_ORDER, groups) if len(g) > 0]
    kw_stat, kw_p = stats.kruskal(*[g for _, g in nonempty])
    print(f"  Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.2e}")
    desc = pd.DataFrame([{
        "category": c, "n_obs": len(g),
        "median": np.median(g), "mean": np.mean(g),
        "p25": np.percentile(g, 25), "p75": np.percentile(g, 75),
        "p95": np.percentile(g, 95),
    } for c, g in nonempty])
    print(desc.to_string(index=False))
    df_dunn = dunn_test([g for _, g in nonempty], [c for c, _ in nonempty])
    print(df_dunn.to_string(index=False))
    return desc, df_dunn


def seasonal_risk(df_scen: pd.DataFrame) -> pd.DataFrame:
    r"""This function analyzes the seasonal risk patterns by expanding
    the monthly_loss column from the synthetic scenarios dataframe into
    a long format.
    """

    print("\n─── 4. SEASONAL RISK ───")
    rows = []
    for _, row in df_scen.iterrows():
        for m, loss in enumerate(row["monthly_loss"], start=1):
            rows.append({
                "hf_category": row["hf_category"],
                "scenario_id": row["scenario_id"],
                "month": m,
                "loss": loss,
            })
    df_m = pd.DataFrame(rows)
    seasonal = (
        df_m.groupby(["hf_category", "month"])["loss"]
        .agg(median="median",
             p05=lambda x: x.quantile(0.05),
             p95=lambda x: x.quantile(0.95))
        .reset_index()
    )
    for cat in CAT_ORDER:
        sub = seasonal[seasonal["hf_category"] == cat]
        if sub.empty:
            continue
        peak = sub.loc[sub["median"].idxmax()]
        print(f"  {cat:10s}: pico en mes {int(peak['month'])} "
              f"(mediana={peak['median']:.3f})")
    return seasonal


def temporal_trend(df_obs: pd.DataFrame) -> pd.DataFrame:
    r"""This function analyzes the temporal trend in the observed series by
    grouping the observed data by category and year.
    """

    print("\n─── 5. TEMPORAL TREND ───")
    if df_obs.empty:
        print("  Sin datos observados.")
        return pd.DataFrame()
    trend = (
        df_obs.groupby(["hf_category", "year"])
        .agg(
            flood_freq_med=("days_flooded", lambda x: (x / 365).median()),
            mean_loss_med=("annual_mean_loss", "median"),
            days_gt50_med=("days_loss_gt50", "median"),
            days_gt75_med=("days_loss_gt75", "median"),
        )
        .reset_index()
    )
    print("  Mann-Kendall (pérdida media):")
    for cat in CAT_ORDER:
        sub = trend[trend["hf_category"] == cat].sort_values("year")
        if len(sub) < 4:
            continue
        tau, p = kendalltau(sub["year"], sub["mean_loss_med"])
        direction = "↑" if tau > 0 else "↓"
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    {cat:10s}: τ={tau:+.3f}, p={p:.4f} {direction} {sig}")
    return trend
