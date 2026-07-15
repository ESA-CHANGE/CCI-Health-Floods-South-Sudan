# -*- coding: utf-8 -*-

from __future__ import annotations

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
Combined uncertainty analysis (v6-compatible workflow).

This script reproduces the v6 uncertainty logic used in the standalone
analysis scripts, adapted to repository paths and coding style.

Main components:
1) VIIRS channel correction and uncertainty propagation
2) Facility-level uncertainty table from synthetic scenarios
3) Summary report and uncertainty decomposition plots
4) Decision-confidence plots (exceedance-based and median-based)
5) Sensitivity heatmap to VIIRS error-rate assumptions
"""

import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
P_FD_NOMINAL = 0.0072
P_OM_NOMINAL = 0.2124
_CALIB_DENOM = 1.0 - P_OM_NOMINAL - P_FD_NOMINAL
_EPS = 1e-9

CATEGORY_COLORS = {
    "SAFE": "#4CAF50",
    "CHRONIC": "#2196F3",
    "UNSTABLE": "#FF9800",
    "CRITICAL": "#F44336",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. SENSOR CHANNEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def correct_prob(p_obs, p_fd: float = P_FD_NOMINAL, p_om: float = P_OM_NOMINAL):
    r"""Invert VIIRS channel from observed probability to corrected probability.

    Parameters
    ----------
    p_obs : array-like
        Observed flood probability values.
    p_fd : float
        False detection probability P(obs=1 | true=0).
    p_om : float
        Omission probability P(obs=0 | true=1).

    Returns
    -------
    np.ndarray
        Corrected true flood probabilities clipped to [0, 1].
    """

    denom = 1.0 - p_om - p_fd
    return np.clip((np.asarray(p_obs, float) - p_fd) / denom, 0.0, 1.0)


def forward_channel(p_true, p_fd: float = P_FD_NOMINAL, p_om: float = P_OM_NOMINAL):
    r"""Apply VIIRS channel from true probability to observed probability.

    Parameters
    ----------
    p_true : array-like
        Corrected true flood probabilities.
    p_fd : float
        False detection probability P(obs=1 | true=0).
    p_om : float
        Omission probability P(obs=0 | true=1).

    Returns
    -------
    np.ndarray
        Simulated observed probabilities.
    """

    p = np.asarray(p_true, float)
    return p * (1.0 - p_om) + (1.0 - p) * p_fd


def sensor_variance_annual(p_true_mean: float, n_eff: float) -> float:
    r"""Compute annual variance floor from VIIRS sensor imperfections.

    Parameters
    ----------
    p_true_mean : float
        Mean corrected flood probability for a facility-year.
    n_eff : float
        Effective independent observations per facility-year.

    Returns
    -------
    float
        Sensor-propagated variance of corrected annual flood probability.
    """

    p_obs = forward_channel(p_true_mean)
    return float(p_obs * (1.0 - p_obs) / (n_eff * _CALIB_DENOM ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# 2. MC PROPAGATION OF ERROR-RATE UNCERTAINTY
# ─────────────────────────────────────────────────────────────────────────────
def sample_viirs_error_uncertainty(
    p_obs_samples,
    n_mc: int = 2_000,
    seed: int = 0,
    p_fd_a: float = 3.6,
    p_fd_b: float = 496.4,
    p_om_a: float = 42.5,
    p_om_b: float = 157.5,
):
    r"""Propagate VIIRS error-rate uncertainty through calibration by MC.

    Parameters
    ----------
    p_obs_samples : array-like
        Observed probability samples (typically one per scenario).
    n_mc : int
        Number of Monte Carlo draws for p_fd/p_om.
    seed : int
        Random seed.
    p_fd_a, p_fd_b : float
        Beta prior parameters for p_fd.
    p_om_a, p_om_b : float
        Beta prior parameters for p_om.

    Returns
    -------
    dict[str, np.ndarray]
        Mean, SD and 90% interval summaries of corrected probabilities.
    """

    rng = np.random.default_rng(seed)
    p_fd_draws = rng.beta(p_fd_a, p_fd_b, size=n_mc)
    p_om_draws = rng.beta(p_om_a, p_om_b, size=n_mc)
    p_obs = np.asarray(p_obs_samples, float)

    p_true_mc = np.stack(
        [correct_prob(p_obs, p_fd=pfd, p_om=pom) for pfd, pom in zip(p_fd_draws, p_om_draws)],
        axis=0,
    )

    return {
        "p_true_mean": p_true_mc.mean(axis=0),
        "p_true_sd": p_true_mc.std(axis=0, ddof=1),
        "p_true_hdi_lo": np.percentile(p_true_mc, 5, axis=0),
        "p_true_hdi_hi": np.percentile(p_true_mc, 95, axis=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN UNCERTAINTY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def facility_uncertainty_table(
    pool_csv: str,
    obs_year_cutoff: int = 2021,
    hdi_prob: float = 0.90,
    propagate_rate_unc: bool = True,
    n_mc: int = 2_000,
    flood_rate_threshold: float = 0.10,
    n_eff_annual: float = 106.6,
) -> pd.DataFrame:
    r"""Build per-facility uncertainty table from combined model output.

    Parameters
    ----------
    pool_csv : str
        Combined observed+synthetic pool CSV.
    obs_year_cutoff : int
        Real rows with year <= cutoff are excluded from observed baseline.
    hdi_prob : float
        HDI probability mass.
    propagate_rate_unc : bool
        If True, include additional uncertainty from p_fd/p_om uncertainty.
    n_mc : int
        Monte Carlo draws for error-rate propagation.
    flood_rate_threshold : float
        Exceedance threshold for high-flood scenario probability.
    n_eff_annual : float
        Effective independent daily observations per facility-year.

    Returns
    -------
    pd.DataFrame
        Facility-level uncertainty metrics.
    """

    print("Loading pool CSV...")
    df_all = pd.read_csv(pool_csv)
    print(
        f"  Rows: {len(df_all):,} | synthetic: {(df_all['is_synthetic'] == 1).sum():,} "
        f"| real: {(df_all['is_synthetic'] == 0).sum():,}"
    )

    df_syn = df_all[df_all["is_synthetic"] == 1].copy()
    df_obs = df_all[(df_all["is_synthetic"] == 0) & (df_all["year"] > obs_year_cutoff)].copy()

    meta = (
        df_all[df_all["is_synthetic"] == 0][["hf_id", "latitude", "longitude", "hf_category"]]
        .drop_duplicates("hf_id")
        .assign(hf_id=lambda d: d["hf_id"].astype(str).str.strip())
        .set_index("hf_id")
    )
    print(
        f"  Facilities: {len(meta)} | "
        + " | ".join(f"{c}:{n}" for c, n in meta["hf_category"].value_counts().items())
    )

    df_syn["hf_id"] = df_syn["hf_id"].astype(str).str.strip()
    scen_means = (
        df_syn.groupby(["hf_id", "scenario_id"], observed=True)["occurrence"]
        .mean()
        .unstack("scenario_id")
    )
    print(f"  scen_means: {scen_means.shape[0]} facilities × {scen_means.shape[1]} scenarios")

    df_obs["hf_id"] = df_obs["hf_id"].astype(str).str.strip()
    obs_means = df_obs.groupby("hf_id", observed=True)["occurrence"].mean().rename("obs_mean")
    obs_means.index = obs_means.index.astype(str).str.strip()
    print(
        f"  Observed baseline: post-{obs_year_cutoff}, "
        f"{df_obs['year'].nunique()} years, {df_obs['hf_id'].nunique()} facilities"
    )

    alpha = (1 - hdi_prob) / 2
    rows = []

    for hf_id in tqdm(scen_means.index, desc="Computing uncertainty"):
        p_obs_per_scen = scen_means.loc[hf_id].dropna().values.astype(float)
        n_scen = len(p_obs_per_scen)
        if n_scen == 0:
            continue

        p_true_per_s = correct_prob(p_obs_per_scen)
        p_obs_mean = float(p_obs_per_scen.mean())
        p_true_mean = float(p_true_per_s.mean())
        p_true_med = float(np.median(p_true_per_s))

        var_posterior = float(p_true_per_s.var(ddof=1)) if n_scen > 1 else 0.0
        var_sensor = sensor_variance_annual(p_true_mean, n_eff_annual)
        var_total = var_posterior + var_sensor
        sd_total = float(np.sqrt(var_total))

        hdi_lo = float(np.quantile(p_true_per_s, alpha))
        hdi_hi = float(np.quantile(p_true_per_s, 1 - alpha))
        exceedance = float((p_true_per_s > flood_rate_threshold).mean())

        row = {
            "hf_id": hf_id,
            "p_obs_mean": p_obs_mean,
            "p_true_mean": p_true_mean,
            "p_true_median": p_true_med,
            "bias_correction": p_true_mean - p_obs_mean,
            "var_posterior": var_posterior,
            "var_sensor": var_sensor,
            "var_total": var_total,
            "sd_total": sd_total,
            "hdi_lo": hdi_lo,
            "hdi_hi": hdi_hi,
            "exceedance_prob": exceedance,
            "n_scenarios": n_scen,
            "sd_rate_unc": np.nan,
            "sd_combined": sd_total,
        }

        if propagate_rate_unc and n_scen > 1:
            mc = sample_viirs_error_uncertainty(p_obs_per_scen, n_mc=n_mc)
            sd_rate = float(mc["p_true_sd"].mean())
            row["sd_rate_unc"] = sd_rate
            row["sd_combined"] = float(np.sqrt(var_total + sd_rate ** 2))

        rows.append(row)

    result = pd.DataFrame(rows).set_index("hf_id")
    result.index = result.index.astype(str).str.strip()
    result.attrs.update(
        {
            "hdi_prob": hdi_prob,
            "obs_year_cutoff": obs_year_cutoff,
            "flood_rate_threshold": flood_rate_threshold,
            "n_eff_annual": n_eff_annual,
        }
    )

    result = result.join(obs_means, how="left")
    result["obs_true_mean"] = correct_prob(result["obs_mean"].fillna(0).values)
    result = result.join(meta, how="left")

    n_miss = int(result["hf_category"].isna().sum())
    if n_miss:
        warnings.warn(f"{n_miss} facilities missing hf_category after join.")
    else:
        print("  Metadata join: OK, 0 missing categories.")

    safe_high = result[(result["hf_category"] == "SAFE") & (result["p_true_mean"] > 0.10)]
    if not safe_high.empty:
        print(
            f"  Note: {len(safe_high)} SAFE facilities have p_true_mean > 0.10. "
            f"Max = {safe_high['p_true_mean'].max():.3f}"
        )
    else:
        print("  Sanity check: 0 SAFE facilities with p_true_mean > 0.10.")

    return result.sort_values("p_true_mean", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_uncertainty_summary(unc_table: pd.DataFrame) -> None:
    r"""Print aggregated uncertainty summary.

    Parameters
    ----------
    unc_table : pd.DataFrame
        Facility uncertainty table from `facility_uncertainty_table`.
    """

    t = unc_table
    flood_thr = t.attrs.get("flood_rate_threshold", 0.10)
    cutoff = t.attrs.get("obs_year_cutoff", 2021)
    n_eff = t.attrs.get("n_eff_annual", 106.6)

    sum_post = t["var_posterior"].sum()
    sum_sensor = t["var_sensor"].sum()
    sum_rate = (t["sd_rate_unc"] ** 2).sum() if "sd_rate_unc" in t else 0.0
    sum_total = sum_post + sum_sensor + sum_rate + _EPS

    print("\n" + "═" * 65)
    print("  VIIRS + POSTERIOR UNCERTAINTY SUMMARY  v6")
    print("═" * 65)
    print(f"  Facilities                       : {len(t)}")
    print(f"  VIIRS false detection rate       : {P_FD_NOMINAL:.2%}")
    print(f"  VIIRS omission rate              : {P_OM_NOMINAL:.2%}")
    print(f"  Calibration denominator          : {_CALIB_DENOM:.4f}")
    print(f"  Observed baseline                : post-{cutoff}")
    print(f"  Flood rate threshold             : {flood_thr:.0%}")
    print(f"  n_eff_annual (ACF-estimated)     : {n_eff}")
    print()
    print(f"  Mean observed flood rate         : {t['p_obs_mean'].mean():.3f}")
    print(f"  Mean corrected p_true (mean)     : {t['p_true_mean'].mean():.3f}")
    print(f"  Mean corrected p_true (median)   : {t['p_true_median'].mean():.3f}")
    print(f"  Mean bias correction             : +{t['bias_correction'].mean():.3f}")
    print()
    print(f"  Mean total SD σ(p_true)          : {t['sd_total'].mean():.3f}")
    print("  Variance decomposition:")
    print(f"    Posterior (epistemic)          : {100 * sum_post / sum_total:.1f}%")
    print(f"    Sensor (aleatory)              : {100 * sum_sensor / sum_total:.1f}%")
    if sum_rate > 0:
        print(f"    Rate uncertainty               : {100 * sum_rate / sum_total:.1f}%")
    print(f"    TOTAL                          : {100 * (sum_post + sum_sensor + sum_rate) / sum_total:.1f}%")
    print()
    print(f"  Facilities P(true>{flood_thr:.0%}) > 50%  : {(t['exceedance_prob'] > 0.5).sum()}")
    print(f"  Facilities median p_true > {flood_thr:.0%}  : {(t['p_true_median'] > flood_thr).sum()}")
    print("═" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOT: BIAS CORRECTION
# ─────────────────────────────────────────────────────────────────────────────
def plot_bias_correction_effect(unc_table: pd.DataFrame, output_dir: str = ".") -> None:
    r"""Plot observed-vs-corrected scatter and correction histogram.

    Parameters
    ----------
    unc_table : pd.DataFrame
        Facility uncertainty table.
    output_dir : str
        Directory where the figure is saved.
    """

    df = unc_table.dropna(subset=["obs_mean"]).copy()
    cutoff = unc_table.attrs.get("obs_year_cutoff", 2021)
    if df.empty:
        warnings.warn("No obs_mean available. Skipping bias correction plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    lo = max(df["obs_mean"].max(), df["p_true_mean"].max()) * 1.05
    ax.plot([0, lo], [0, lo], "k--", lw=1, label="No correction", zorder=1)

    sd = df["sd_total"].values
    sd_min, sd_max = sd.min(), sd.max()
    sizes = 20 + 160 * (sd - sd_min) / (sd_max - sd_min + _EPS)

    for cat in ["SAFE", "CHRONIC", "UNSTABLE", "CRITICAL"]:
        sub = df[df["hf_category"] == cat]
        if sub.empty:
            continue
        ax.scatter(
            sub["obs_mean"],
            sub["p_true_mean"],
            c=CATEGORY_COLORS[cat],
            s=sizes[df.index.get_indexer(sub.index)],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
            label=cat,
        )

    for sd_val, lbl in zip(
        [sd_min, np.percentile(sd, 50), sd_max],
        [f"SD={sd_min:.2f}", f"SD={np.percentile(sd, 50):.2f}", f"SD={sd_max:.2f}"],
    ):
        s = 20 + 160 * (sd_val - sd_min) / (sd_max - sd_min + _EPS)
        ax.scatter([], [], c="grey", s=s, alpha=0.7, label=lbl)

    ax.set_xlabel(f"Post-{cutoff} observed flood rate (VIIRS-corrupted)")
    ax.set_ylabel("Corrected true flood rate p_true")
    ax.set_title("VIIRS bias correction per facility\nColour = category | Size = SD(p_true)")
    ax.legend(fontsize=7, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for cat in ["SAFE", "CHRONIC", "UNSTABLE", "CRITICAL"]:
        sub = df[df["hf_category"] == cat]
        if sub.empty:
            continue
        ax.hist(
            sub["bias_correction"],
            bins=20,
            color=CATEGORY_COLORS[cat],
            alpha=0.6,
            label=f"{cat} (n={len(sub)})",
            edgecolor="white",
        )
    ax.axvline(df["bias_correction"].mean(), color="black", lw=1.5, label=f"Overall mean = +{df['bias_correction'].mean():.3f}")
    ax.axvline(0, color="gray", lw=1, ls="--")
    ax.set_xlabel("p_true − p_obs (bias correction)")
    ax.set_ylabel("Number of facilities")
    ax.set_title("Distribution of VIIRS omission bias\nby facility category")
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "bias_correction_v6.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: bias_correction_v6.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. DECISION CONFIDENCE PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def _draw_quadrant_scaffold(ax, x_thr: float, sd_thr: float, y_hi: float, labels: list[str]) -> None:
    r"""Draw background quadrants and threshold lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    x_thr : float
        X-axis threshold.
    sd_thr : float
        Y-axis threshold.
    y_hi : float
        Upper y-axis limit.
    labels : list[str]
        Quadrant labels in [top-left, top-right, bottom-left, bottom-right] order.
    """

    x_lo = 0.0
    ax.fill_between([x_lo, x_thr], [0, 0], [sd_thr, sd_thr], color="#C8E6C9", alpha=0.35, zorder=0)
    ax.fill_between([x_thr, 1.0], [0, 0], [sd_thr, sd_thr], color="#FFCDD2", alpha=0.35, zorder=0)
    ax.fill_between([x_lo, x_thr], [sd_thr, sd_thr], [y_hi, y_hi], color="#FFF9C4", alpha=0.45, zorder=0)
    ax.fill_between([x_thr, 1.0], [sd_thr, sd_thr], [y_hi, y_hi], color="#FFE0B2", alpha=0.45, zorder=0)

    kw = dict(fontsize=9, color="#444444", style="italic", ha="center", va="center")
    xlo_mid = x_thr / 2
    xhi_mid = (x_thr + 1.0) / 2
    ylo_mid = sd_thr / 2
    yhi_mid = (sd_thr + y_hi) / 2
    ax.text(xlo_mid, yhi_mid, labels[0], **kw)
    ax.text(xhi_mid, yhi_mid, labels[1], **kw)
    ax.text(xlo_mid, ylo_mid, labels[2], **kw)
    ax.text(xhi_mid, ylo_mid, labels[3], **kw)
    ax.axvline(x_thr, color="#555555", lw=1.4, ls="--", zorder=2)
    ax.axhline(sd_thr, color="#555555", lw=1.4, ls="--", zorder=2)


def _scatter_by_category(ax, df: pd.DataFrame, x_col: str, size_col: str = "p_true_mean"):
    r"""Scatter facilities by category with size scaled by selected metric.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    df : pd.DataFrame
        Table with category and uncertainty columns.
    x_col : str
        X-axis column name.
    size_col : str
        Column used to scale dot sizes.

    Returns
    -------
    list
        Scatter handles for legend building.
    """

    handles = []
    for cat in ["SAFE", "CHRONIC", "UNSTABLE", "CRITICAL"]:
        sub = df[df["hf_category"] == cat]
        if sub.empty:
            continue
        sc = ax.scatter(
            sub[x_col],
            sub["sd_total"],
            c=CATEGORY_COLORS[cat],
            s=30 + 120 * sub[size_col].values,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            zorder=4,
            label=cat,
        )
        handles.append(sc)

    unknown = df[~df["hf_category"].isin(CATEGORY_COLORS)]
    if not unknown.empty:
        sc = ax.scatter(unknown[x_col], unknown["sd_total"], c="#9E9E9E", s=30, alpha=0.5, zorder=3, label="UNKNOWN")
        handles.append(sc)

    return handles


def plot_decision_confidence(
    unc_table: pd.DataFrame,
    output_dir: str = ".",
    exceedance_threshold: float = 0.50,
    sd_threshold: float = 0.15,
) -> None:
    r"""Decision plot using exceedance probability on x-axis.

    Parameters
    ----------
    unc_table : pd.DataFrame
        Facility uncertainty table.
    output_dir : str
        Output directory.
    exceedance_threshold : float
        X-axis threshold on exceedance probability.
    sd_threshold : float
        Y-axis threshold on uncertainty. If default 0.15, replaced by median SD.
    """

    df = unc_table.dropna(subset=["exceedance_prob", "sd_total"]).copy()
    flood_thr = unc_table.attrs.get("flood_rate_threshold", 0.10)
    cutoff = unc_table.attrs.get("obs_year_cutoff", 2021)

    if sd_threshold == 0.15:
        sd_threshold = float(df["sd_total"].median())

    y_hi = df["sd_total"].max() * 1.08
    fig, ax = plt.subplots(figsize=(10, 7))

    _draw_quadrant_scaffold(
        ax,
        exceedance_threshold,
        sd_threshold,
        y_hi,
        [
            "MONITOR\nlow data / ambiguous",
            "INVESTIGATE\nprobable risk, uncertain",
            "LOW RISK\nconfident, no action needed",
            "ACT\nconfident high risk",
        ],
    )

    handles = _scatter_by_category(ax, df, "exceedance_prob")

    x_thr = exceedance_threshold
    for (xc, yc, n) in [
        ((x_thr + 1) / 2, y_hi * 0.97, ((df["exceedance_prob"] > x_thr) & (df["sd_total"] > sd_threshold)).sum()),
        (x_thr / 2, y_hi * 0.97, ((df["exceedance_prob"] <= x_thr) & (df["sd_total"] > sd_threshold)).sum()),
        ((x_thr + 1) / 2, sd_threshold * 0.92, ((df["exceedance_prob"] > x_thr) & (df["sd_total"] <= sd_threshold)).sum()),
        (x_thr / 2, sd_threshold * 0.92, ((df["exceedance_prob"] <= x_thr) & (df["sd_total"] <= sd_threshold)).sum()),
    ]:
        ax.text(xc, yc, f"n = {n}", fontsize=8, fontweight="bold", ha="center", va="top", color="#222222")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_hi)
    ax.set_xlabel(f"Exceedance probability P(mean annual flood rate > {flood_thr:.0%})")
    ax.set_ylabel("Total uncertainty SD(p_true)")
    ax.set_title(
        f"Decision confidence — EXCEEDANCE-based\npost-{cutoff} | "
        f"x-thr={exceedance_threshold:.0%}, y-thr (median SD)={sd_threshold:.2f}"
    )
    ax.grid(alpha=0.2, zorder=1)

    cat_leg = ax.legend(handles=handles, title="Facility category", loc="upper left", fontsize=8, title_fontsize=8, framealpha=0.9)
    ax.add_artist(cat_leg)
    size_handles = [
        plt.scatter([], [], s=30 + 120 * p, color="#888", alpha=0.75, edgecolors="white", lw=0.4, label=f"p_true={p:.1f}")
        for p in [0.1, 0.5, 0.9]
    ]
    ax.legend(handles=size_handles, title="Dot size = p_true_mean", loc="upper right", fontsize=8, title_fontsize=8, framealpha=0.9)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "decision_confidence_exceedance_v6.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: decision_confidence_exceedance_v6.png")


def plot_decision_confidence_median(
    unc_table: pd.DataFrame,
    output_dir: str = ".",
    median_threshold: float = 0.10,
    sd_threshold: float = 0.15,
) -> None:
    r"""Decision plot using median corrected probability on x-axis.

    Parameters
    ----------
    unc_table : pd.DataFrame
        Facility uncertainty table.
    output_dir : str
        Output directory.
    median_threshold : float
        X-axis threshold on median corrected probability.
    sd_threshold : float
        Y-axis threshold on uncertainty. If default 0.15, replaced by median SD.
    """

    df = unc_table.dropna(subset=["p_true_median", "sd_total"]).copy()
    cutoff = unc_table.attrs.get("obs_year_cutoff", 2021)

    if sd_threshold == 0.15:
        sd_threshold = float(df["sd_total"].median())

    y_hi = df["sd_total"].max() * 1.08
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

    _draw_quadrant_scaffold(
        ax,
        median_threshold,
        sd_threshold,
        y_hi,
        [
            "MONITOR\nlow median, uncertain",
            "INVESTIGATE\nhigh median, uncertain",
            "LOW RISK\nlow median, confident",
            "ACT\nhigh median, confident",
        ],
    )

    handles = _scatter_by_category(ax, df, "p_true_median")

    x_thr = median_threshold
    for (xc, yc, n) in [
        ((x_thr + 1) / 2, y_hi * 0.97, ((df["p_true_median"] > x_thr) & (df["sd_total"] > sd_threshold)).sum()),
        (x_thr / 2, y_hi * 0.97, ((df["p_true_median"] <= x_thr) & (df["sd_total"] > sd_threshold)).sum()),
        ((x_thr + 1) / 2, sd_threshold * 0.92, ((df["p_true_median"] > x_thr) & (df["sd_total"] <= sd_threshold)).sum()),
        (x_thr / 2, sd_threshold * 0.92, ((df["p_true_median"] <= x_thr) & (df["sd_total"] <= sd_threshold)).sum()),
    ]:
        ax.text(xc, yc, f"n = {n}", fontsize=8, fontweight="bold", ha="center", va="top", color="#222222")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_hi)
    ax.set_xlabel("Median corrected flood probability p_true (median across scenarios)")
    ax.set_ylabel("Total uncertainty SD(p_true)")
    ax.set_title(
        f"Decision confidence — MEDIAN-based\npost-{cutoff} | "
        f"x-thr={median_threshold:.0%}, y-thr (median SD)={sd_threshold:.2f}"
    )
    ax.grid(alpha=0.2, zorder=1)

    cat_leg = ax.legend(
        handles=handles,
        title="Facility category",
        loc="upper left",
        bbox_to_anchor=(-0.22, 1.0),
        fontsize=8,
        title_fontsize=8,
        framealpha=0.9,
    )
    ax.add_artist(cat_leg)

    size_handles = [
        plt.scatter([], [], s=30 + 120 * p, color="#888", alpha=0.75, edgecolors="white", lw=0.4, label=f"p_true={p:.1f}")
        for p in [0.1, 0.5, 0.9]
    ]
    ax.legend(
        handles=size_handles,
        title="Dot size = p_true_mean",
        loc="upper right",
        bbox_to_anchor=(1.22, 1.0),
        fontsize=8,
        title_fontsize=8,
        framealpha=0.9,
    )

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "decision_confidence_median_v6.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: decision_confidence_median_v6.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. UNCERTAINTY DECOMPOSITION AND SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────
def decompose_uncertainty(unc_table: pd.DataFrame, output_dir: str = ".") -> None:
    r"""Plot uncertainty decomposition for top-uncertainty facilities.

    Parameters
    ----------
    unc_table : pd.DataFrame
        Facility uncertainty table.
    output_dir : str
        Output directory.
    """

    df = unc_table.sort_values("sd_total", ascending=False).head(80).copy()

    has_cat = "hf_category" in df.columns
    cat_colors_per_facility = [CATEGORY_COLORS.get(c, "#9E9E9E") for c in (df["hf_category"] if has_cat else ["#9E9E9E"] * len(df))]

    fig, axes = plt.subplots(2, 2, figsize=(15, 7), gridspec_kw={"height_ratios": [12, 1], "hspace": 0.04, "wspace": 0.25})
    ax_bar, ax_hdi = axes[0, 0], axes[0, 1]
    ax_strip_l, ax_strip_r = axes[1, 0], axes[1, 1]

    x = np.arange(len(df))
    bar_w = 0.8

    ax_bar.bar(x, df["var_sensor"], label="Sensor (aleatory)", color="#E07B54", alpha=0.85, width=bar_w)
    ax_bar.bar(x, df["var_posterior"], bottom=df["var_sensor"], label="Posterior (epistemic)", color="#4B9CD3", alpha=0.85, width=bar_w)
    if "sd_rate_unc" in df.columns and df["sd_rate_unc"].notna().any():
        ax_bar.bar(x, df["sd_rate_unc"] ** 2, bottom=df["var_sensor"] + df["var_posterior"], label="Rate uncertainty", color="#8E44AD", alpha=0.75, width=bar_w)
    ax_bar.set_xticks([])
    ax_bar.set_ylabel("Variance Var(p_true)")
    ax_bar.set_title("Uncertainty decomposition — top 80 facilities by total SD")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis="y", alpha=0.3)

    hdi_prob = unc_table.attrs.get("hdi_prob", 0.90)
    for i, (_, row) in enumerate(df.iterrows()):
        cat = row.get("hf_category", "UNKNOWN") if has_cat else "UNKNOWN"
        color = CATEGORY_COLORS.get(cat, "#9E9E9E")
        ax_hdi.errorbar(
            i,
            row["p_true_mean"],
            yerr=[[row["p_true_mean"] - row["hdi_lo"]], [row["hdi_hi"] - row["p_true_mean"]]],
            fmt="o",
            ms=4,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=0,
            alpha=0.75,
        )
    ax_hdi.set_xticks([])
    ax_hdi.set_ylabel("True flood probability p_true")
    ax_hdi.set_title(f"Corrected flood probability ({int(hdi_prob * 100)}% HDI)")
    ax_hdi.axhline(0.10, color="firebrick", ls="--", lw=1.2, label="10% threshold")
    ax_hdi.grid(axis="y", alpha=0.3)

    cat_handles = [
        plt.scatter([], [], s=30, color=CATEGORY_COLORS[c], label=c)
        for c in ["SAFE", "CHRONIC", "UNSTABLE", "CRITICAL"]
        if c in (df["hf_category"].values if has_cat else [])
    ]
    ax_hdi.legend(handles=cat_handles + [plt.Line2D([0], [0], color="firebrick", ls="--", lw=1.2, label="10% threshold")], fontsize=7, framealpha=0.9)

    for ax_strip in (ax_strip_l, ax_strip_r):
        for i, col in enumerate(cat_colors_per_facility):
            ax_strip.barh(0, 1, left=i, height=1, color=col, linewidth=0)
        ax_strip.set_xlim(0, len(df))
        ax_strip.set_ylim(0, 1)
        ax_strip.set_yticks([])
        ax_strip.set_xticks([])
        for spine in ax_strip.spines.values():
            spine.set_visible(False)

    ax_strip_l.set_xlabel("Facilities sorted by total SD (colour = category)")
    ax_strip_r.set_xlabel("Facilities sorted by total SD (colour = category)")

    cat_patch_handles = [
        mpatches.Patch(color=CATEGORY_COLORS[c], label=c)
        for c in ["SAFE", "CHRONIC", "UNSTABLE", "CRITICAL"]
        if c in (df["hf_category"].values if has_cat else [])
    ]
    fig.legend(
        handles=cat_patch_handles,
        title="Facility category",
        loc="lower center",
        ncol=4,
        fontsize=9,
        title_fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "uncertainty_decomposition_v6.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: uncertainty_decomposition_v6.png")


def sensitivity_to_viirs_errors(output_dir: str = ".") -> None:
    r"""Generate heatmap sensitivity of corrected p_true to p_fd/p_om values.

    Parameters
    ----------
    output_dir : str
        Output directory.
    """

    p_obs_grid = np.array([0.05, 0.10, 0.20, 0.30, 0.50])
    p_fds = np.linspace(0.001, 0.02, 20)
    p_oms = np.linspace(0.05, 0.40, 20)

    records = []
    for p_obs in p_obs_grid:
        for pfd in p_fds:
            for pom in p_oms:
                denom = 1 - pom - pfd
                if denom < 0.05:
                    continue
                pt = float(np.clip((p_obs - pfd) / denom, 0, 1))
                records.append({"p_obs": p_obs, "p_fd": pfd, "p_om": pom, "p_true": pt})

    df = pd.DataFrame(records)
    sub = df[df["p_obs"].round(2) == 0.10]
    pivot = sub.pivot_table(index="p_om", columns="p_fd", values="p_true")

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[0.001, 0.02, 0.05, 0.40],
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.5,
    )
    ax.axhline(P_OM_NOMINAL, color="white", ls="--", lw=1.5, label=f"p_om={P_OM_NOMINAL}")
    ax.axvline(P_FD_NOMINAL, color="cyan", ls="--", lw=1.5, label=f"p_fd={P_FD_NOMINAL}")
    plt.colorbar(im, ax=ax, label="p_true (corrected)")
    ax.set_xlabel("False detection rate p_fd")
    ax.set_ylabel("Omission rate p_om")
    ax.set_title("Corrected flood prob p_true | p_obs = 0.10")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sensitivity_heatmap_v6.png"), dpi=150)
    plt.close(fig)
    print("  Saved: sensitivity_heatmap_v6.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    POOL_CSV = "../data/model_output/bayfloodgen_output.csv"
    PLOT_DIR = "../data/uncertainty/combined_uncertainty_6"
    os.makedirs(PLOT_DIR, exist_ok=True)

    unc_table = facility_uncertainty_table(
        pool_csv=POOL_CSV,
        obs_year_cutoff=2021,
        hdi_prob=0.90,
        propagate_rate_unc=True,
        n_mc=2_000,
        flood_rate_threshold=0.10,
        n_eff_annual=106.6,
    )
    unc_table.to_csv(os.path.join(PLOT_DIR, "facility_uncertainty_v6.csv"))

    print_uncertainty_summary(unc_table)
    decompose_uncertainty(unc_table, output_dir=PLOT_DIR)
    plot_bias_correction_effect(unc_table, output_dir=PLOT_DIR)
    sensitivity_to_viirs_errors(output_dir=PLOT_DIR)
    plot_decision_confidence(unc_table, output_dir=PLOT_DIR, exceedance_threshold=0.50)
    plot_decision_confidence_median(unc_table, output_dir=PLOT_DIR, median_threshold=0.10)
