# -*- coding: utf-8 -*-

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
BayFloodGEN Generator — Tier 1 Uncertainty Analysis
=====================================================
Runs four analyses on model's existing outputs (no re-fitting required):

    A) Scenario-count stability        → how many scenarios are "enough"?
    B) Facility-level posterior widths → which facilities are most uncertain?
    C) t_year_sim sensitivity          → how much does the future year assumption matter?
    D) Key metric credible intervals   → the numbers to report in your methodology

All analyses read from:
    - bayfloodgen_output.csv  (combined observed + synthetic output)
    - bayfloodgen_trace.nc   (saved ArviZ trace)

Outputs: one folder of PNGs + a summary CSV with the quantified uncertainty table.

Usage:
    python trace_analysis.py
    (or run block by block in a notebook)
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
CSV_PATH   = "../data/model_output/bayfloodgen_output.csv"
TRACE_PATH = "../data/model_output/bayfloodgen_trace.nc"
OUTPUT_DIR = "../data/uncertainty/trace_uncertainty"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colour palette — consistent across all plots
C_OBS  = "#C0392B"   # observed: red
C_SYN  = "#2980B9"   # synthetic: blue
C_BAND = "#AED6F1"   # credible interval fill


# =============================================================
# LOAD DATA
# =============================================================

print("Loading data...")
df = pd.read_csv(CSV_PATH, parse_dates=["date"])

df_obs = df[df["is_synthetic"] == 0].copy()
df_syn = df[df["is_synthetic"] == 1].copy()

n_scenarios  = df_syn["scenario_id"].nunique()
n_facilities = df_syn["hf_id"].nunique()
print(f"  Observed records  : {len(df_obs):,}")
print(f"  Synthetic records : {len(df_syn):,}  "
      f"({n_scenarios} scenarios × {n_facilities} facilities)")

# Load ArviZ trace (for parameter uncertainty plots)
print("Loading MCMC trace...")
try:
    trace = az.from_netcdf(TRACE_PATH)
    TRACE_LOADED = True
    print("  Trace loaded OK")
except Exception as e:
    TRACE_LOADED = False
    print(f"  WARNING: Could not load trace ({e}). "
          f"Skipping parameter posterior plots.")


# =============================================================
# HELPER: compute key metrics per scenario
# =============================================================

def compute_scenario_metrics(df_syn: pd.DataFrame) -> pd.DataFrame:
    r"""For each scenario, compute a set of summary statistics relevant
    to flood vulnerability rules.

    Returns a DataFrame: one row per scenario, columns = metrics.
    
    Parameters
    ----------
    df_syn : pd.DataFrame
        DataFrame containing the synthetic scenarios with columns:
        - 'hf_id', 'hf_payam', 'latitude', 'longitude', 'buffer_pixels', 'date', 'occurrence',
          'pct_flooded', 'min_distance', 'day_of_year', 'year', 'scenario_id', 'is_synthetic'
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'scenario_id': unique identifier for each synthetic scenario
        - 'annual_days': mean annual flood days per facility (averaged across facilities)
        - 'flood_freq': overall flood frequency (proportion of days flooded)
        - 'p_spell_ge10': probability of wet spells ≥ 10 days
        - 'mean_pct_wet': mean percentage of buffer flooded on wet days
    """
    
    records = []
    for scen_id, g in df_syn.groupby("scenario_id"):
        # Annual flood days per facility (mean across facilities)
        annual_days = g.groupby("hf_id")["occurrence"].sum().mean()

        # Flood frequency (proportion of days flooded)
        freq = g["occurrence"].mean()

        # Flood-spell exceedance P(spell ≥ 10 days)
        spells = _all_spell_lengths(g)
        p_10   = np.mean(np.array(spells) >= 10) if spells else 0.0

        # Mean % flooded on wet days
        wet = g[g["occurrence"] == 1]["pct_flooded"]
        mean_pct = wet.mean() * 100 if len(wet) > 0 else 0.0

        records.append({
            "scenario_id"  : scen_id,
            "annual_days"  : annual_days,
            "flood_freq"   : freq,
            "p_spell_ge10" : p_10,
            "mean_pct_wet" : mean_pct,
        })
    return pd.DataFrame(records)


def _all_spell_lengths(df_subset: pd.DataFrame) -> list:
    r"""Extract flood-spell lengths across all facilities in a subset.
    A "flood spell" is a sequence of consecutive days where "occurrence" = 1.
    
    Parameters
    ----------
    df_subset : pd.DataFrame
        Subset of the synthetic scenarios DataFrame for which to compute flood-spell lengths.
        Must contain columns 'hf_id', 'date', and 'occurrence'.
    
    Returns
    -------
    list
        List of flood-spell lengths across all facilities in the subset.
    """
    
    lengths = []
    for _, g in df_subset.groupby("hf_id"):
        occ = g.sort_values("date")["occurrence"].values
        count = 0
        for v in occ:
            if v == 1:
                count += 1
            elif count > 0:
                lengths.append(count)
                count = 0
        if count > 0:
            lengths.append(count)
    return lengths


# Pre-compute metrics for all scenarios (used in multiple analyses)
print("\nComputing per-scenario metrics...")
metrics_df = compute_scenario_metrics(df_syn)
print(metrics_df.describe().round(3).to_string())


# =============================================================
# ANALYSIS A — SCENARIO-COUNT STABILITY
# =============================================================
# Question: does the uncertainty estimate stabilise before n=50 scenarios?
# Method: repeatedly subsample n scenarios, compute CI width, plot vs n.
# If CI width flattens by n=30, you have evidence 50 is sufficient.

print("\n[A] Scenario-count stability...")

subsample_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
n_reps          = 200          # resampling repetitions per size
rng             = np.random.default_rng(42)

metric_cols = {
    "annual_days"  : "Mean annual flood days (per facility)",
    "flood_freq"   : "Flood frequency (proportion of days)",
    "p_spell_ge10" : "P(wet spell ≥ 10 days)",
    "mean_pct_wet" : "Mean % buffer flooded (wet days)",
}

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.ravel()

for ax, (col, label) in zip(axes, metric_cols.items()):
    all_vals = metrics_df[col].values
    ci_widths = []
    for n in subsample_sizes:
        widths = []
        for _ in range(n_reps):
            sample = rng.choice(all_vals, size=min(n, len(all_vals)), replace=False)
            widths.append(np.percentile(sample, 95) - np.percentile(sample, 5))
        ci_widths.append(np.mean(widths))

    ax.plot(subsample_sizes, ci_widths, color=C_SYN, lw=2.5, marker="o", markersize=5)
    ax.axvline(n_scenarios, color="gray", ls="--", lw=1.5, label=f"Current n={n_scenarios}")
    ax.set_xlabel("Number of scenarios")
    ax.set_ylabel("Mean 90% CI width")
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle(
    "Analysis A — Scenario-count stability\n"
    "Does the uncertainty estimate stabilise before n=50?",
    fontsize=12, fontweight="bold", y=1.01
)
fig.tight_layout()
path_a = os.path.join(OUTPUT_DIR, "A_scenario_stability.png")
fig.savefig(path_a, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path_a}")


# =============================================================
# ANALYSIS B — FACILITY-LEVEL CREDIBLE INTERVALS (fan plot)
# =============================================================
# For each facility, show the range of annual flood days across scenarios.
# Facilities with wide fans = high uncertainty.

print("\n[B] Facility-level credible intervals...")

fac_scen = (
    df_syn
    .groupby(["hf_id", "scenario_id"])["occurrence"]
    .sum()                         # annual flood days per facility per scenario
    .reset_index()
    .rename(columns={"occurrence": "annual_days"})
)

fac_stats = (
    fac_scen
    .groupby("hf_id")["annual_days"]
    .agg(
        median  = "median",
        q05     = lambda x: x.quantile(0.05),
        q95     = lambda x: x.quantile(0.95),
        ci_width= lambda x: x.quantile(0.95) - x.quantile(0.05),
    )
    .sort_values("median")
    .reset_index()
)

# Also add observed mean for comparison
# Just use the most recent year available, to be a fair comparison to the 2026-level scenarios.
last_year = df_obs["year"].max()
obs_fac = (
    df_obs[df_obs["year"] == last_year]
    .groupby("hf_id")["occurrence"].sum()
    .reset_index()
)
obs_fac.columns = ["hf_id", "obs_days"]
fac_stats = fac_stats.merge(obs_fac, on="hf_id", how="left")

fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(fac_stats) * 0.25)))

# Left: caterpillar / forest plot
ax = axes[0]
y = np.arange(len(fac_stats))
ax.barh(y, fac_stats["q95"] - fac_stats["q05"],
        left=fac_stats["q05"], height=0.6,
        color=C_BAND, alpha=0.8, label="90% CI (synthetic)")
ax.scatter(fac_stats["median"], y,
           color=C_SYN, s=30, zorder=5, label="Synthetic median")
if "obs_days" in fac_stats.columns:
    ax.scatter(fac_stats["obs_days"], y,
               color=C_OBS, s=20, marker="D", zorder=6, label="Observed total")
ax.set_yticks(y)
ax.set_yticklabels(fac_stats["hf_id"], fontsize=6)
ax.set_xlabel("Annual flood days (simulated year 2026)")
ax.set_title("Per-facility 90% credible interval\n(sorted by median)", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="x")

# Right: CI width histogram — shows overall uncertainty distribution
ax = axes[1]
ax.hist(fac_stats["ci_width"], bins=20, color=C_SYN, alpha=0.7, edgecolor="white")
ax.axvline(fac_stats["ci_width"].median(), color=C_OBS, lw=2,
           label=f"Median CI width = {fac_stats['ci_width'].median():.1f} days")
ax.set_xlabel("90% CI width (days)")
ax.set_ylabel("Number of facilities")
ax.set_title("Distribution of CI widths across facilities", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle("Analysis B — Per-facility uncertainty (annual flood days)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
path_b = os.path.join(OUTPUT_DIR, "B_facility_credible_intervals.png")
fig.savefig(path_b, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path_b}")


# =============================================================
# ANALYSIS C — t_year_sim SENSITIVITY
# =============================================================
# t_year_sim controls "how far into the future" we're projecting.
# t=1.0 = stationary 2025-level; t=1.1 ≈ one year beyond training.
# We can test this WITHOUT re-fitting: re-run generate_scenarios
# with different t_year_sim values on the same trace.
#
# If re-running is not feasible, we approximate the sensitivity
# by re-weighting: for each posterior draw, compute the breakpoint
# transition at different t_year values. This is an analytical
# approximation of what generate_scenarios would produce.

print("\n[C] t_year_sim sensitivity (analytical approximation)...")

if TRACE_LOADED:
    posterior = trace.posterior
    kappa_vals  = posterior["kappa"].values.ravel()        # all MCMC draws
    t_break_vals = posterior["t_break"].values             # shape: (chains, draws, fac)
    delta_vals   = posterior["delta"].values               # shape: (chains, draws, fac)

    # Flatten chains × draws
    n_total = kappa_vals.shape[0]
    # reshape t_break and delta: (total_draws, n_fac)
    n_chains_  = t_break_vals.shape[0]
    n_draws_   = t_break_vals.shape[1]
    t_break_2d = t_break_vals.reshape(n_chains_ * n_draws_, -1)
    delta_2d   = delta_vals.reshape(n_chains_ * n_draws_, -1)

    t_year_values = np.arange(0.8, 1.31, 0.05)   # 2023-equivalent to 2027-equivalent

    # For each t_year value, compute the mean breakpoint contribution
    # across all facilities and posterior draws
    mean_contrib   = []
    p05_contrib    = []
    p95_contrib    = []

    for t_y in t_year_values:
        # transition shape: (total_draws, n_fac)
        transition = 1.0 / (1.0 + np.exp(-kappa_vals[:, None] * (t_y - t_break_2d)))
        contrib = delta_2d * transition          # breakpoint contribution to log-odds
        mean_per_draw = contrib.mean(axis=1)     # mean across facilities
        mean_contrib.append(mean_per_draw.mean())
        p05_contrib.append(np.percentile(mean_per_draw, 5))
        p95_contrib.append(np.percentile(mean_per_draw, 95))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(t_year_values, p05_contrib, p95_contrib,
                    color=C_BAND, alpha=0.7, label="90% posterior CI")
    ax.plot(t_year_values, mean_contrib, color=C_SYN, lw=2.5, label="Posterior mean")
    ax.axvline(1.0, color="gray", ls="--", lw=1.5, label="t=1.0 (2025-level, used in generation)")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("t_year_sim value")
    ax.set_ylabel("Mean breakpoint contribution to log-odds of flooding\n(averaged over facilities and posterior draws)")
    ax.set_title(
        "Analysis C — Sensitivity to t_year_sim\n"
        "How much does the simulated year assumption affect flood probability?",
        fontweight="bold"
    )
    # Annotate approximate years
    year_min, year_max = 2012, 2025   # update if your data differs
    for t_y in t_year_values[::2]:
        approx_year = int(year_min + t_y * (year_max - year_min))
        ax.annotate(str(approx_year),
                    xy=(t_y, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.1),
                    ha="center", fontsize=7, color="gray")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path_c = os.path.join(OUTPUT_DIR, "C_t_year_sensitivity.png")
    fig.savefig(path_c, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path_c}")
else:
    print("  Skipped (trace not loaded).")


# =============================================================
# ANALYSIS D — PARAMETER POSTERIOR WIDTHS (caterpillar plot)
# =============================================================
# Show, for the most decision-relevant parameters, how wide the
# posterior is. Wide = the model is uncertain about that parameter.
# Particularly important for delta[facility] (trend shift) and
# t_break[facility] (when the regime changed).

print("\n[D] Parameter posterior widths...")

if TRACE_LOADED:

    # ── D1: Global parameters ──────────────────────────────────────────
    global_params = [
        "beta0", "beta_lag", "beta_cos", "beta_sin",
        "beta0_P", "beta_lag_P", "phi",
        "mu_tbreak", "kappa", "sigma_fac",
    ]
    # Filter to those actually in the trace
    global_params = [p for p in global_params if p in trace.posterior]

    summary_global = az.summary(trace, var_names=global_params, hdi_prob=0.9)

    fig, ax = plt.subplots(figsize=(10, max(4, len(global_params) * 0.5)))
    y = np.arange(len(summary_global))
    means  = summary_global["mean"].values
    hdi_lo = summary_global["hdi_5%"].values
    hdi_hi = summary_global["hdi_95%"].values

    ax.barh(y, hdi_hi - hdi_lo, left=hdi_lo, height=0.5,
            color=C_BAND, alpha=0.8, label="90% HDI")
    ax.scatter(means, y, color=C_SYN, s=60, zorder=5, label="Posterior mean")
    ax.axvline(0, color="black", lw=0.8, alpha=0.5, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(summary_global.index, fontsize=9)
    ax.set_xlabel("Parameter value")
    ax.set_title(
        "Analysis D1 — Global parameter posteriors (90% HDI)\n"
        "Wider bars = more uncertainty in that parameter",
        fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    path_d1 = os.path.join(OUTPUT_DIR, "D1_global_parameter_posteriors.png")
    fig.savefig(path_d1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path_d1}")

    # ── D2: Per-facility delta (trend shift) ───────────────────────────
    if "delta" in trace.posterior:
        summary_delta = az.summary(trace, var_names=["delta"], hdi_prob=0.9)
        summary_delta = summary_delta.sort_values("mean")

        fig, ax = plt.subplots(figsize=(10, max(5, len(summary_delta) * 0.25)))
        y = np.arange(len(summary_delta))
        ax.barh(y,
                summary_delta["hdi_95%"].values - summary_delta["hdi_5%"].values,
                left=summary_delta["hdi_5%"].values,
                height=0.6, color=C_BAND, alpha=0.8, label="90% HDI")
        ax.scatter(summary_delta["mean"].values, y,
                   color=C_SYN, s=20, zorder=5, label="Posterior mean")
        ax.axvline(0, color=C_OBS, lw=1.5, ls="--",
                   label="delta=0 (no trend shift)")
        ax.set_yticks(y)
        ax.set_yticklabels(
            [idx.replace("delta[", "").replace("]", "") for idx in summary_delta.index],
            fontsize=5
        )
        ax.set_xlabel("delta (log-odds shift post-breakpoint)")
        ax.set_title(
            "Analysis D2 — Per-facility trend-shift posteriors (delta)\n"
            "Facilities crossing zero = uncertain direction of change",
            fontweight="bold"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        path_d2 = os.path.join(OUTPUT_DIR, "D2_delta_per_facility.png")
        fig.savefig(path_d2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path_d2}")

    # ── D3: R-hat heatmap (convergence overview) ───────────────────────
    full_summary = az.summary(trace, hdi_prob=0.9)
    rhat_vals    = full_summary["r_hat"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(rhat_vals, bins=30, color=C_SYN, edgecolor="white", alpha=0.8)
    ax.axvline(1.01, color="orange", lw=2, ls="--", label="R-hat = 1.01 (good)")
    ax.axvline(1.05, color=C_OBS,   lw=2, ls="--", label="R-hat = 1.05 (warning)")
    ax.set_xlabel("R-hat")
    ax.set_ylabel("Number of parameters")
    ax.set_title("R-hat distribution across all parameters\n(all bars left of 1.05 = good convergence)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ess_vals = full_summary["ess_bulk"].dropna()
    ax.hist(ess_vals, bins=30, color=C_SYN, edgecolor="white", alpha=0.8)
    ax.axvline(400,  color="orange", lw=2, ls="--", label="ESS = 400 (minimum)")
    ax.axvline(1000, color="green",  lw=2, ls="--", label="ESS = 1000 (good)")
    ax.set_xlabel("Effective Sample Size (ESS)")
    ax.set_ylabel("Number of parameters")
    ax.set_title("ESS distribution across all parameters\n(bars right of 400 = reliable estimates)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Analysis D3 — MCMC convergence diagnostics",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path_d3 = os.path.join(OUTPUT_DIR, "D3_convergence_diagnostics.png")
    fig.savefig(path_d3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path_d3}")

else:
    print("  Skipped (trace not loaded).")


# =============================================================
# ANALYSIS E — SUMMARY TABLE (the numbers to report)
# =============================================================
# Computes the quantitative uncertainty table for your methodology section.

print("\n[E] Building quantitative summary table...")

rows = []
for col, label in metric_cols.items():
    vals = metrics_df[col].values
    rows.append({
        "Metric"         : label,
        "Mean"           : round(vals.mean(), 3),
        "Std"            : round(vals.std(), 3),
        "5th percentile" : round(np.percentile(vals, 5), 3),
        "50th percentile": round(np.percentile(vals, 50), 3),
        "95th percentile": round(np.percentile(vals, 95), 3),
        "90% CI width"   : round(np.percentile(vals, 95) - np.percentile(vals, 5), 3),
        "CV (%)"         : round(100 * vals.std() / vals.mean(), 1) if vals.mean() != 0 else None,
    })

summary_table = pd.DataFrame(rows)
table_path = os.path.join(OUTPUT_DIR, "trace_uncertainty_summary_table.csv")
summary_table.to_csv(table_path, index=False)

print("\n  ── Quantitative uncertainty summary ──")
print(summary_table.to_string(index=False))
print(f"\n  Saved: {table_path}")


# =============================================================
# ANALYSIS F — MONTHLY UNCERTAINTY FAN (presentation-ready)
# =============================================================
# Classic fan chart: observed line + shaded uncertainty band.
# Directly usable as a single slide figure.

print("\n[F] Monthly uncertainty fan chart...")

df_syn2 = df_syn.copy()
df_syn2["month"] = df_syn2["date"].dt.month
df_obs2 = df_obs.copy()
df_obs2["month"] = pd.to_datetime(df_obs2["date"]).dt.month

obs_monthly = df_obs2.groupby("month")["occurrence"].mean()

syn_monthly_per_scen = (
    df_syn2.groupby(["scenario_id", "month"])["occurrence"]
    .mean()
    .reset_index()
)
syn_monthly = syn_monthly_per_scen.groupby("month")["occurrence"].agg(
    median  = "median",
    q05     = lambda x: x.quantile(0.05),
    q25     = lambda x: x.quantile(0.25),
    q75     = lambda x: x.quantile(0.75),
    q95     = lambda x: x.quantile(0.95),
)

months     = range(1, 13)
month_lbls = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(10, 5))

ax.fill_between(months, syn_monthly["q05"], syn_monthly["q95"],
                color=C_BAND, alpha=0.5, label="Synthetic 5–95% CI")
ax.fill_between(months, syn_monthly["q25"], syn_monthly["q75"],
                color=C_SYN, alpha=0.35, label="Synthetic 25–75% CI")
ax.plot(months, syn_monthly["median"],
        color=C_SYN, lw=2.5, label="Synthetic median (2026)")
ax.plot(months, obs_monthly.values,
        color=C_OBS, lw=2.5, marker="o", markersize=5, label="Observed historical")

ax.set_xticks(list(months))
ax.set_xticklabels(month_lbls)
ax.set_ylabel("Mean daily flood probability")
ax.set_xlabel("")
ax.set_title(
    "Monthly flood probability — observed vs. simulated 2026\n"
    "Shaded bands represent parameter + stochastic uncertainty",
    fontweight="bold"
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
path_f = os.path.join(OUTPUT_DIR, "F_monthly_fan_chart.png")
fig.savefig(path_f, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path_f}")


# =============================================================
# DONE
# =============================================================

print(f"""
╔══════════════════════════════════════════════════════════════╗
║            Tier 1 Uncertainty Analysis Complete              ║
╠══════════════════════════════════════════════════════════════╣
║  Output folder: {OUTPUT_DIR:<44}║
╠══════════════════════════════════════════════════════════════╣
║  Files produced:                                             ║
║    A_scenario_stability.png      — is n=50 sufficient?       ║
║    B_facility_credible_intervals — which HFs are uncertain?  ║
║    C_t_year_sensitivity.png      — future year assumption    ║
║    D1_global_parameter_posteriors — parameter uncertainty    ║
║    D2_delta_per_facility.png      — trend shift uncertainty  ║
║    D3_convergence_diagnostics.png — MCMC R-hat & ESS         ║
║    F_monthly_fan_chart.png        — presentation-ready fan   ║
║    uncertainty_summary_table.csv  — numbers for your paper   ║
╚══════════════════════════════════════════════════════════════╝
""")