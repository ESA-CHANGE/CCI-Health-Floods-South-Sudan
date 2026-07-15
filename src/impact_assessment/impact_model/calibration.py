# -*- coding: utf-8 -*-
__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
Empirical threshold calibration utilities.

This module computes data-driven quantiles for the variables used by rules:
- R1 spell duration
- R2 spell intensity (mean_pct_flooded * duration)
- R3 wet-day distance to water

The output can be used to calibrate `SPELL_THRESHOLDS`,
`INTENSITY_THRESHOLDS`, `DISTANCE_CRITICAL_M`, and `DISTANCE_MODERATE_M`.
"""

import json
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from impact_model.config import (
    CATEGORY_COLORS,
    INTENSITY_THRESHOLDS,
    SPELL_THRESHOLDS,
)
from impact_model.engine import VulnerabilityConfig


def _label_spells(occ: np.ndarray) -> np.ndarray:
    r"""Label consecutive wet spells from a binary occurrence array.

    Parameters
    ----------
    occ : np.ndarray
        Binary array with 1 = flooded, 0 = dry.

    Returns
    -------
    np.ndarray
        Spell id per row (`-1` for dry rows).
    """

    spell_id = np.full(len(occ), -1, dtype=int)
    sid = 0
    i = 0
    while i < len(occ):
        if occ[i] == 1:
            start = i
            while i < len(occ) and occ[i] == 1:
                i += 1
            spell_id[start:i] = sid
            sid += 1
        else:
            i += 1
    return spell_id


def compute_spells(df_group: pd.DataFrame) -> list[dict[str, Any]]:
    r"""Compute wet-spell descriptors for one sorted facility/year series.

    Parameters
    ----------
    df_group : pd.DataFrame
        Daily records for one `(hf_id, category, year)` group with at least
        `date`, `occurrence`, `pct_flooded`, and `min_distance`.

    Returns
    -------
    list[dict[str, Any]]
        One dictionary per wet spell with:
        - `start_date`
        - `end_date`
        - `duration`
        - `mean_pct_flooded`
        - `min_distance`
        - `intensity_idx` (`mean_pct_flooded * duration`)
    """

    g = df_group.sort_values("date").copy()
    occ = g["occurrence"].to_numpy(dtype=int)
    pct = g["pct_flooded"].to_numpy(dtype=float)
    dist = g["min_distance"].to_numpy(dtype=float)
    dates = pd.to_datetime(g["date"]).to_numpy()

    spells: list[dict[str, Any]] = []
    i = 0
    n = len(g)
    while i < n:
        if occ[i] == 1:
            start = i
            while i < n and occ[i] == 1:
                i += 1
            end = i - 1

            dur = int(end - start + 1)
            mean_pct = float(np.mean(pct[start:end + 1]))
            min_dist = float(np.min(dist[start:end + 1]))
            intensity_idx = float(mean_pct * dur)

            spells.append({
                "start_date": pd.Timestamp(dates[start]),
                "end_date": pd.Timestamp(dates[end]),
                "duration": dur,
                "mean_pct_flooded": mean_pct,
                "min_distance": min_dist,
                "intensity_idx": intensity_idx,
            })
        else:
            i += 1

    return spells


def calibrate_thresholds(df_obs: pd.DataFrame) -> dict[str, Any]:
    r"""Compute empirical spell percentiles to support threshold calibration.

    This follows the initial calibration logic used in analysis:
    - build wet spells from observed day-level data,
    - compute global quantiles for duration, intensity and distance,
    - compute category-level quantile summaries,
    - run a KS check (SAFE vs CRITICAL),
    - compute annual severe-spell frequency.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observed daily dataframe with at least:
        `hf_id`, `hf_category`, `year`, `date`, `occurrence`,
        `pct_flooded`, `min_distance`.

    Returns
    -------
    dict[str, Any]
        Calibration bundle with `df_spells`, percentile dictionaries,
        category summary, and annual severe spell counts.
    """

    print("\n─── THRESHOLD CALIBRATION ───")
    all_spells: list[dict[str, Any]] = []

    required = ["hf_id", "hf_category", "year", "date", "occurrence", "pct_flooded", "min_distance"]
    missing = [c for c in required if c not in df_obs.columns]
    if missing:
        raise ValueError(f"Missing required columns for threshold calibration: {missing}")

    obs = df_obs.copy()
    if not np.issubdtype(obs["date"].dtype, np.datetime64):
        obs["date"] = pd.to_datetime(obs["date"])

    for (hf_id, cat, year), grp in obs.groupby(["hf_id", "hf_category", "year"], sort=False):
        grp = grp.sort_values("date")
        for sp in compute_spells(grp):
            sp["hf_id"] = hf_id
            sp["category"] = cat
            sp["year"] = year
            all_spells.append(sp)

    df_spells = pd.DataFrame(all_spells)
    if df_spells.empty:
        print("  ⚠  No wet spells found in observed data.")
        return {}

    print(f"\n  Total wet spells: {len(df_spells):,}")
    print(f"  Spells ≥1d: {len(df_spells):,} | ≥5d: {(df_spells['duration'] >= 5).sum()} "
          f"| ≥15d: {(df_spells['duration'] >= 15).sum()} "
          f"| ≥40d: {(df_spells['duration'] >= 40).sum()}")

    # ── Global percentiles ───────────────────────────────────────────────
    percs = [50, 75, 90, 95, 99]
    print("\n  Spell duration percentiles (days) — global:")
    dur_percs = np.percentile(df_spells["duration"], percs)
    for p, v in zip(percs, dur_percs):
        print(f"    p{p:2d}: {v:.1f} days")

    print("\n  Cumulative intensity percentiles (pct×days) — global:")
    int_percs = np.percentile(df_spells["intensity_idx"], percs)
    for p, v in zip(percs, int_percs):
        print(f"    p{p:2d}: {v:.2f}")

    print("\n  Min-distance percentiles during spell (m) — global:")
    dist_percs = np.percentile(df_spells["min_distance"], [5, 10, 25, 50])
    for p, v in zip([5, 10, 25, 50], dist_percs):
        print(f"    p{p:2d}: {v:.0f} m")

    # ── By category ──────────────────────────────────────────────────────
    print("\n  Duration percentiles p75/p90/p95 by category:")
    cat_summary = (
        df_spells.groupby("category")["duration"]
        .quantile([0.75, 0.90, 0.95])
        .unstack()
    )
    print(cat_summary.to_string())

    # ── KS test: are category distributions different? ───────────────────
    print("\n  KS test (duration): SAFE vs CRITICAL —", end=" ")
    safe_dur = df_spells[df_spells["category"] == "SAFE"]["duration"].values
    crit_dur = df_spells[df_spells["category"] == "CRITICAL"]["duration"].values
    if len(safe_dur) > 5 and len(crit_dur) > 5:
        ks_stat, ks_p = stats.ks_2samp(safe_dur, crit_dur)
        print(f"KS={ks_stat:.3f}, p={ks_p:.4f}")
    else:
        print("insufficient data")

    # ── Annual frequency of severe spells by facility ────────────────────
    df_spells["year"] = pd.to_datetime(df_spells["start_date"]).dt.year
    annual_severe = (
        df_spells[df_spells["duration"] >= 15]
        .groupby(["hf_id", "year"])
        .size()
        .reset_index(name="n_severe_spells")
    )
    print(f"\n  Mean spells ≥15 days per facility-year: "
          f"{annual_severe['n_severe_spells'].mean():.2f}" if not annual_severe.empty else
          "\n  Mean spells ≥15 days per facility-year: 0.00")

    # Suggested thresholds using existing rule structure
    spell_thresholds_suggested = [
        (round(float(dur_percs[4])), 1.00),
        (round(float(dur_percs[3])), 0.75),
        (round(float(dur_percs[2])), 0.50),
        (round(float(dur_percs[1])), 0.25),
        (round(float(dur_percs[0])), 0.10),
    ]
    intensity_thresholds_suggested = [
        (round(float(int_percs[4]), 2), 0.80),
        (round(float(int_percs[3]), 2), 0.50),
        (round(float(int_percs[2]), 2), 0.25),
        (round(float(int_percs[1]), 2), 0.10),
    ]
    distance_thresholds_suggested = {
        "distance_critical_m": round(float(dist_percs[2]), 1),
        "distance_moderate_m": round(float(dist_percs[3]), 1),
    }

    return {
        "df_spells": df_spells,
        "dur_percentiles": dict(zip(percs, dur_percs)),
        "int_percentiles": dict(zip(percs, int_percs)),
        "dist_percentiles": dict(zip([5, 10, 25, 50], dist_percs)),
        "cat_summary": cat_summary,
        "annual_severe": annual_severe,
        "spell_thresholds_suggested": spell_thresholds_suggested,
        "intensity_thresholds_suggested": intensity_thresholds_suggested,
        "distance_thresholds_suggested": distance_thresholds_suggested,
    }


def plot_threshold_calibration(calib: dict[str, Any], output_dir: str) -> None:
    r"""Plot calibration panel from observed spell statistics.

    The panel includes:
    - duration histograms by category,
    - duration boxplots by category,
    - intensity histograms by category,
    - duration vs intensity scatter by category.
    """

    os.makedirs(output_dir, exist_ok=True)
    df_sp = calib.get("df_spells") if calib else None
    if df_sp is None or df_sp.empty:
        return

    cats = ["SAFE", "CHRONIC", "UNSTABLE", "CRITICAL"]
    colors = [CATEGORY_COLORS.get(c, "#999") for c in cats]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # ── Spell duration histogram by category ─────────────────────────────
    ax = axes[0, 0]
    bins_dur = np.arange(0.5, 65.5, 1)
    for cat, col in zip(cats, colors):
        data = df_sp[df_sp["category"] == cat]["duration"]
        if len(data):
            ax.hist(data, bins=bins_dur, density=True, alpha=0.45, color=col, label=cat)
    for thr, _ in SPELL_THRESHOLDS:
        ax.axvline(thr, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Spell duration (days)")
    ax.set_ylabel("Density")
    ax.set_title("Spell duration distribution by category")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── Duration boxplot by category ─────────────────────────────────────
    ax = axes[0, 1]
    data_by_cat = [df_sp[df_sp["category"] == c]["duration"].values for c in cats]
    non_empty = [(c, d) for c, d in zip(cats, data_by_cat) if len(d) > 0]
    bp = ax.boxplot(
        [d for _, d in non_empty],
        labels=[c for c, _ in non_empty],
        patch_artist=True, notch=True, showfliers=False
    )
    for patch, col in zip(bp["boxes"], [CATEGORY_COLORS.get(c, "#999") for c, _ in non_empty]):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    for thr, _ in SPELL_THRESHOLDS:
        ax.axhline(thr, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_ylabel("Spell duration (days)")
    ax.set_title("Spell duration boxplot by category")
    ax.grid(axis="y", alpha=0.25)

    # ── Cumulative intensity histogram ───────────────────────────────────
    ax = axes[1, 0]
    q98 = float(df_sp["intensity_idx"].quantile(0.98)) if len(df_sp) else 1.0
    bins_int = np.linspace(0, q98 if q98 > 0 else 1.0, 40)
    for cat, col in zip(cats, colors):
        data = df_sp[df_sp["category"] == cat]["intensity_idx"]
        if len(data):
            ax.hist(data, bins=bins_int, density=True, alpha=0.45, color=col, label=cat)
    for thr, _ in INTENSITY_THRESHOLDS:
        ax.axvline(thr, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Intensity index (mean pct_flooded × days)")
    ax.set_ylabel("Density")
    ax.set_title("Cumulative intensity distribution by category")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── Duration vs intensity scatter by category ────────────────────────
    ax = axes[1, 1]
    for cat, col in zip(cats, colors):
        sub = df_sp[df_sp["category"] == cat]
        ax.scatter(sub["duration"], sub["intensity_idx"],
                   color=col, alpha=0.3, s=12, label=cat)
    ax.set_xlabel("Spell duration (days)")
    ax.set_ylabel("Cumulative intensity")
    ax.set_title("Spell duration vs intensity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("Threshold calibration — observed data", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "threshold_calibration.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def print_suggested_thresholds(calib: dict[str, Any]) -> None:
    r"""Print threshold suggestions extracted from calibration output."""

    if not calib:
        print("No calibration output available.")
        return

    print("\n─── Suggested thresholds from calibration ───")
    print(f"SPELL_THRESHOLDS     = {calib.get('spell_thresholds_suggested')}")
    print(f"INTENSITY_THRESHOLDS = {calib.get('intensity_thresholds_suggested')}")
    dist = calib.get("distance_thresholds_suggested", {})
    print(f"DISTANCE_CRITICAL_M  = {dist.get('distance_critical_m')}")
    print(f"DISTANCE_MODERATE_M  = {dist.get('distance_moderate_m')}")


def build_config_from_calibration(
    calib: dict[str, Any],
    base_cfg: Optional[VulnerabilityConfig] = None,
) -> VulnerabilityConfig:
    r"""Build a `VulnerabilityConfig` from calibration suggestions.

    Parameters
    ----------
    calib : dict[str, Any]
        Output returned by `calibrate_thresholds(...)`.
    base_cfg : VulnerabilityConfig | None
        Optional base config to copy/modify.

    Returns
    -------
    VulnerabilityConfig
        Config with calibrated R1/R2/R3 thresholds.
    """

    cfg = base_cfg or VulnerabilityConfig()

    spell = calib.get("spell_thresholds_suggested")
    if spell:
        cfg.spell_thresholds = spell

    inten = calib.get("intensity_thresholds_suggested")
    if inten:
        cfg.intensity_thresholds = inten

    dist = calib.get("distance_thresholds_suggested", {})
    if "distance_critical_m" in dist:
        cfg.distance_critical = float(dist["distance_critical_m"])
    if "distance_moderate_m" in dist:
        cfg.distance_moderate = float(dist["distance_moderate_m"])

    return cfg


def compute_empirical_thresholds(
    df_pool_raw: pd.DataFrame,
    synthetic_only: bool = True,
) -> dict[str, Any]:
    r"""Compute empirical calibration statistics from raw daily pool data.

    This helper keeps backward compatibility. For threshold-first calibration
    from observed data, use `calibrate_thresholds()`.

    Parameters
    ----------
    df_pool_raw : pd.DataFrame
        Raw day-level pool with at least:
        `date`, `occurrence`, `pct_flooded`, `min_distance`, and group ids
        (`hf_id`, optionally `scenario_id`).
    synthetic_only : bool, default=True
        If True and `is_synthetic` exists, only synthetic rows are used.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - spell duration quantiles (p50,p75,p90,p95,p99)
        - intensity quantiles (p75,p90,p95,p99)
        - wet distance quantiles (p10,p25,p50)
        - suggested thresholds matching current rule structure.
    """

    df = df_pool_raw.copy()

    if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])

    if synthetic_only and "is_synthetic" in df.columns:
        df = df[df["is_synthetic"] == 1].copy()

    required = ["occurrence", "pct_flooded", "min_distance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for calibration: {missing}")

    if "hf_id" not in df.columns:
        raise ValueError("Column 'hf_id' is required for calibration grouping.")

    group_cols = ["hf_id"]
    if "scenario_id" in df.columns:
        group_cols.append("scenario_id")

    spell_durations: list[float] = []
    spell_intensities: list[float] = []
    wet_distances: list[float] = []

    for _, grp in df.groupby(group_cols, sort=False):
        g = grp.sort_values("date") if "date" in grp.columns else grp
        occ = g["occurrence"].to_numpy(dtype=int)
        pct = g["pct_flooded"].to_numpy(dtype=float)
        dist = g["min_distance"].to_numpy(dtype=float)

        wet_mask = occ == 1
        if wet_mask.any():
            wet_distances.extend(dist[wet_mask].tolist())

        sid = _label_spells(occ)
        if (sid >= 0).any():
            tmp = pd.DataFrame({"sid": sid[wet_mask], "pct": pct[wet_mask]})
            if not tmp.empty:
                by_spell = tmp.groupby("sid").agg(
                    duration=("sid", "count"),
                    mean_pct=("pct", "mean"),
                )
                by_spell["intensity"] = by_spell["duration"] * by_spell["mean_pct"]
                spell_durations.extend(by_spell["duration"].astype(float).tolist())
                spell_intensities.extend(by_spell["intensity"].astype(float).tolist())

    if len(spell_durations) == 0:
        raise ValueError("No wet spells found. Cannot calibrate thresholds.")

    if len(spell_intensities) == 0:
        raise ValueError("No spell intensities found. Cannot calibrate thresholds.")

    if len(wet_distances) == 0:
        raise ValueError("No wet-day distances found. Cannot calibrate thresholds.")

    sd = np.asarray(spell_durations, dtype=float)
    it = np.asarray(spell_intensities, dtype=float)
    wd = np.asarray(wet_distances, dtype=float)

    spell_q = {
        "p50": float(np.quantile(sd, 0.50)),
        "p75": float(np.quantile(sd, 0.75)),
        "p90": float(np.quantile(sd, 0.90)),
        "p95": float(np.quantile(sd, 0.95)),
        "p99": float(np.quantile(sd, 0.99)),
    }
    intensity_q = {
        "p75": float(np.quantile(it, 0.75)),
        "p90": float(np.quantile(it, 0.90)),
        "p95": float(np.quantile(it, 0.95)),
        "p99": float(np.quantile(it, 0.99)),
    }
    distance_q = {
        "p10": float(np.quantile(wd, 0.10)),
        "p25": float(np.quantile(wd, 0.25)),
        "p50": float(np.quantile(wd, 0.50)),
    }

    # Suggested thresholds preserve the existing rule structure/order.
    spell_thresholds_suggested = [
        (round(spell_q["p99"]), 1.00),
        (round(spell_q["p95"]), 0.75),
        (round(spell_q["p90"]), 0.50),
        (round(spell_q["p75"]), 0.25),
        (round(spell_q["p50"]), 0.10),
    ]
    intensity_thresholds_suggested = [
        (round(intensity_q["p99"], 2), 0.80),
        (round(intensity_q["p95"], 2), 0.50),
        (round(intensity_q["p90"], 2), 0.25),
        (round(intensity_q["p75"], 2), 0.10),
    ]
    distance_thresholds_suggested = {
        "distance_critical_m": round(distance_q["p25"], 1),
        "distance_moderate_m": round(distance_q["p50"], 1),
    }

    return {
        "n_rows_used": int(len(df)),
        "n_spells": int(len(sd)),
        "n_wet_days": int((df["occurrence"].to_numpy(dtype=int) == 1).sum()),
        "spell_duration_quantiles": spell_q,
        "intensity_quantiles": intensity_q,
        "distance_quantiles": distance_q,
        "spell_thresholds_suggested": spell_thresholds_suggested,
        "intensity_thresholds_suggested": intensity_thresholds_suggested,
        "distance_thresholds_suggested": distance_thresholds_suggested,
    }


def save_calibration_report(calibration: dict[str, Any], output_path: str) -> None:
    r"""Save calibration dictionary to JSON report.

    DataFrames are converted to record-style dictionaries for serialization.
    """

    payload: dict[str, Any] = {}
    for k, v in calibration.items():
        if isinstance(v, pd.DataFrame):
            payload[k] = v.to_dict(orient="records")
        elif isinstance(v, pd.Series):
            payload[k] = v.to_dict()
        else:
            payload[k] = v

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
