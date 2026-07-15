# -*- coding: utf-8 -*-

__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

"""
BayFloodGEN validation utilities.

This script contains functions for validating the generated synthetic scenarios by
comparing them to the observed data. It includes visualizations to assess the
monthly frequency of flooding, the distribution of wet spell lengths, and the
distribution of the percentage of area flooded. The function generates and saves
plots that show how well the synthetic scenarios match the observed data in terms of
these key characteristics, providing insights into the realism of the generated scenarios.
"""

import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# ===================
# 4. VALIDATION PLOTS
# ===================
def validate_scenarios(df_obs: pd.DataFrame, df_syn: pd.DataFrame,
                       output_dir: str = "."):
    r"""This function performs validation of the generated synthetic scenarios by
    comparing them to the observed data. It creates visualizations to assess the
    monthly frequency of flooding, the distribution of wet spell lengths, and the
    distribution of the percentage of area flooded. The function generates and saves
    plots that show how well the synthetic scenarios match the observed data in terms of
    these key characteristics, providing insights into the realism of the generated scenarios.

    Parameters
    ----------
    df_obs : pd.DataFrame
        DataFrame containing the observed data with columns:
        - 'hf_id', 'hf_payam', 'latitude', 'longitude', 'buffer_pixels', 'date', 'occurrence',
        'pct_flooded', 'min_distance', 'day_of_year', 'year', 'hf_category'
    df_syn : pd.DataFrame
        DataFrame containing the synthetic scenarios with columns:
        - 'hf_id', 'hf_payam', 'latitude', 'longitude', 'buffer_pixels', 'date', 'occurrence',
        'pct_flooded', 'min_distance', 'day_of_year', 'year', 'scenario_id', 'is_synthetic'
    output_dir : str
        Directory where the validation plots will be saved. Defaults to the current directory.

    """

    os.makedirs(output_dir, exist_ok=True)

    df_obs2 = df_obs.copy()
    df_syn2 = df_syn.copy()
    df_obs2['month'] = pd.to_datetime(df_obs2['date']).dt.month
    df_syn2['month'] = pd.to_datetime(df_syn2['date']).dt.month

    obs_monthly = df_obs2.groupby('month')['occurrence'].mean()
    syn_monthly = (
        df_syn2.groupby(['scenario_id', 'month'])['occurrence']
        .mean().reset_index()
        .groupby('month')['occurrence']
        .agg(['median', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)])
    )
    syn_monthly.columns = ['median', 'q05', 'q95']

    # Plot monthly frequency with confidence intervals from synthetic scenarios
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(syn_monthly.index, syn_monthly['q05'], syn_monthly['q95'],
                    alpha=0.25, color='steelblue', label='Synthetic 5–95%')
    ax.plot(syn_monthly.index, syn_monthly['median'], color='steelblue', lw=2, label='Synthetic median')
    ax.plot(obs_monthly.index, obs_monthly.values, color='firebrick', lw=2, marker='o', label='Observed')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_ylabel('P(flood)')
    ax.set_title('Monthly flood frequency')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'val_monthly_frequency_v2_4.png'), dpi=150)
    plt.close(fig)
    print("  Saved: val_monthly_frequency_v2_4.png")

    def spell_lengths(occ):
        r"""This helper function calculates the lengths of consecutive flooding
        spells (occurrence == 1) from a binary array of occurrences. It iterates
        through the array, counting the length of each wet spell and appending it
        to the list of lengths when a dry day (occurrence == 0) is encountered.
        At the end, it returns a list of wet spell lengths.

        Parameters
        ----------
        occ : numpy.ndarray
            Array of binary values indicating occurrence (1 for flooded, 0 for not flooded).

        Returns
        -------
        list
            List of lengths of consecutive wet spells (occurrence == 1).
        """

        lengths, count = [], 0
        for v in occ:
            if v == 1:
                count += 1
            elif count > 0:
                lengths.append(count)
                count = 0
        if count > 0:
            lengths.append(count)
        return lengths

    obs_spells = []
    for _, g in df_obs.sort_values('date').groupby('hf_id'):
        obs_spells.extend(spell_lengths(g['occurrence'].values))

    syn_spells_all = []
    for _, g_scen in df_syn.groupby('scenario_id'):
        sc = []
        for _, g in g_scen.sort_values('date').groupby('hf_id'):
            sc.extend(spell_lengths(g['occurrence'].values))
        syn_spells_all.append(sc)
    syn_spells_flat = [s for sc in syn_spells_all for s in sc]

    # Plot distribution of flooding spell lengths with confidence intervals from synthetic scenarios
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.arange(0.5, 31.5, 1)
    axes[0].hist(obs_spells, bins=bins, density=True, color='firebrick', alpha=0.7, label='Observed')
    axes[0].hist(syn_spells_flat, bins=bins, density=True, color='steelblue', alpha=0.5, label='Synthetic')
    axes[0].set_xlabel('Wet spell duration (days)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    ks = np.arange(1, 31)
    obs_e = [np.mean(np.array(obs_spells) >= k) for k in ks]
    syn_e = np.array([[np.mean(np.array(sc) >= k) if sc else 0 for k in ks] for sc in syn_spells_all])
    axes[1].plot(ks, obs_e, color='firebrick', lw=2, marker='o', markersize=4, label='Observed')
    axes[1].fill_between(ks, np.percentile(syn_e, 5, axis=0), np.percentile(syn_e, 95, axis=0),
                         alpha=0.25, color='steelblue')
    axes[1].plot(ks, np.median(syn_e, axis=0), color='steelblue', lw=2, label='Synthetic median')
    axes[1].axvline(10, color='gray', ls='--', lw=1.5, label='Threshold 10 days')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'val_spell_lengths_v2_4.png'), dpi=150)
    plt.close(fig)
    print("  Saved: val_spell_lengths_v2_4.png")

    # Plot distribution of percentage flooded on flooded days with confidence intervals from synthetic scenarios
    obs_pct = df_obs.loc[df_obs['occurrence'] == 1, 'pct_flooded'].values * 100
    syn_pct = df_syn.loc[df_syn['occurrence'] == 1, 'pct_flooded'].values * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(obs_pct, bins=np.linspace(0, 100, 41), density=True, color='firebrick', alpha=0.7, label='Observed')
    ax.hist(syn_pct, bins=np.linspace(0, 100, 41), density=True, color='steelblue', alpha=0.5, label='Synthetic')
    ax.set_xlabel('% buffer flooded')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'val_pct_flooded_v2_4.png'), dpi=150)
    plt.close(fig)
    print("  Saved: val_pct_flooded_v2_4.png")

    # ── 4. Per-facility mean flood frequency ─────────────────────────────
    obs_fac_mean = df_obs.groupby('hf_id')['occurrence'].mean()
    syn_fac_mean = (
        df_syn.groupby(['hf_id', 'scenario_id'])['occurrence']
        .mean()
        .reset_index()
        .groupby('hf_id')['occurrence']
        .median()
    )
    common = obs_fac_mean.index.intersection(syn_fac_mean.index)

    # Map hf_id → category
    hf_cat_map = (
        df_obs[['hf_id', 'hf_category']]
        .drop_duplicates('hf_id')
        .set_index('hf_id')['hf_category']
        .to_dict()
    )

    CATEGORY_COLORS = {
        'SAFE': '#4CAF50',
        'CHRONIC': '#2196F3',
        'UNSTABLE': '#FF9800',
        'CRITICAL': '#F44336',
    }
    # Plot observed vs synthetic mean flood frequency per facility, colored by category
    fig, ax = plt.subplots(figsize=(6, 6))

    cats_present = sorted(set(hf_cat_map.get(fid, 'UNKNOWN') for fid in common))
    for cat in cats_present:
        fids_cat = [fid for fid in common if hf_cat_map.get(fid) == cat]
        ax.scatter(
            obs_fac_mean[fids_cat],
            syn_fac_mean[fids_cat],
            color=CATEGORY_COLORS.get(cat, '#9E9E9E'),
            alpha=0.65, s=25, label=cat, edgecolors='none'
        )

    lim = max(obs_fac_mean[common].max(), syn_fac_mean[common].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1.5, label='1:1')
    ax.set_xlabel('Observed post-2021 mean flood frequency (per facility)')
    ax.set_ylabel('Synthetic median flood frequency')
    ax.set_title('Per-facility flood frequency\nObserved post-2021 vs Synthetic')
    ax.legend(title='Category', framealpha=0.8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'val_per_facility_v2_4.png'), dpi=150)
    plt.close(fig)
    print("  Saved: val_per_facility_v2_4.png")
    print("\nValidation completed.")
