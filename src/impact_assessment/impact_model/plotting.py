# -*- coding: utf-8 -*-
__author__ = "Macarena Merida Floriano, mmef"
__maintainer__ = "Macarena Merida Floriano, mmef"
__email__ = "mmef@gmv.com"

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats

from impact_model.config import CAT_ORDER, CATEGORY_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_ranking(ranking, output_dir, top_n=30):
    r"""Function to plot the facility ranking based on the median annual
    mean loss, including error bars for the 5th and 95th percentiles,
    and color-coded by category.
    
    Parameters
    ----------
    ranking : pd.DataFrame
        DataFrame containing the facility ranking with columns `hf_id`, `hf_category`,
        `loss_p50`, `loss_p05`, `loss_p95`.
    output_dir : str
        Directory where the plot will be saved.
    top_n : int
        Number of top facilities to include in the plot. Defaults to 30.
    """

    os.makedirs(output_dir, exist_ok=True)
    top = ranking.head(top_n).copy().reset_index()
    y = np.arange(len(top))
    cols = [CATEGORY_COLORS.get(c, "#999") for c in top["hf_category"]]
    fig, ax = plt.subplots(figsize=(11, max(5, top_n * 0.32)))
    ax.barh(y, top["loss_p50"], color=cols, alpha=0.75, height=0.6)
    ax.hlines(y, top["loss_p05"], top["loss_p95"], color=cols, lw=2.5, alpha=0.9)
    ax.scatter(top["loss_p05"], y, marker="|", color=cols, s=60, zorder=3)
    ax.scatter(top["loss_p95"], y, marker="|", color=cols, s=60, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"#{r}  {row['hf_id']}" for r, row in top.iterrows()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Annual mean loss (fraction)")
    ax.set_title(f"Top {top_n} facilities — median + 90% CI")
    for v in [0.25, 0.50, 0.75]:
        ax.axvline(v, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.grid(axis="x", alpha=0.2)
    patches = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[c], alpha=0.75, label=c)
               for c in CAT_ORDER if c in top["hf_category"].values]
    ax.legend(handles=patches, title="Category", fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, "facility_ranking.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_exceedance_heatmap(df_exc, output_dir, top_n=40):
    r"""Function to plot a heatmap of exceedance probabilities for the
    top facilities.
    
    Parameters
    ----------
    df_exc : pd.DataFrame
        DataFrame containing the exceedance table with columns `hf_id`, `hf_category`,
        and `p_exceed_{threshold}` for various thresholds.
    output_dir : str
        Directory where the plot will be saved.
    top_n : int
        Number of top facilities to include in the heatmap. Defaults to 40.
    """

    os.makedirs(output_dir, exist_ok=True)
    thr_cols = [c for c in df_exc.columns if c.startswith("p_exceed_")]
    thr_labels = [c.replace("p_exceed_", "≥").replace("pct", "%") for c in thr_cols]
    top = df_exc.head(top_n).reset_index(drop=True)
    mat = top[thr_cols].values
    fig, ax = plt.subplots(figsize=(len(thr_cols) * 1.4 + 3, max(5, top_n * 0.28)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(thr_labels)))
    ax.set_xticklabels(thr_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{r['hf_id']} [{r['hf_category'][:3]}]"
                        for _, r in top.iterrows()], fontsize=7.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if v > 0.55 else "black")
    for tick, (_, row) in zip(ax.get_yticklabels(), top.iterrows()):
        tick.set_color(CATEGORY_COLORS.get(row["hf_category"], "#333"))
    plt.colorbar(im, ax=ax, label="Exceedance probability", shrink=0.5, pad=0.02)
    ax.set_title(f"Exceedance probability — Top {top_n} facilities", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "exceedance_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_category_comparison(df_desc, df_dunn, df_scen, output_dir):
    r"""Function to plot the category comparison of annual mean loss using a violin
    plot, along with a table of descriptive statistics and the results of Dunn's test.
    
    Parameters
    ----------
    df_desc : pd.DataFrame
        DataFrame containing descriptive statistics by category with columns `category`,
        `n_obs`, `median`, `mean`, `p25`, `p75`, `p95`.
    df_dunn : pd.DataFrame
        DataFrame containing Dunn's test results with columns `group_1`, `group_2`,
        `z_stat`, `p_bonf`, `sig`.
    df_scen : pd.DataFrame
        DataFrame containing scenario statistics with columns `hf_category` and `annual_mean_loss`.
    output_dir : str
        Directory where the plot will be saved.
    """

    os.makedirs(output_dir, exist_ok=True)
    cats_present = [c for c in CAT_ORDER if c in df_scen["hf_category"].values]
    data_by_cat = [df_scen[df_scen["hf_category"] == c]["annual_mean_loss"].values
                   for c in cats_present]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    parts = ax.violinplot(data_by_cat, positions=range(len(cats_present)),
                          showmedians=True, showextrema=False)
    for pc, cat in zip(parts["bodies"], cats_present):
        pc.set_facecolor(CATEGORY_COLORS.get(cat, "#999"))
        pc.set_alpha(0.55)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)
    y_max = max(g.max() if len(g) else 0 for g in data_by_cat)
    step = y_max * 0.07
    if not df_dunn.empty:
        for ri, row in df_dunn[df_dunn["sig"] != "ns"].iterrows():
            try:
                i1 = cats_present.index(row["group_1"])
                i2 = cats_present.index(row["group_2"])
            except ValueError:
                continue
            y = y_max * 1.05 + ri * step
            ax.plot([i1, i2], [y, y], color="black", lw=1)
            ax.plot([i1, i1], [y - step * 0.2, y], color="black", lw=1)
            ax.plot([i2, i2], [y - step * 0.2, y], color="black", lw=1)
            ax.text((i1 + i2) / 2, y + step * 0.1, row["sig"], ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(cats_present)))
    ax.set_xticklabels(cats_present)
    ax.set_ylabel("Annual mean loss")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Distribution by category\n(* p<0.05, Dunn-Bonferroni)")
    ax = axes[1]
    ax.axis("off")
    cols = ["Category", "N", "Median", "Mean", "P25", "P75", "P95"]
    rows = [[r["category"], f"{int(r['n_obs']):,}", f"{r['median']:.3f}",
             f"{r['mean']:.3f}", f"{r['p25']:.3f}", f"{r['p75']:.3f}",
             f"{r['p95']:.3f}"] for _, r in df_desc.iterrows()]
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center",
                   loc="center", bbox=[0, 0.45, 1, 0.5])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif r > 0:
            cell.set_facecolor(CATEGORY_COLORS.get(rows[r - 1][0], "#fff") + "33")
    if not df_dunn.empty:
        rows_d = [[f"{r['group_1']} vs {r['group_2']}", f"{r['z_stat']:.2f}",
                   f"{r['p_bonf']:.4f}", r["sig"]] for _, r in df_dunn.iterrows()]
        tbl2 = ax.table(cellText=rows_d, colLabels=["Pair", "z", "p_bonf", "Sig."],
                        cellLoc="center", loc="center", bbox=[0, 0, 1, 0.38])
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(8.5)
        for (r, c), cell in tbl2.get_celld().items():
            if r == 0:
                cell.set_facecolor("#455A64")
                cell.set_text_props(color="white", fontweight="bold")
    fig.suptitle("Comparison between categories", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "category_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_seasonal_risk(seasonal, output_dir):
    r"""Plot the seasonal risk profile for each category.
    
    Parameters
    ----------
    seasonal : pd.DataFrame
        DataFrame containing seasonal risk analysis with columns `hf_category`, `month`,
        `p05`, `median`, `p95`.
    output_dir : str
        Directory where the plot will be saved.
    """

    os.makedirs(output_dir, exist_ok=True)
    months = range(1, 13)
    month_abbr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    for ax, cat in zip(axes.flatten(), CAT_ORDER):
        sub = seasonal[seasonal["hf_category"] == cat]
        if sub.empty:
            ax.set_visible(False)
            continue
        sub = sub.set_index("month").reindex(months)
        col = CATEGORY_COLORS.get(cat, "#999")
        ax.fill_between(months, sub["p05"], sub["p95"], alpha=0.25, color=col)
        ax.plot(months, sub["median"], color=col, lw=2.5, marker="o", markersize=5)
        if sub["median"].notna().any():
            pm = sub["median"].idxmax()
            ax.axvspan(pm - 0.45, pm + 0.45, alpha=0.12, color=col, zorder=0)
            ax.annotate(f"Pico: {month_abbr[pm - 1]}",
                        xy=(pm, sub.loc[pm, "median"]),
                        xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=8,
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
        ax.set_xticks(list(months))
        ax.set_xticklabels(month_abbr, fontsize=8)
        ax.set_ylabel("Daily mean loss")
        ax.set_title(f"Category: {cat}", fontweight="bold", color=col)
        ax.grid(alpha=0.2)
        ax.set_xlim(0.5, 12.5)
        ax.set_ylim(bottom=0)
    fig.suptitle("Seasonal risk profile", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "seasonal_risk.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_temporal_trend(trend, output_dir):
    r"""Plot the temporal trend for each category.
    
    Parameters
    ----------
    trend : pd.DataFrame
        DataFrame containing temporal trend analysis with columns `hf_category`, `year`,
        `flood_freq_med`, `mean_loss_med`, `days_gt50_med`, `days_gt75_med`.
    output_dir : str
        Directory where the plot will be saved.
    """

    if trend.empty:
        return
    os.makedirs(output_dir, exist_ok=True)
    metrics = [("flood_freq_med", "Flood frequency (frac. days/year)"),
               ("mean_loss_med", "Mean service loss"),
               ("days_gt50_med", "Days/year with loss >50%"),
               ("days_gt75_med", "Days/year with loss >75%")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (metric, ylabel) in zip(axes.flatten(), metrics):
        for cat in CAT_ORDER:
            sub = trend[trend["hf_category"] == cat].sort_values("year")
            if sub.empty:
                continue
            col = CATEGORY_COLORS.get(cat, "#999")
            ax.plot(sub["year"], sub[metric], color=col, lw=2, marker="o",
                    markersize=5, label=cat)
            if len(sub) >= 3:
                m, b, r, p, _ = stats.linregress(sub["year"], sub[metric])
                x_fit = np.array([sub["year"].min(), sub["year"].max()])
                ax.plot(x_fit, m * x_fit + b, color=col, lw=1.2,
                        ls="--" if p < 0.05 else ":", alpha=0.7)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Year")
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axes.flatten()[0].legend(title="Category", fontsize=8)
    leg_lines = [Line2D([0], [0], ls="--", color="gray", lw=1.2, label="Trend p<0.05"),
                 Line2D([0], [0], ls=":", color="gray", lw=1.2, label="Trend ns")]
    axes.flatten()[1].legend(handles=leg_lines, fontsize=8)
    fig.suptitle("Temporal trend — observed data", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "temporal_trend.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_sensitivity(sens_results, output_dir):
    r"""Plot the sensitivity of R1 thresholds for each category.
    
    Parameters
    ----------
    sens_results : dict
        Dictionary where keys are threshold labels and values are dictionaries mapping
        categories to mean R1 loss for median scenarios, e.g.:
        {
            "R1 > 0.25": {"Hospital": 0.12, "Clinic": 0.08, ...},
            "R1 > 0.50": {"Hospital": 0.05, "Clinic": 0.02, ...},
            ...
        }
    output_dir : str
        Directory where the plot will be saved.
    """

    os.makedirs(output_dir, exist_ok=True)
    labels = list(sens_results.keys())
    x = np.arange(len(CAT_ORDER))
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, cat_means) in enumerate(sens_results.items()):
        vals = [cat_means.get(c, 0.0) for c in CAT_ORDER]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=label, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CAT_ORDER)
    ax.set_ylabel("R1 mean loss (median scenarios)")
    ax.set_title("R1 threshold sensitivity by category")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, "sensitivity_r1.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
