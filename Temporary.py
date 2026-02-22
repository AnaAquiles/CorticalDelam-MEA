# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 19:12:46 2026

@author:aquiles 
"""


'''

      Statistical test between subject to evaluate different between Info direction across 
              coditions and groups

'''

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import seaborn as sns

df = pd.read_csv("DirectMI2026-All.csv")  # ← your actual load here



# ── Setup 
pair_map = {
    "1-2": "1↔2", "2-1": "1↔2",
    "1-3": "1↔3", "3-1": "1↔3",
    "2-3": "2↔3", "3-2": "2↔3",
}
df["Cluster_Pair"] = df["Direction"].map(pair_map)
df["Group_Label"]  = df["Group"]

cluster_pairs = sorted(df["Cluster_Pair"].unique())
group_labels  = ["Eulaminated", "Dyslaminated"]
palette       = {"Eulaminated": "#4C72B0", "Dyslaminated": "#DD8452"}

condition_order = ["Baseline", "Cero Mg", "Cero Mg 2", "Cero Mg 3", "Mg", "Mg 2"]
activities      = [a for a in condition_order if a in df["Time"].unique()]

condition_colors = {
    "Baseline":  "#F5F5F5",
    "Cero Mg":   "#EAF3FB", "Cero Mg 2": "#D4E8F5", "Cero Mg 3": "#BDD9ED",
    "Mg":        "#FEF3E2", "Mg 2":      "#FDE3C0",
}

# ── Helpers 
def rank_biserial(u, n1, n2):
    return 1 - (2 * u) / (n1 * n2)

def effect_label(r):
    r = abs(r)
    return "L" if r >= 0.5 else "M" if r >= 0.3 else "S"

def draw_boxplot(ax, data, x_pos, color, width=0.25):
    """Version-safe manual boxplot for a single group at a given x position."""
    if len(data) == 0:
        return
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr   = q3 - q1
    lo    = max(data.min(), q1 - 1.5 * iqr)
    hi    = min(data.max(), q3 + 1.5 * iqr)
    # Box
    ax.add_patch(plt.Rectangle(
        (x_pos - width / 2, q1), width, q3 - q1,
        linewidth=1.3, edgecolor=color,
        facecolor=color, alpha=0.3
    ))
    # Median line
    ax.plot([x_pos - width / 2, x_pos + width / 2], [med, med],
            color=color, linewidth=2)
    # Whiskers
    ax.plot([x_pos, x_pos], [lo, q1], color=color, linewidth=1.2)
    ax.plot([x_pos, x_pos], [q3, hi], color=color, linewidth=1.2)
    # Whisker caps
    for y in [lo, hi]:
        ax.plot([x_pos - width / 4, x_pos + width / 4], [y, y],
                color=color, linewidth=1.2)
    # Outliers
    outliers = data[(data < lo) | (data > hi)]
    if len(outliers):
        ax.scatter([x_pos] * len(outliers), outliers,
                   color=color, s=18, zorder=5, alpha=0.7)
    # Strip points
    jitter = np.random.uniform(-width / 4, width / 4, size=len(data))
    ax.scatter(x_pos + jitter, data, color=color,
               s=16, alpha=0.55, zorder=4, edgecolors="none")

n_comparisons = len(cluster_pairs) * len(activities) * len(group_labels)

# ── Figure 
n_rows = len(cluster_pairs)
n_cols = len(activities)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(3.8 * n_cols, 4.5 * n_rows),
    sharey="row",
    squeeze=False
)

# x layout: directions on x, groups side by side within each direction
group_offsets = {"Eulaminated": -0.2, "Dyslaminated": 0.2}   # side-by-side offset

for i, pair in enumerate(cluster_pairs):
    pair_data = df[df["Cluster_Pair"] == pair]
    dirs      = sorted(pair_data["Direction"].unique())
    x_ticks   = list(range(len(dirs)))   # 0, 1

    for j, activity in enumerate(activities):
        ax     = axes[i, j]
        subset = pair_data[pair_data["Time"] == activity].copy()
        ax.set_facecolor(condition_colors.get(activity, "#FFFFFF"))

        if subset.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey", fontsize=8)
            ax.set_xticks([])
            continue
        

        # ── Draw boxes per direction × group ─────────────────────────────────
        all_y = []
        for d_idx, direction in enumerate(dirs):
            for grp in group_labels:
                vals = subset[
                    (subset["Direction"]  == direction) &
                    (subset["Group_Label"] == grp)
                ]["MeanMI"].values
                x_pos = d_idx + group_offsets[grp]
                draw_boxplot(ax, vals, x_pos, palette[grp], width=0.28)
                all_y.extend(vals)

        all_y = np.array(all_y)
        
        # ── Guard against empty subplots ──────────────────────────────────────
        if len(all_y) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey", fontsize=8)
            ax.set_xticks([])
            continue
        
        y_range   = all_y.max() - all_y.min() if len(all_y) > 1 else 1
        y_top     = all_y.max() + y_range * 0.05
        # y_range   = all_y.max() - all_y.min() if len(all_y) > 1 else 1
        # y_top     = all_y.max() + y_range * 0.05
        step      = y_range * 0.14
        bracket_y = y_top
        

        # ── (A) Control vs Treatment — per direction (black bracket) ──────────
        for d_idx, direction in enumerate(dirs):
            g1 = subset[(subset["Direction"]   == direction) &
                        (subset["Group_Label"] == "Eulaminated")]["MeanMI"].values
            g2 = subset[(subset["Direction"]   == direction) &
                        (subset["Group_Label"] == "Dyslaminated")]["MeanMI"].values
            if len(g1) >= 2 and len(g2) >= 2:
                u, p_raw = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                p_corr   = min(p_raw * n_comparisons, 1.0)
                r        = rank_biserial(u, len(g1), len(g2))
                sig      = ("***" if p_corr < 0.001 else
                            "**"  if p_corr < 0.01  else
                            "*"   if p_corr < 0.05  else "ns")
                x1 = d_idx + group_offsets["Eulaminated"]
                x2 = d_idx + group_offsets["Dyslaminated"]
                ax.plot([x1, x1, x2, x2],
                        [bracket_y, bracket_y + step * 0.3,
                         bracket_y + step * 0.3, bracket_y],
                        color="black", linewidth=1.1)
                ax.text(d_idx, bracket_y + step * 0.35,
                        f"{sig} r={r:.2f}({effect_label(r)})",
                        ha="center", va="bottom", fontsize=6.5, color="black")
            bracket_y += step * 0.9

        # ── (B) Direction asymmetry — per group (colored bracket) ─────────────
        bracket_y += step * 0.2
        for grp in group_labels:
            g_data = subset[subset["Group_Label"] == grp]
            if len(dirs) < 2:
                continue
            d1 = g_data[g_data["Direction"] == dirs[0]]["MeanMI"].values
            d2 = g_data[g_data["Direction"] == dirs[1]]["MeanMI"].values
            if len(d1) >= 2 and len(d2) >= 2:
                u, p_raw = stats.mannwhitneyu(d1, d2, alternative="two-sided")
                p_corr   = min(p_raw * n_comparisons, 1.0)
                r        = rank_biserial(u, len(d1), len(d2))
                sig      = ("***" if p_corr < 0.001 else
                            "**"  if p_corr < 0.01  else
                            "*"   if p_corr < 0.05  else "ns")
                col  = palette[grp]
                off  = group_offsets[grp]
                x1   = 0 + off
                x2   = 1 + off
                ax.plot([x1, x1, x2, x2],
                        [bracket_y, bracket_y + step * 0.3,
                         bracket_y + step * 0.3, bracket_y],
                        color=col, linewidth=1.1)
                ax.text((x1 + x2) / 2, bracket_y + step * 0.35,
                        f"{sig} r={r:.2f}({effect_label(r)})",
                        ha="center", va="bottom", fontsize=6.5, color=col)
                bracket_y += step * 0.85

        # ── Formatting 
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(dirs, fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel(f"Cluster {pair}\nMeanMI" if j == 0 else "", fontsize=9, fontweight="bold")
        if i == 0:
            ax.set_title(activity, fontsize=10, fontweight="bold", pad=5)

# ── Legend 
legend_elements = [
    mpatches.Patch(color=palette["Eulaminated"],   label="Eulaminated"),
    mpatches.Patch(color=palette["Dyslaminated"], label="Treatment"),
    Line2D([0], [0], color="black",   lw=1.5, label="Control vs Treatment (per direction)"),
    Line2D([0], [0], color="#4C72B0", lw=1.5, label="Direction asymmetry — Control"),
    Line2D([0], [0], color="#DD8452", lw=1.5, label="Direction asymmetry — Treatment"),
    mpatches.Patch(color="white", label="Effect: S<0.3  M=0.3–0.5  L>0.5"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=8.5, bbox_to_anchor=(0.5, -0.04), frameon=True)

fig.suptitle(
    "Mean MI — Direction × Activity × Group\n"
    "(Bonferroni-corrected Mann–Whitney U | rank-biserial r)",
    fontsize=13, fontweight="bold"
)

plt.tight_layout(rect=[0, 0.06, 1, 0.97])
plt.savefig("MI_full_comparison.svg", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: MI_full_comparison.svg")
