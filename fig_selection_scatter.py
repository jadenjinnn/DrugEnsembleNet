#!/usr/bin/env python3
"""
2-D selection scatter: how proximity AND enrichment jointly pick candidates.
  x = network-proximity z-score (more negative = closer to HD genes)
  y = iGSEA enrichment significance, -log10(FDR), best signature per drug
A drug becomes a RePurposeNet-iGSEA candidate only in the upper-left box
(proximal AND enriched). HD clinical-trial true positives are starred.
Representative stratum: RL_cdHD, astrocytes.
Run:  ./.venv/bin/python fig_selection_scatter.py
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

CT = Path("data/results/RL_cdHD_E_CTRL/ast")
OUT = Path("figures"); OUT.mkdir(exist_ok=True)
INK = "#1d2433"
TEAL, GREY, GOLD = "#2a9d8f", "#c2c7cd", "#e9a000"

TRUE_POS = {"DB17924","DB00470","DB00148","DB21549","DB16975","DB04844","DB11947","DB06819",
            "DB11725","DB01017","DB01043","DB14509","DB00313","DB06685","DB08387","DB09270",
            "DB13978","DB15155","DB01156","DB00915","DB12161","DB11340","DB02709","DB18165",
            "DB17870","DB00334","DB13025","DB21645","DB01065","DB05565","DB16977","DB01039",
            "DB12116","DB11677","DB00740","DB16968","DB08887","DB00734","DB00514","DB00908",
            "DB00289","DB00215","DB11915"}

# --- proximity (z-score per drug) + threshold ----------------------------
with open(CT / "network_proximities.csv") as f:
    line1, line2 = f.readline(), f.readline()
m = re.search(r"mean:\s*([0-9.\-]+).*deviation:\s*([0-9.\-]+)", line1)
mean_ref, std_ref = float(m.group(1)), float(m.group(2))
thr = float(re.search(r"Threshold:\s*([0-9.\-]+)", line2).group(1))
prox = pd.read_csv(CT / "network_proximities.csv", comment="#")
prox["DrugBank_ID"] = prox["DrugBank_ID"].astype(str)
z_thr = (thr - mean_ref) / std_ref           # proximity threshold in z units

# --- iGSEA significance (best FDR per drug) ------------------------------
def to_float(v):
    s = str(v)
    return float(s[1:]) if s.startswith("<") else float(s)

ig = pd.read_csv(CT / "IGSEA_results.csv")
ig["DrugBank_ID"] = ig["DrugBank_ID"].astype(str)
ig["FDR_f"] = ig["FDR"].map(to_float)
best = ig.groupby("DrugBank_ID")["FDR_f"].min().reset_index()

df = prox.merge(best, on="DrugBank_ID", how="inner")
df["y"] = -np.log10(df["FDR_f"].clip(lower=1e-4))
y_thr = -np.log10(0.25)
df["selected"] = (df["Proximity"] < z_thr) & (df["FDR_f"] < 0.25)
df["is_tp"] = df["DrugBank_ID"].isin(TRUE_POS)

# clip x for display (rare strongly-proximal outliers)
xlo, xhi = max(df["Proximity"].quantile(0.002), -6), df["Proximity"].max() + 0.3
n_sel = int(df["selected"].sum()); n_tp_sel = int((df["selected"] & df["is_tp"]).sum())
print(f"z_thr={z_thr:.3f} | selected Proximity range "
      f"[{df.loc[df.selected,'Proximity'].min():.2f}, {df.loc[df.selected,'Proximity'].max():.2f}]")

# spread the points piled at the FDR floor (-log10(1e-4)=4) so the top isn't a hard line
rng = np.random.default_rng(0)
capped = df["FDR_f"] <= 1e-4
df["yp"] = df["y"] + np.where(capped, rng.uniform(-0.12, 0.12, len(df)), 0.0)
ymax = df["y"].max()

# --- plot ----------------------------------------------------------------
from adjustText import adjust_text
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
})
fig, ax = plt.subplots(figsize=(11, 8))
top = ymax + 0.7

# "selected" region: proximal (left of z_thr) AND enriched (above y_thr) — outlined box
ax.add_patch(plt.Rectangle((xlo, y_thr), z_thr - xlo, top - y_thr, facecolor=TEAL,
             alpha=0.09, edgecolor=TEAL, lw=1.4, ls="--", zorder=0))

other = df[~df["selected"] & ~df["is_tp"]]
sel = df[df["selected"] & ~df["is_tp"]]
ax.scatter(other["Proximity"], other["yp"], s=9, c=GREY, alpha=0.40,
           edgecolors="none", zorder=1, label="Not selected")
ax.scatter(sel["Proximity"], sel["yp"], s=30, c=TEAL, alpha=0.9,
           edgecolors="white", linewidths=0.4, zorder=2, label="iGSEA candidate")
tp = df[df["is_tp"]]
ax.scatter(tp["Proximity"], tp["yp"], s=230, marker="*", c=GOLD,
           edgecolors=INK, linewidths=0.8, zorder=3, label="HD clinical-trial drug")
# declutter true-positive labels with adjustText (leader lines + white bbox)
texts = [ax.text(r.Proximity, r.yp, r.DrugBank_Name, fontsize=10.5, fontweight="bold",
                 color=INK, zorder=5,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
         for r in tp.itertuples()]
adjust_text(texts, x=tp["Proximity"].values, y=tp["yp"].values, ax=ax,
            force_text=(0.5, 0.9), expand=(1.3, 1.6),
            arrowprops=dict(arrowstyle="-", color="#777", lw=0.8))

ax.axvline(z_thr, color="#555", ls="--", lw=1.4)
ax.axhline(y_thr, color="#555", ls="--", lw=1.4)
ax.text(z_thr - 0.08, -0.12, "proximity threshold", ha="right", va="bottom",
        fontsize=9, color="#555", style="italic")
ax.text(xhi, y_thr + 0.06, "FDR = 0.25", ha="right", va="bottom", fontsize=9,
        color="#555", style="italic")
ax.text(xlo + 0.12, top - 0.05, f"SELECTED — proximal & enriched  (n = {n_sel})",
        fontsize=11, fontweight="bold", color="#1f6f63", va="top")

ax.set_xlim(xlo, xhi)
ax.set_ylim(-0.25, top)
ax.set_xlabel("Network-proximity z-score   ←  closer to HD genes")
ax.set_ylabel("iGSEA enrichment significance   −log₁₀(FDR)")
ax.set_title("A candidate must be BOTH proximal and enriched", fontweight="bold")
ax.text(1.0, -0.135, "RL_cdHD · astrocytes  ·  each point = one drug (868 with both scores)",
        transform=ax.transAxes, ha="right", fontsize=8.5, color="#888")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False,
          handletextpad=0.3, columnspacing=1.6)
fig.savefig(OUT / "fig_selection_scatter.png", dpi=200, bbox_inches="tight")
print(f"saved figures/fig_selection_scatter.png | drugs plotted={len(df)} "
      f"selected={n_sel} (TP among selected={n_tp_sel})")
