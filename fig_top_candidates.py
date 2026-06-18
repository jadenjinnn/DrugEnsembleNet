#!/usr/bin/env python3
"""
Top-N predicted HD candidates as a slide-ready table illustration.
Columns: rank, drug, DrugBank ID, Borda score, # cell types predicting it, key targets.
Known HD clinical-trial drugs (true positives) are highlighted with a star.
Run from repo root:  ./.venv/bin/python fig_top_candidates.py [N]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = Path("figures"); OUT.mkdir(exist_ok=True)
INK, ACCENT, GOLD, GOLDBG, MUTE = "#1d2433", "#264653", "#c8951b", "#fff3cf", "#6b7280"

TP = {"DB17924","DB00470","DB00148","DB21549","DB16975","DB04844","DB11947","DB06819",
      "DB11725","DB01017","DB01043","DB14509","DB00313","DB06685","DB08387","DB09270",
      "DB13978","DB15155","DB01156","DB00915","DB12161","DB11340","DB02709","DB18165",
      "DB17870","DB00334","DB13025","DB21645","DB01065","DB05565","DB16977","DB01039",
      "DB12116","DB11677","DB00740","DB16968","DB08887","DB00734","DB00514","DB00908",
      "DB00289","DB00215","DB11915"}

d = pd.read_csv("data/results/MASTER_DRUG_RANKINGS_enriched.csv")
d["isTP"] = d["DrugBank_ID"].astype(str).isin(TP)
n_total, n_tp_all = len(d), int(d["isTP"].sum())
top = d.head(N).copy()
n_tp_top = int(top["isTP"].sum())
n_tp_100 = int(d.head(100)["isTP"].sum())


def targets(row, k=4):
    s = str(row.Targets_Summary)
    if not s or s == "nan":
        return "—"
    syms = [t.split(" (")[0] for t in s.split(", ") if t]
    total = int(row.Human_Protein_Target_Count) if not pd.isna(row.Human_Protein_Target_Count) else len(syms)
    shown = ", ".join(syms[:k])
    return shown + (f"  +{total - k} more" if total > k else "")


def short(name, m=24):
    name = str(name)
    return name if len(name) <= m else name[:m - 1] + "…"


# layout: column left-edges and alignment in axes (0..1) coords
COLS = [
    ("#",            0.012, "center", 0.030),
    ("Drug",         0.050, "left",   0.205),
    ("DrugBank ID",  0.265, "left",   0.105),
    ("Borda",        0.375, "center", 0.075),
    ("Cell types",   0.452, "center", 0.090),
    ("Key targets",  0.560, "left",   0.430),
]
row_h = 1.0
n_rows = N + 1  # header + data
fig_h = 1.7 + 0.40 * N
fig, ax = plt.subplots(figsize=(13.5, fig_h))
ax.set_xlim(0, 1); ax.set_ylim(0, n_rows + 2.4); ax.axis("off")

# ---- title + summary banner ----
ax.text(0.0, n_rows + 1.7, f"Top {N} predicted HD repurposing candidates",
        fontsize=20, fontweight="bold", color=INK)
ax.text(0.0, n_rows + 1.05,
        f"★  {n_tp_top} of the top {N} are known HD clinical-trial drugs   ·   "
        f"{n_tp_100} of the top 100   ·   from {n_total:,} ranked candidates",
        fontsize=12.5, color=GOLD, fontweight="bold")

top_y = n_rows + 0.3  # header row center

# ---- header ----
ax.add_patch(Rectangle((0, top_y - 0.45), 1, 0.9, fc=ACCENT, ec="none"))
for name, x, align, w in COLS:
    hx = x + (w / 2 if align == "center" else 0.004)
    ax.text(hx, top_y, name, ha=align, va="center", fontsize=11.5,
            fontweight="bold", color="white")

# ---- data rows ----
for i, row in enumerate(top.itertuples()):
    y = top_y - 1.0 - i
    is_tp = row.isTP
    if is_tp:
        ax.add_patch(Rectangle((0, y - 0.5), 1, 1.0, fc=GOLDBG, ec="none"))
        ax.add_patch(Rectangle((0, y - 0.5), 0.006, 1.0, fc=GOLD, ec="none"))
    elif i % 2 == 0:
        ax.add_patch(Rectangle((0, y - 0.5), 1, 1.0, fc="#f7f8fa", ec="none"))

    rank_txt = f"{row.Rank}  ★" if is_tp else f"{row.Rank}"
    vals = [
        (rank_txt, "center", INK, "bold" if is_tp else "normal"),
        (short(row.DrugBank_Name), "left", INK, "bold"),
        (str(row.DrugBank_ID), "left", MUTE, "normal"),
        (f"{row.Borda_Score:.1f}", "center", ACCENT, "bold"),
        (f"{int(row.Predicted_Cell_Type_Count)} / 21", "center", INK, "normal"),
        (targets(row), "left", "#374151", "normal"),
    ]
    for (name, x, align, w), (txt, al, col, weight) in zip(COLS, vals):
        tx = x + (w / 2 if al == "center" else 0.004)
        fs = 10 if name != "Key targets" else 9.3
        fam = "DejaVu Sans Mono" if name == "DrugBank ID" else "DejaVu Sans"
        ax.text(tx, y, txt, ha=al, va="center", fontsize=fs, color=col,
                fontweight=weight, family=fam)

ax.text(1.0, 0.1, "★ = known HD clinical-trial drug (true positive)   ·   "
        "Borda = consensus rank across cell types",
        ha="right", va="top", fontsize=9, color="#888")

fig.savefig(OUT / f"fig_top{N}_candidates.png", dpi=200, bbox_inches="tight")
print(f"saved figures/fig_top{N}_candidates.png  "
      f"({n_tp_top}/{N} TP in top, {n_tp_100}/100, {n_tp_all} TP among {n_total} total)")
