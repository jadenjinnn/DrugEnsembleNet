#!/usr/bin/env python3
"""
Literature-evidence illustration for the top-20 predicted HD candidates.
Each drug's published support is encoded by STRENGTH and DIRECTION:
  supportive (strong/moderate/weak), double-edged, adverse, or none —
for Huntington's disease (HD) and for related neurodegeneration (AD/PD/ALS).
Curated from web-verified literature (citations in the accompanying writeup).
Run from repo root:  ./.venv/bin/python fig_top20_evidence.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path("figures"); OUT.mkdir(exist_ok=True)
INK, ACCENT, GOLD, GOLDBG = "#1d2433", "#264653", "#c8951b", "#fff3cf"

# strength/direction -> (fill, text, label)
CHIP = {
    "strong":  ("#1b7837", "white",   "Strong"),
    "moderate":("#5aae61", "white",   "Moderate"),
    "weak":    ("#a6dba0", "#14532d", "Weak"),
    "mixed":   ("#f4a259", "#5a3210", "Double-edged"),
    "adverse": ("#d6604d", "white",   "Adverse"),
    "none":    ("#d9d9d9", "#555555", "None"),
}

# rank, drug label, class, HD level, neuro level, one-line HD finding (cited), isTP
DATA = [
    (1,  "Geldanamycin",      "HSP90 inhibitor",   "strong",   "strong",  "↓ mutant-HTT aggregation in HD cells (Sittler 2001)", False),
    (2,  "Mitoxantrone",      "TOP2i / MS drug",   "weak",     "strong",  "Contracts CAG·CTG repeats (DM1); ALS TDP-43 (Fang 2019)", False),
    (3,  "MK-2206",           "AKT inhibitor",     "moderate", "moderate","AKT axis context-dependent in HD models (Liévens 2008)", False),
    (4,  "Trifluoperazine",   "D2 antag / autophagy","moderate","moderate","Autophagy ↓ polyQ aggregates (Zhang 2007)", False),
    (5,  "Idarubicin",        "TOP2i / anthracycline","weak",  "moderate","Class binds amyloid fibrils (Merlini 1995)", False),
    (6,  "Wortmannin",        "PI3K inhibitor",    "adverse",  "moderate","Blocks protective IGF-1/AKT→HTT-Ser421 (Humbert 2002)", False),
    (7,  "Tanespimycin (17-AAG)","HSP90 inhibitor","moderate", "strong",  "polyQ rescue in SBMA mice (Waza 2005); weak BBB", False),
    (8,  "Genistein",         "isoflavone / autophagy","strong","strong", "Autophagic mHTT clearance, patient fibroblasts (Pierzynowska 2019)", False),
    (9,  "Camptothecin",      "TOP1 inhibitor",    "mixed",    "adverse", "Dros HD-screen hit, but TOP1 is neurotoxic (Schulte 2011)", False),
    (10, "Valproic acid",     "HDAC inhibitor",    "strong",   "moderate","↑ survival & motor in N171-82Q HD mice (Zádori 2009)", True),
    (11, "BIIB021",           "HSP90 inhibitor",   "moderate", "moderate","Class ↓ aggregates in R6/2 (Labbadia 2011); SCA1", False),
    (12, "SN-38",             "TOP1 inhibitor",    "moderate", "weak",    "Class TOP1i lowers long-gene HTT (Shekhar 2016)", False),
    (13, "Pyrazolanthrone (SP600125)","JNK inhibitor","strong","strong",  "JNK inhibition protects MSNs from mHTT (Ribeiro 2010)", False),
    (14, "PF-431396",         "FAK/PTK2 inhibitor","mixed",    "moderate","FAK/PYK2 is REDUCED in HD — direction unclear (Giralt 2017)", False),
    (15, "Tanzisertib (CC-930)","JNK inhibitor",   "moderate", "strong",  "Class JNKi (CEP-1347) ↑ motor/BDNF R6/2 (Apostol 2008)", False),
    (16, "CC-401",            "JNK inhibitor",     "moderate", "moderate","JNK drives axonal-transport defects in HD (Morfini 2009)", False),
    (17, "Topotecan",         "TOP1 inhibitor",    "strong",   "strong",  "↓ HTT & aggregates, ↑ lifespan R6/2 (Shekhar 2016)", False),
    (18, "Daunorubicin",      "TOP2i / anthracycline","none",  "moderate","No HD data; dissolves AD tau filaments (Pickhardt 2005)", False),
    (19, "Caffeine",          "A2A antagonist",    "adverse",  "mixed",   "Higher intake → earlier HD onset (Simonin 2013)", False),
    (20, "Colforsin (forskolin)","cAMP / CREB activator","moderate","moderate","Reverses 3-NP HD-like deficits via CREB/BDNF (Mehan 2017)", False),
]

# columns (axes fraction): label/edge, width, alignment
DRUG_X, HD_X, NEU_X, FIND_X = 0.012, 0.300, 0.405, 0.515
CHIP_W = 0.092

n = len(DATA)
fig_h = 2.0 + 0.42 * n
fig, ax = plt.subplots(figsize=(15, fig_h))
ax.set_xlim(0, 1); ax.set_ylim(-1.7, n + 3.0); ax.axis("off")

# counts for banner
sup = sum(1 for *_, hd, _, _, _ in [(0,0,0,d[3],0,0,0) for d in DATA] if hd in ("strong","moderate","weak"))
n_strong = sum(1 for d in DATA if d[3] == "strong")
n_adv = sum(1 for d in DATA if d[3] == "adverse")
n_mix = sum(1 for d in DATA if d[3] == "mixed")

ax.text(0.0, n + 2.3, "Literature support for the top 20 candidates",
        fontsize=20, fontweight="bold", color=INK)
ax.text(0.0, n + 1.55,
        f"{sup} of 20 have supportive HD literature   ·   {n_strong} with strong direct HD evidence"
        f"   ·   {n_adv} adverse + {n_mix} double-edged (mechanistic caveats)",
        fontsize=12.5, color=ACCENT, fontweight="bold")

# header
hy = n + 0.55
ax.add_patch(Rectangle((0, hy - 0.42), 1, 0.84, fc=ACCENT, ec="none"))
ax.text(DRUG_X + 0.004, hy, "Drug  (mechanism)", ha="left", va="center", fontsize=11.5, fontweight="bold", color="white")
ax.text(HD_X + CHIP_W / 2, hy, "HD", ha="center", va="center", fontsize=11.5, fontweight="bold", color="white")
ax.text(NEU_X + CHIP_W / 2, hy, "Neuro", ha="center", va="center", fontsize=11.5, fontweight="bold", color="white")
ax.text(FIND_X + 0.004, hy, "Key Huntington's-disease finding", ha="left", va="center", fontsize=11.5, fontweight="bold", color="white")


def chip(x, y, level):
    fc, tc, lab = CHIP[level]
    ax.add_patch(FancyBboxPatch((x, y - 0.30), CHIP_W, 0.60,
                 boxstyle="round,pad=0.004,rounding_size=0.02", fc=fc, ec="none"))
    ax.text(x + CHIP_W / 2, y, lab, ha="center", va="center", fontsize=8.6,
            color=tc, fontweight="bold")


for i, (rank, drug, klass, hd, neu, find, is_tp) in enumerate(DATA):
    y = hy - 1.0 - i
    if is_tp:
        ax.add_patch(Rectangle((0, y - 0.5), 1, 1.0, fc=GOLDBG, ec="none"))
        ax.add_patch(Rectangle((0, y - 0.5), 0.006, 1.0, fc=GOLD, ec="none"))
    elif i % 2 == 0:
        ax.add_patch(Rectangle((0, y - 0.5), 1, 1.0, fc="#f7f8fa", ec="none"))
    star = "★ " if is_tp else ""
    ax.text(DRUG_X + 0.004, y + 0.12, f"{star}{rank}. {drug}", ha="left", va="center",
            fontsize=10.2, fontweight="bold", color=INK)
    ax.text(DRUG_X + 0.022, y - 0.20, klass, ha="left", va="center", fontsize=8.4, color="#7a8290", style="italic")
    chip(HD_X, y, hd)
    chip(NEU_X, y, neu)
    ax.text(FIND_X + 0.004, y, find, ha="left", va="center", fontsize=9.0, color="#374151")

# legend
ly = -0.6
ax.text(0.0, ly + 0.35, "Evidence strength / direction:", fontsize=9.5, fontweight="bold", color=INK)
for j, key in enumerate(["strong", "moderate", "weak", "mixed", "adverse", "none"]):
    lx = 0.16 + j * 0.135
    fc, tc, lab = CHIP[key]
    ax.add_patch(FancyBboxPatch((lx, ly + 0.18), 0.03, 0.32,
                 boxstyle="round,pad=0.002,rounding_size=0.01", fc=fc, ec="none"))
    ax.text(lx + 0.038, ly + 0.34, lab, ha="left", va="center", fontsize=8.6, color="#444")
ax.text(1.0, ly, "★ = known HD clinical-trial drug   ·   curated from web-verified literature",
        ha="right", va="center", fontsize=8.5, color="#888")

fig.savefig(OUT / "fig_top20_evidence.png", dpi=200, bbox_inches="tight")
print("saved figures/fig_top20_evidence.png")
print(f"supportive HD: {sup}/20 | strong: {n_strong} | adverse: {n_adv} | double-edged: {n_mix}")
