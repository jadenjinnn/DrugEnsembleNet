#!/usr/bin/env python3
"""
Top-20 predicted HD candidates — clean single (wide) table.
Columns: DrugBank ID, Name, Cell-type count, Targets, Evidence (Author Year, Journal),
         What the evidence shows (short supports/against note).
Red rows  = mechanistically adverse / double-edged in HD.
Green row = known HD clinical-trial drug (Valproic acid).
Run from repo root:  ./.venv/bin/python fig_top20_table.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

OUT = Path("figures"); OUT.mkdir(exist_ok=True)
N = 20
INK = "#1d2433"

# DrugBank ID -> (evidence citation, short "what the evidence shows" note)
INFO = {
    "DB02424": ("Sittler 2001, Hum Mol Genet",        "↓ mutant-HTT aggregation (HSP90/HSF1)"),
    "DB00831": ("Zhang 2007, PNAS",                   "autophagy clears polyQ aggregates"),
    "DB05134": ("Waza 2005, Nat Med",                 "rescues polyQ toxicity in SBMA"),
    "DB01645": ("Pierzynowska 2019, Metab Brain Dis", "autophagic mHTT clearance (patient cells)"),
    "DB00313": ("Zádori 2009, Pharmacol Biochem Behav", "↑ survival & motor in HD mice"),
    "DB12359": ("Ding 2016, Neuroscience",            "clears polyQ ataxin-1 (SCA1); HSP90 class"),
    "DB05482": ("Shekhar 2017, Hum Mol Genet",        "TOP1 inhibition lowers HTT (class)"),
    "DB01782": ("Huang 2010, Cell Mol Neurobiol",     "JNK block protects striatal MSNs"),
    "DB11798": ("Apostol 2008, Mol Cell Neurosci",    "JNK inhibition neuroprotective (class)"),
    "DB12432": ("Morfini 2009, Nat Neurosci",         "JNK drives HD transport defects"),
    "DB01030": ("Shekhar 2017, Hum Mol Genet",        "↓ HTT & aggregates, ↑ lifespan (HD mice)"),
    "DB02587": ("Mehan 2017, Neural Regen Res",       "reverses HD-like deficits (CREB/BDNF)"),
    # evidence AGAINST / adverse direction in HD (rendered in red)
    "DB08059": ("Humbert 2002, Dev Cell",             "blocks protective AKT→HTT pathway"),
    "DB04690": ("Morris 1996, J Cell Biol",           "TOP1 inhibition is neurotoxic to neurons"),
    "DB07460": ("Giralt 2017, Nat Commun",            "FAK/PYK2 reduced in HD (wrong direction)"),
    "DB00201": ("Simonin 2013, Neurobiol Dis",        "higher caffeine intake → earlier HD onset"),
}
RED = {"DB08059", "DB04690", "DB07460", "DB00201"}   # adverse / double-edged in HD
GREEN = {"DB00313"}                                   # known HD trial drug
NAME_FIX = {"DB07460": "PF-431396"}                   # clean name for IUPAC-only entry
# show evidence for direct HD + shared-mechanism / related-disease support
# (polyQ SBMA/SCA, JNK/cAMP/autophagy neuroprotection); excludes amyloid-specific,
# mixed-direction, and adverse evidence.
SHOW = set(INFO)

df = pd.read_csv("data/results/MASTER_DRUG_RANKINGS_enriched.csv").head(N)


def targets(row, k=2):
    s = str(row.Targets_Summary)
    if not s or s == "nan":
        return "—"
    syms = [t.split(" (")[0] for t in s.split(", ") if t]
    total = int(row.Human_Protein_Target_Count) if not pd.isna(row.Human_Protein_Target_Count) else len(syms)
    return ", ".join(syms[:k]) + (f"  +{total - k}" if total > k else "")


# columns: (header, x, align, width)
COLS = [
    ("DrugBank ID", 0.008, "left",   0.072),
    ("Name",        0.084, "left",   0.140),
    ("Cell types",  0.228, "center", 0.058),
    ("Targets",     0.292, "left",   0.150),
    ("Evidence (Author Year, Journal)", 0.448, "left", 0.205),
    ("What the evidence shows",         0.660, "left", 0.335),
]

fig_h = 1.4 + 0.40 * N
fig, ax = plt.subplots(figsize=(18, fig_h))
ax.set_xlim(0, 1); ax.set_ylim(-1.2, N + 2.0); ax.axis("off")

ax.text(0.0, N + 1.45, "Top 20 predicted HD repurposing candidates",
        fontsize=18, fontweight="bold", color=INK)

hy = N + 0.5
for name, x, align, w in COLS:
    hx = x + (w / 2 if align == "center" else 0.0)
    ax.text(hx, hy, name, ha=align, va="center", fontsize=10.5, fontweight="bold", color=INK)
ax.plot([0, 1], [hy - 0.5, hy - 0.5], color="#333", lw=1.4)

for i, row in enumerate(df.itertuples()):
    y = hy - 1.0 - i
    dbid = str(row.DrugBank_ID)
    ev, note = INFO.get(dbid, ("", "")) if dbid in SHOW else ("", "")
    if dbid in GREEN:
        ax.add_patch(Rectangle((0, y - 0.5), 1, 1.0, fc="#e3f4e7", ec="none"))
        ax.add_patch(Rectangle((0, y - 0.5), 0.005, 1.0, fc="#2a9d4a", ec="none"))
    elif dbid in RED:
        ax.add_patch(Rectangle((0, y - 0.5), 1, 1.0, fc="#fdecea", ec="none"))
        ax.add_patch(Rectangle((0, y - 0.5), 0.005, 1.0, fc="#d6453a", ec="none"))
    ax.plot([0, 1], [y - 0.5, y - 0.5], color="#e3e3e3", lw=0.6)

    name = NAME_FIX.get(dbid, str(row.DrugBank_Name))
    name = name if len(name) <= 26 else name[:25] + "…"
    note_col = "#b03a2e" if dbid in RED else ("#1e7a3a" if dbid in GREEN else "#374151")
    row_cells = [
        (COLS[0], dbid, "DejaVu Sans Mono", INK, "normal"),
        (COLS[1], name, "DejaVu Sans", INK, "bold"),
        (COLS[2], f"{int(row.Predicted_Cell_Type_Count)} / 21", "DejaVu Sans", INK, "normal"),
        (COLS[3], targets(row), "DejaVu Sans", INK, "normal"),
        (COLS[4], ev, "DejaVu Sans", INK, "normal"),
        (COLS[5], note, "DejaVu Sans", note_col, "normal"),
    ]
    for (_, x, align, w), txt, fam, col, weight in row_cells:
        tx = x + (w / 2 if align == "center" else 0.0)
        ax.text(tx, y, txt, ha=align, va="center", fontsize=9.4, family=fam,
                fontweight=weight, color=col)

bottom = hy - 1.0 - (N - 1) - 0.5
ax.plot([0, 1], [bottom, bottom], color="#333", lw=1.2)
ax.text(0.0, bottom - 0.55, "Red = mechanistically adverse / double-edged in HD      "
        "Green = known HD clinical-trial drug      "
        "Evidence column: supporting (direct-HD / shared-mechanism) in dark, opposing (red rows) in red",
        ha="left", va="center", fontsize=9, color="#666")

fig.savefig(OUT / "fig_top20_table.png", dpi=200, bbox_inches="tight")
print("saved figures/fig_top20_table.png")
