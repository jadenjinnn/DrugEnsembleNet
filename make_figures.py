#!/usr/bin/env python3
"""
Presentation figures for DrugEnsembleNet, built from real results in data/results/.
Writes PNGs (200 dpi, slide-ready) into ./figures/. Each figure is independent;
a failure in one is reported and skipped so the rest still render.
Run from repo root:  ./.venv/bin/python make_figures.py
"""
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

OUT = Path("figures"); OUT.mkdir(exist_ok=True)
RESULTS = Path("data/results")
EXCLUDE = "osktosky"

# ---- consistent style ------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "font.family": "DejaVu Sans",
})
INK = "#1d2433"
C = {"igsea": "#2a9d8f", "l2s2": "#e76f51", "dir": "#5b6ee1",
     "accent": "#264653", "gold": "#e9c46a", "muted": "#9aa3b2",
     "approved": "#2a9d8f", "invest": "#e9c46a", "exper": "#e76f51", "other": "#9aa3b2"}


def save(fig, name):
    p = OUT / name
    fig.savefig(p); plt.close(fig)
    print(f"  saved {p}")


def ids_from(pattern, col="DrugBank_ID"):
    s = set()
    for f in glob.glob(pattern):
        if EXCLUDE in f:
            continue
        try:
            df = pd.read_csv(f, comment="#")
        except Exception:
            continue
        if col in df.columns:
            s |= set(df[col].astype(str))
    return s


# ===========================================================================
# Fig 1 — data at a glance
# ===========================================================================
def fig_scale():
    # interactome size (compute; dedupe undirected edges; fall back to manuscript numbers)
    try:
        e = pd.read_csv("data/sources/interactome_slim.tsv.gz", sep="\t", compression="gzip")
        a = e.iloc[:, 0].astype(str).values; b = e.iloc[:, 1].astype(str).values
        lo = np.where(a < b, a, b); hi = np.where(a < b, b, a)
        n_edges = len(pd.DataFrame({"u": lo, "v": hi}).drop_duplicates())
        n_nodes = len(set(a) | set(b))
    except Exception:
        n_nodes, n_edges = 20692, 1316837
    n_drugs = len(ids_from("data/results/*/*/network_proximities.csv"))
    n_celltypes = len(glob.glob("data/results/consensus_ALL_DATASETS_*.csv"))
    n_datasets = len([d for d in RESULTS.iterdir() if d.is_dir() and d.name != EXCLUDE
                      and any((c / "promising_drug_candidates_igsea.csv").exists()
                              for c in d.iterdir() if c.is_dir())])
    try:
        n_ranked = len(pd.read_csv("data/results/MASTER_DRUG_RANKINGS_enriched.csv"))
    except Exception:
        n_ranked = 0

    cards = [
        (f"{n_nodes:,}", "proteins", f"{n_edges:,} interactions\n7 PPI databases", C["accent"]),
        (f"{n_drugs:,}", "drugs screened", "DrugBank targets\nmapped to interactome", C["igsea"]),
        ("591,697", "LINCS 2017 signatures", "+ L2S2: 1.68M sets\n33,621 compounds", C["l2s2"]),
        (f"{n_datasets}", "HD datasets", f"{n_celltypes} cell types", C["dir"]),
        (f"{n_ranked:,}", "ranked candidates", "single master list", C["gold"]),
        ("43", "HD trial drugs", "validation set", C["muted"]),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.8)); ax.set_xlim(0, 3); ax.set_ylim(0, 2.3); ax.axis("off")
    ax.text(1.5, 2.2, "DrugEnsembleNet at a glance", ha="center",
            fontsize=21, fontweight="bold", color=INK)
    for i, (big, lab, sub, col) in enumerate(cards):
        r, cc = divmod(i, 3); x, y = cc, 1.28 - r * 0.82
        ax.add_patch(FancyBboxPatch((x + 0.06, y - 0.02), 0.88, 0.74,
                     boxstyle="round,pad=0.02,rounding_size=0.04",
                     fc="white", ec=col, lw=2.2))
        ax.add_patch(plt.Rectangle((x + 0.06, y + 0.60), 0.88, 0.12, fc=col, ec="none",
                     transform=ax.transData, clip_on=False))
        ax.text(x + 0.5, y + 0.40, big, ha="center", va="center",
                fontsize=27, fontweight="bold", color=col)
        ax.text(x + 0.5, y + 0.20, lab, ha="center", va="center", fontsize=12.5,
                fontweight="bold", color=INK)
        ax.text(x + 0.5, y + 0.04, sub, ha="center", va="center", fontsize=9.5, color="#666")
    save(fig, "fig1_data_at_a_glance.png")


# ===========================================================================
# Fig 2 — network proximity distance distribution + threshold
# ===========================================================================
def fig_proximity(stratum="data/results/RL_cdHD_E_CTRL/ast/network_proximities.csv"):
    with open(stratum) as fh:
        head = [next(fh) for _ in range(3)]
    mean_ref = float(re.search(r"mean:\s*([0-9.\-]+)", head[0]).group(1))
    std_ref = float(re.search(r"deviation:\s*([0-9.\-]+)", head[0]).group(1))
    thr = float(re.search(r"Threshold:\s*([0-9.\-]+)", head[1]).group(1))
    df = pd.read_csv(stratum, comment="#")
    d = df["Distance"].dropna().values
    prox_pct = 100 * (d < thr).mean()
    n_prox = int((d < thr).sum())

    # clip the view so the proximal candidates are visible (note any hidden tail)
    xlo = max(d.min(), -4.0); xhi = d.max() + 0.2
    bins = np.linspace(xlo, xhi, 55)
    dist_clip = d[d >= xlo]; hidden = int((d < xlo).sum())

    fig, ax = plt.subplots(figsize=(9, 5.4))
    counts, edges = np.histogram(dist_clip, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    bar_cols = [C["l2s2"] if c < thr else C["muted"] for c in centers]
    ax.bar(centers, counts, width=(edges[1] - edges[0]), color=bar_cols,
           edgecolor="white", linewidth=0.3, align="center")
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=C["l2s2"], label=f"PROXIMAL candidates ({prox_pct:.0f}%)"),
                      Patch(color=C["muted"], label="Typical drugs")]
    xs = np.linspace(xlo, xhi, 400)
    null = (1 / (std_ref * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mean_ref) / std_ref) ** 2)
    nl, = ax.plot(xs, null, color=C["accent"], lw=2.0, ls=(0, (4, 2)), alpha=0.85,
                  label="Random expectation")
    legend_handles.append(nl)
    ax.axvline(thr, color=INK, lw=2.2, ls="--")
    ax.text(thr - 0.05, ax.get_ylim()[1] * 0.92, f"threshold = {thr}",
            ha="right", fontsize=10.5, fontweight="bold", color=INK)
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("Mean shortest-path distance, drug targets → HD genes")
    ax.set_ylabel("Density")
    ax.set_title("Network proximity keeps drugs closest to HD genes")
    note = "RL_cdHD · astrocytes (representative stratum)"
    if hidden:
        note += f"   ·   {hidden} strongly-proximal drugs at distance < {xlo:.0f} not shown"
    ax.text(0.99, -0.16, note, transform=ax.transAxes, ha="right", fontsize=8.5, color="#888")
    ax.legend(handles=legend_handles, frameon=False, loc="upper right")
    save(fig, "fig2_network_proximity.png")


# ===========================================================================
# Fig 3 — method complementarity Venn
# ===========================================================================
def fig_venn():
    from matplotlib_venn import venn3, venn3_circles
    ig = ids_from("data/results/*/*/promising_drug_candidates_igsea.csv")
    l2 = ids_from("data/results/*/*/promising_drug_candidates_l2s2.csv")
    dr = ids_from("data/results/*/*/promising_drug_candidates_l2s2_updown.csv")
    fig, ax = plt.subplots(figsize=(8, 6.2))
    v = venn3([ig, l2, dr],
              set_labels=("RePurposeNet\niGSEA", "RePurposeNet\nL2S2", "Directional\nL2S2"), ax=ax)
    for patch, col in zip(["100", "010", "001"], [C["igsea"], C["l2s2"], C["dir"]]):
        if v.get_patch_by_id(patch):
            v.get_patch_by_id(patch).set_color(col); v.get_patch_by_id(patch).set_alpha(0.55)
    venn3_circles([ig, l2, dr], lw=1.2, color="#555", ax=ax)
    for t in (v.set_labels or []):
        if t: t.set_fontsize(12); t.set_fontweight("bold")
    ax.set_title("Three methods, mostly different candidates\n→ motivation for an ensemble", fontsize=14)
    ax.text(0.5, -0.04, "Pooled promising candidates across all HD strata",
            transform=ax.transAxes, ha="center", fontsize=9, color="#888")
    save(fig, "fig3_method_complementarity_venn.png")


# ===========================================================================
# Fig 4 — top predicted drugs
# ===========================================================================
def fig_top_drugs(n=15):
    d = pd.read_csv("data/results/MASTER_DRUG_RANKINGS_enriched.csv").head(n).iloc[::-1]
    def cat(g):
        g = str(g).lower()
        if "approved" in g: return ("Approved", C["approved"])
        if "investigational" in g: return ("Investigational", C["invest"])
        return ("Experimental", C["exper"])
    cols = [cat(g)[1] for g in d["Drug_Groups"]]
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    bars = ax.barh(d["DrugBank_Name"], d["Borda_Score"], color=cols, edgecolor="white")
    for b, ct in zip(bars, d["Predicted_Cell_Type_Count"]):
        ax.text(b.get_width() - 0.15, b.get_y() + b.get_height() / 2,
                f"{int(ct)} cell types", va="center", ha="right", fontsize=8, color="white")
    ax.set_xlabel("Borda consensus score (higher = stronger across cell types)")
    ax.set_title(f"Top {n} predicted HD repurposing candidates")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C["approved"], label="Approved"),
                       Patch(color=C["invest"], label="Investigational"),
                       Patch(color=C["exper"], label="Experimental")],
              frameon=False, loc="lower right", title="Drug status")
    save(fig, "fig4_top_predicted_drugs.png")


# ===========================================================================
# Fig 5 — drug × cell-type heatmap
# ===========================================================================
def fig_heatmap(n=20):
    master = pd.read_csv("data/results/MASTER_DRUG_RANKINGS_enriched.csv").head(n)
    top_ids = master["DrugBank_ID"].astype(str).tolist()
    id2name = dict(zip(master["DrugBank_ID"].astype(str), master["DrugBank_Name"]))
    mat = {}
    for f in sorted(glob.glob("data/results/consensus_ALL_DATASETS_*.csv")):
        ct = Path(f).stem.replace("consensus_ALL_DATASETS_", "")
        df = pd.read_csv(f)
        if {"DrugBank_ID", "Borda_Score"} - set(df.columns):
            continue
        s = df.set_index(df["DrugBank_ID"].astype(str))["Borda_Score"]
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)   # min-max within cell type
        mat[ct] = {i: s.get(i, np.nan) for i in top_ids}
    M = pd.DataFrame(mat).reindex(top_ids)
    M = M.loc[:, M.notna().sum().sort_values(ascending=False).index]   # busiest cell types first

    def short(i):
        nm = str(id2name.get(i, i))
        return nm if len(nm) <= 22 else nm[:20] + "…"

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.viridis.copy(); cmap.set_bad("#dcdcdc")   # grey = not predicted
    im = ax.imshow(M.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=55, ha="right", fontsize=9)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([short(i) for i in M.index], fontsize=9)
    ax.set_title("Where each top drug is predicted, by cell type")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Relative consensus rank within cell type", fontsize=9)
    ax.text(1.0, -0.16, "grey = drug not predicted in that cell type",
            transform=ax.transAxes, ha="right", fontsize=8.5, color="#888")
    save(fig, "fig5_drug_celltype_heatmap.png")


# ===========================================================================
# Fig 6 — pie: candidate composition by drug status
# ===========================================================================
def fig_pie():
    d = pd.read_csv("data/results/MASTER_DRUG_RANKINGS_enriched.csv")
    def cat(g):
        g = str(g).lower()
        if "withdrawn" in g: return "Withdrawn"
        if "approved" in g: return "Approved"
        if "investigational" in g: return "Investigational"
        if "experimental" in g: return "Experimental"
        return "Other"
    vc = d["Drug_Groups"].map(cat).value_counts()
    order = [k for k in ["Approved", "Investigational", "Experimental", "Withdrawn", "Other"] if k in vc]
    vals = [vc[k] for k in order]
    palette = {"Approved": C["approved"], "Investigational": C["invest"],
               "Experimental": C["exper"], "Withdrawn": "#b56576", "Other": C["muted"]}
    fig, ax = plt.subplots(figsize=(7.6, 6))
    wedges, _, autot = ax.pie(
        vals, labels=order, colors=[palette[k] for k in order],
        autopct=lambda p: f"{p:.0f}%\n({int(round(p*sum(vals)/100))})",
        startangle=90, counterclock=False, pctdistance=0.72,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11))
    for a in autot: a.set_fontsize(9.5); a.set_color(INK)
    n_app = vc.get("Approved", 0)
    ax.text(0, 0, f"{len(d):,}\ncandidates", ha="center", va="center",
            fontsize=15, fontweight="bold", color=INK)
    ax.set_title(f"{n_app} candidates are already-approved drugs\n(ready for repurposing)", fontsize=13.5)
    save(fig, "fig6_candidate_composition_pie.png")


if __name__ == "__main__":
    print("Generating figures into ./figures/ ...")
    for fn in (fig_scale, fig_proximity, fig_venn, fig_top_drugs, fig_heatmap, fig_pie):
        try:
            fn()
        except Exception as e:
            print(f"  SKIP {fn.__name__}: {type(e).__name__}: {e}")
    print("Done.")
