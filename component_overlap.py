#!/usr/bin/env python3
"""
Component overlap analysis for DrugEnsembleNet HD results.

Structure:
  data/results/<DATASET>/<cell_type>/promising_drug_candidates_igsea.csv
                                     /promising_drug_candidates_l2s2.csv
                                     /promising_drug_candidates_l2s2_updown.csv  (sometimes absent or empty)

Produces, in ./overlap_figs/ :
  - availability tally (printed): how often each component is present/empty/absent
  - overlap_pooled.png            : all drugs pooled across datasets+cell types (UNION)
  - overlap_<dataset>.png         : per-dataset Venn
  - overlap_all3present.png       : Venn over ONLY cell types where all three
                                    components returned drugs (fair agreement view)

Run from repo root:  python component_overlap.py
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3

RESULTS = Path("data/results")
OUTDIR = Path("overlap_figs"); OUTDIR.mkdir(exist_ok=True)

# Skip non-HD dataset dirs that may sit under data/results (e.g. endometriosis)
EXCLUDE_DIRS = {"osktosky", "oskotsky"}

COMPONENTS = {
    "iGSEA": "promising_drug_candidates_igsea.csv",
    "L2S2": "promising_drug_candidates_l2s2.csv",
    "L2S2_dir": "promising_drug_candidates_l2s2_updown.csv",
}


def is_dataset_dir(d):
    if not d.is_dir() or d.name in EXCLUDE_DIRS:
        return False
    for sub in d.iterdir():
        if sub.is_dir() and (sub / COMPONENTS["iGSEA"]).exists():
            return True
    return False


def load_ids(path):
    """None=file absent; set()=present but no drugs; set(ids)=drugs present."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return set()
    except Exception as e:
        print(f"  WARN: could not read {path}: {e}")
        return None
    if "DrugBank_ID" not in df.columns or df.empty:
        return set()
    return set(df["DrugBank_ID"].astype(str))


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def draw(sets_dict, title, outpath):
    present = {k: v for k, v in sets_dict.items() if v}
    fig, ax = plt.subplots(figsize=(7, 7))
    if len(present) >= 3:
        keys = list(present.keys())[:3]
        venn3([present[keys[0]], present[keys[1]], present[keys[2]]], set_labels=keys, ax=ax)
    elif len(present) == 2:
        keys = list(present.keys())
        venn2([present[keys[0]], present[keys[1]]], set_labels=keys, ax=ax)
    elif len(present) == 1:
        k = list(present.keys())[0]
        ax.text(0.5, 0.5, f"Only {k} present ({len(present[k])} drugs)", ha="center")
    else:
        ax.text(0.5, 0.5, "No drugs", ha="center")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  saved {outpath}")


def main():
    if not RESULTS.exists():
        sys.exit(f"ERROR: {RESULTS} not found. Run from repo root.")

    datasets = sorted([d for d in RESULTS.iterdir() if is_dataset_dir(d)])
    print("Datasets found:", [d.name for d in datasets])
    if not datasets:
        sys.exit("No dataset dirs with promising CSVs found.")

    pooled = {k: set() for k in COMPONENTS}
    per_dataset = {}
    status = {k: {"absent": 0, "empty": 0, "present": 0} for k in COMPONENTS}
    n_celltypes = 0
    all3_pooled = {k: set() for k in COMPONENTS}
    n_all3 = 0

    for ds in datasets:
        ds_sets = {k: set() for k in COMPONENTS}
        for ct in [c for c in ds.iterdir() if c.is_dir()]:
            n_celltypes += 1
            this_ct = {}
            for comp, fname in COMPONENTS.items():
                ids = load_ids(ct / fname)
                if ids is None:
                    status[comp]["absent"] += 1
                    this_ct[comp] = None
                elif len(ids) == 0:
                    status[comp]["empty"] += 1
                    this_ct[comp] = set()
                else:
                    status[comp]["present"] += 1
                    this_ct[comp] = ids
                    ds_sets[comp] |= ids
                    pooled[comp] |= ids
            if all(this_ct.get(c) for c in COMPONENTS):
                n_all3 += 1
                for comp in COMPONENTS:
                    all3_pooled[comp] |= this_ct[comp]
        per_dataset[ds.name] = ds_sets

    print(f"\nTotal cell-type folders scanned: {n_celltypes}")
    print("\n=== Component availability across cell types ===")
    for comp, s in status.items():
        tot = s["absent"] + s["empty"] + s["present"]
        print(f"  {comp:9s}: present={s['present']:4d}  empty={s['empty']:4d}  absent={s['absent']:4d}   (of {tot})")
    print(f"\n  Cell types where ALL THREE returned drugs: {n_all3}/{n_celltypes}")

    print("\n=== POOLED (UNION across all datasets & cell types; overstates agreement) ===")
    for k in COMPONENTS:
        print(f"  {k}: {len(pooled[k])} unique drugs")
    print(f"  iGSEA n L2S2: {len(pooled['iGSEA'] & pooled['L2S2'])} (Jaccard {jaccard(pooled['iGSEA'], pooled['L2S2']):.3f})")
    if pooled["L2S2_dir"]:
        print(f"  iGSEA n L2S2_dir: {len(pooled['iGSEA'] & pooled['L2S2_dir'])} (Jaccard {jaccard(pooled['iGSEA'], pooled['L2S2_dir']):.3f})")
        print(f"  L2S2 n L2S2_dir:  {len(pooled['L2S2'] & pooled['L2S2_dir'])} (Jaccard {jaccard(pooled['L2S2'], pooled['L2S2_dir']):.3f})")
        print(f"  all three:        {len(pooled['iGSEA'] & pooled['L2S2'] & pooled['L2S2_dir'])}")
    draw(pooled, "Component overlap - pooled (union, all datasets & cell types)", OUTDIR / "overlap_pooled.png")

    if n_all3 > 0:
        print(f"\n=== ALL-THREE-PRESENT cell types only ({n_all3}) — fair agreement view ===")
        for k in COMPONENTS:
            print(f"  {k}: {len(all3_pooled[k])} unique drugs")
        print(f"  all three: {len(all3_pooled['iGSEA'] & all3_pooled['L2S2'] & all3_pooled['L2S2_dir'])}")
        draw(all3_pooled, f"Component overlap - cell types where all 3 returned drugs (n={n_all3})",
             OUTDIR / "overlap_all3present.png")
    else:
        print("\n(No cell type had all three return drugs - skipping all-3 Venn.)")

    print("\n=== PER-DATASET ===")
    for name, sets_dict in per_dataset.items():
        print(f"  {name}: iGSEA={len(sets_dict['iGSEA'])}, L2S2={len(sets_dict['L2S2'])}, "
              f"L2S2_dir={len(sets_dict['L2S2_dir'])}, iGSEA&L2S2 Jaccard={jaccard(sets_dict['iGSEA'], sets_dict['L2S2']):.3f}")
        draw(sets_dict, f"Component overlap - {name}", OUTDIR / f"overlap_{name.replace('/', '_')}.png")

    print("\nDone. Figures in", OUTDIR.resolve())


if __name__ == "__main__":
    main()