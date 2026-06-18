#!/usr/bin/env python3
"""
TP coverage decomposition: cell-line scope vs. data vintage vs. structural ceiling.

Of the 43 HD clinical-trial true positives, why does the pipeline recover so few?
Split the loss into causes that need different fixes:

  (S) STRUCTURAL  - no protein target in the interactome  -> the two proximity-
                    filtered arms (RePurposeNet-*) can NEVER return it (likely
                    ASOs/biologics/newer agents). Fixable only by directional L2S2.
  (V) VINTAGE     - has targets, but no compound signature in the 2017 LINCS data
                    at all -> iGSEA can't see it; only a LINCS-2020 upgrade helps.
  (C) CELL-LINE   - in 2017 LINCS, but only in cell lines NOT in the 12 used here
                    -> recovered by widening cell-line scope, no data upgrade.
  (A) AVAILABLE   - in 2017 LINCS in a used cell line -> iGSEA could see it; if it
                    still wasn't scored/selected the loss is downstream (filters/rank).

Realized coverage (igsea_scored/selected, l2s2_scored/selected) is read exactly
from the result CSVs (pipeline's own inchikey mapping). 2017-availability is
name-matched against LINCS sig_info (validated: every realized-scored TP must
name-match; any that don't are folded in so we never undercount).

Local-only (no DrugBank/LINCS class load). Run from repo root.
"""
import glob
from pathlib import Path
import pandas as pd

RESULTS = Path("data/results")
LINCS = Path("data/sources/LINCS")
SIG_INFO = [
    LINCS / "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz",
    LINCS / "GSE92742_Broad_LINCS_sig_info.txt.gz",
]
EXCLUDE = "osktosky"

TRUE_POS = {"DB17924","DB00470","DB00148","DB21549","DB16975","DB04844","DB11947",
            "DB06819","DB11725","DB01017","DB01043","DB14509","DB00313","DB06685",
            "DB08387","DB09270","DB13978","DB15155","DB01156","DB00915","DB12161",
            "DB11340","DB02709","DB18165","DB17870","DB00334","DB13025","DB21645",
            "DB01065","DB05565","DB16977","DB01039","DB12116","DB11677","DB00740",
            "DB16968","DB08887","DB00734","DB00514","DB00908","DB00289","DB00215","DB11915"}


def union_ids(pattern):
    out = set()
    for f in glob.glob(pattern):
        if EXCLUDE in f:
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "DrugBank_ID" in df.columns:
            out |= set(df["DrugBank_ID"].astype(str))
    return out & TRUE_POS


def id_to_name():
    """DrugBank_ID -> name, pooled from every result file that carries both."""
    names = {}
    for f in glob.glob("data/results/*/*/*.csv"):
        if EXCLUDE in f:
            continue
        try:
            df = pd.read_csv(f, comment="#")
        except Exception:
            continue
        if {"DrugBank_ID", "DrugBank_Name"} <= set(df.columns):
            for i, n in zip(df["DrugBank_ID"].astype(str), df["DrugBank_Name"].astype(str)):
                if i in TRUE_POS and i not in names and n and n != "nan":
                    names[i] = n
    return names


def main():
    names = id_to_name()

    # has a protein target in the interactome <=> appears in network_proximities universe
    has_target = set()
    for f in glob.glob("data/results/*/*/network_proximities.csv"):
        if EXCLUDE in f:
            continue
        df = pd.read_csv(f, comment="#")
        has_target |= set(df["DrugBank_ID"].astype(str)) & TRUE_POS

    # realized coverage (exact, pipeline inchikey mapping)
    ig_scored = union_ids("data/results/*/*/IGSEA_results.csv")
    ig_sel    = union_ids("data/results/*/*/promising_drug_candidates_igsea.csv")
    l2_scored = union_ids("data/results/*/*/l2s2_results.csv")
    l2_sel    = union_ids("data/results/*/*/promising_drug_candidates_l2s2.csv")

    # realized iGSEA cell-line scope
    used_cells = set()
    for f in glob.glob("data/results/*/*/IGSEA_results.csv"):
        if EXCLUDE in f:
            continue
        used_cells |= set(pd.read_csv(f)["Cell_Line"].dropna().astype(str))

    # 2017 LINCS compound signatures: drug name (lower) -> set(cell_id)
    name2cells = {}
    for p in SIG_INFO:
        si = pd.read_csv(p, sep="\t", compression="gzip",
                         usecols=lambda c: c in ("pert_iname", "pert_type", "cell_id"),
                         low_memory=False)
        si = si[si["pert_type"].isin(["trt_cp", "trt_lig"])]
        for nm, cell in zip(si["pert_iname"].astype(str).str.lower(), si["cell_id"].astype(str)):
            name2cells.setdefault(nm, set()).add(cell)

    def in_2017_cells(tp):
        nm = names.get(tp, "").lower()
        return name2cells.get(nm, set())

    rows = []
    for tp in sorted(TRUE_POS):
        cells2017 = in_2017_cells(tp)
        # validation/safety: if iGSEA actually scored it, it IS in 2017 data even if
        # the name didn't match -> treat as available in a used cell line.
        realized_here = tp in ig_scored
        in2017 = bool(cells2017) or realized_here
        in_used = bool(cells2017 & used_cells) or realized_here

        if tp not in has_target:
            bucket = "S_structural_no_target"
        elif not in2017:
            bucket = "V_vintage_not_in_2017"
        elif not in_used:
            bucket = "C_cellline_other_lines_only"
        else:
            bucket = "A_available_to_iGSEA"

        rows.append({
            "DrugBank_ID": tp, "name": names.get(tp, "<unknown>"),
            "has_interactome_target": tp in has_target,
            "in_2017_LINCS": in2017, "n_2017_cells": len(cells2017),
            "in_used_cells": in_used, "bucket": bucket,
            "igsea_scored": tp in ig_scored, "igsea_selected": tp in ig_sel,
            "l2s2_scored": tp in l2_scored, "l2s2_selected": tp in l2_sel,
            "cells_2017": ",".join(sorted(cells2017)),
        })

    df = pd.DataFrame(rows)
    df.to_csv("tp_coverage_decomp.csv", index=False)
    print(f"Wrote tp_coverage_decomp.csv  | used cell lines (n={len(used_cells)}): "
          f"{sorted(used_cells)}\n")

    print("=== 43 HD true positives, by why they are (un)recoverable ===")
    order = ["S_structural_no_target", "V_vintage_not_in_2017",
             "C_cellline_other_lines_only", "A_available_to_iGSEA"]
    labels = {
        "S_structural_no_target":      "STRUCTURAL  no interactome target (proximity arms can't ever return)",
        "V_vintage_not_in_2017":       "VINTAGE     has target, absent from 2017 LINCS (needs data upgrade)",
        "C_cellline_other_lines_only": "CELL-LINE   in 2017 LINCS but only in UNused cell lines (widen scope)",
        "A_available_to_iGSEA":        "AVAILABLE   in 2017 LINCS in a used cell line (iGSEA could see it)",
    }
    for b in order:
        g = df[df["bucket"] == b]
        print(f"  [{len(g):2d}] {labels[b]}")
        for _, r in g.iterrows():
            print(f"        {r['DrugBank_ID']}  {r['name']}")
    print()

    print("=== Realized recovery (union across all HD strata) ===")
    print(f"  iGSEA scored {len(ig_scored)} TPs, selected {len(ig_sel)}")
    print(f"  L2S2  scored {len(l2_scored)} TPs, selected {len(l2_sel)}")
    print(f"  combined SELECTED union: {len(ig_sel | l2_sel)} of 43  "
          f"(= the actual recall ceiling of the two arms as configured)")
    print()

    print("=== Answers to the two questions ===")
    n_struct = (df["bucket"] == "S_structural_no_target").sum()
    n_vint   = (df["bucket"] == "V_vintage_not_in_2017").sum()
    n_cell   = (df["bucket"] == "C_cellline_other_lines_only").sum()
    n_avail  = (df["bucket"] == "A_available_to_iGSEA").sum()
    vintage_gain = (l2_scored - {r["DrugBank_ID"] for r in rows if r["in_2017_LINCS"]})
    print(f"  CELL-LINE SCOPE cost: {n_cell} TP(s) sit in 2017 LINCS but only in cell")
    print(f"    lines you did NOT use -> widening cell-line scope could recover them,")
    print(f"    no data upgrade needed.")
    print(f"  DATA VINTAGE: {n_vint} TP(s) have targets but are absent from 2017 LINCS;")
    print(f"    {len(vintage_gain)} TP(s) that L2S2 (2020) scored are not in the 2017 data at all")
    print(f"    -> only a LINCS-2020 upgrade brings these into iGSEA's reach.")
    print(f"  STRUCTURAL ceiling: {n_struct} TP(s) have no interactome target (biologics/")
    print(f"    ASOs/newer agents) -> neither proximity arm can ever return them.")
    print(f"  AVAILABLE: {n_avail} TP(s) were visible to iGSEA in a used cell line.")


if __name__ == "__main__":
    main()
