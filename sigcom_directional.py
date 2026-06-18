#!/usr/bin/env python3
"""
Run the SigCom LINCS directional (reverser) arm standalone over the HD strata,
WITHOUT re-running proximity or iGSEA. Builds a cleaned up/down signature from each
EarlyHD DEG file, queries SigCom for top-N reversers, writes
promising_drug_candidates_sigcom.csv into the matching results dir, then rebuilds
the Borda master so the new arm is folded in.

Usage:  ./.venv/bin/python sigcom_directional.py [top_n] [log2fc] [--no-rank]
"""
import sys
from pathlib import Path

import pandas as pd

import databases
import sigcom
from run_parallel import ConsensusRanker

TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100
LFC = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].replace(".", "").isdigit() else 0.25
DO_RANK = "--no-rank" not in sys.argv

DEG_ROOT = Path("data/sources/EarlyHD")
RESULTS = Path("data/results")


def up_down(deg_csv, lfc):
    d = pd.read_csv(deg_csv)
    if "gene" not in d.columns:          # some datasets leave the gene column unnamed
        d = d.rename(columns={d.columns[0]: "gene"})
    sig = d[d["p_val_adj"] < 0.05]
    up = sig[sig["avg_log2FC"] > lfc].sort_values("avg_log2FC", ascending=False)["gene"].tolist()
    dn = sig[sig["avg_log2FC"] < -lfc].sort_values("avg_log2FC")["gene"].tolist()
    up = sigcom.clean_genes(up)[:150]
    dn = sigcom.clean_genes(dn)[:150]
    return up, dn


def main():
    print(f"SigCom directional arm | top_n={TOP_N} | |log2FC|>{LFC} | clean signatures\n")
    print("Loading DrugBank for name->ID mapping (cached)...")
    db = databases.DrugBank()
    name2id = sigcom.build_name2id(db)
    print(f"  name map: {len(name2id)} drugs\n")

    rows = []
    for dataset_dir in sorted(p for p in DEG_ROOT.iterdir() if p.is_dir()):
        for deg in sorted(dataset_dir.glob("*.csv")):
            dataset, ct = dataset_dir.name, deg.stem
            results_ct = RESULTS / dataset / ct
            if not results_ct.exists():
                continue  # only strata that were analyzed (so the ranker includes them)
            up, dn = up_down(deg, LFC)
            try:
                df = sigcom.enrich_sigcom_directional(up, dn, name2id, dataset, ct,
                                                      top_n=TOP_N, clean=False)
                n = len(df)
            except Exception as exc:
                print(f"  {dataset}/{ct}: FAILED ({type(exc).__name__}: {exc})")
                n = -1
            # compare to the old directional-L2S2 yield for context
            l2dir = results_ct / "promising_drug_candidates_l2s2_updown.csv"
            l2n = len(pd.read_csv(l2dir)) if l2dir.exists() else 0
            rows.append({"dataset": dataset, "cell_type": ct, "up": len(up), "down": len(dn),
                         "sigcom_drugs": n, "l2s2dir_drugs": l2n})
            print(f"  {dataset}/{ct:<18} up={len(up):>3} dn={len(dn):>3}  "
                  f"SigCom={n:>4}  (L2S2-dir={l2n})")

    summary = pd.DataFrame(rows)
    summary.to_csv("sigcom_directional_yield.csv", index=False)
    ok = summary[summary["sigcom_drugs"] >= 0]
    print(f"\nWrote sigcom_directional_yield.csv ({len(summary)} strata)")
    print(f"SigCom directional yield: mean {ok['sigcom_drugs'].mean():.0f} drugs/stratum "
          f"vs L2S2-directional mean {ok['l2s2dir_drugs'].mean():.1f}")

    if DO_RANK:
        print("\nRebuilding Borda master with the SigCom arm folded in...")
        ConsensusRanker(RESULTS).rank()
        print("Done. MASTER_DRUG_RANKINGS.csv now includes the SigCom directional arm.")


if __name__ == "__main__":
    main()
