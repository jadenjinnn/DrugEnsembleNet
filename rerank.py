#!/usr/bin/env python3
"""
Apply the iGSEA |NES| fix to EXISTING results WITHOUT re-analyzing.

The expensive stages (network proximity, iGSEA permutations, L2S2) are already
cached on disk. This only re-orders the iGSEA promising-candidate lists by
enrichment effect size |NES| (read from the cached IGSEA_results.csv) and then
re-runs the Borda consensus, regenerating tool_consensus / consensus_ALL_DATASETS
/ MASTER_DRUG_RANKINGS. Seconds, no DB/network/GSEA.

Usage:  ./.venv/bin/python rerank.py [results_dir]   (default: data/results)
"""
import glob
import sys
from pathlib import Path

import pandas as pd

from run_parallel import ConsensusRanker

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/results")


def reorder_igsea_by_nes(results_dir):
    n = 0
    for prom in glob.glob(str(results_dir / "*" / "*" / "promising_drug_candidates_igsea.csv")):
        cd = Path(prom).parent
        raw_path = cd / "IGSEA_results.csv"
        if not raw_path.exists():
            continue
        d = pd.read_csv(prom)
        if "DrugBank_ID" not in d.columns or d.empty:
            continue
        d["DrugBank_ID"] = d["DrugBank_ID"].astype(str)
        raw = pd.read_csv(raw_path)
        raw["DrugBank_ID"] = raw["DrugBank_ID"].astype(str)
        raw["aNES"] = pd.to_numeric(raw["Normalized_Enrichment_Score"], errors="coerce").abs()
        best = raw.groupby("DrugBank_ID")["aNES"].max()
        d["Normalized_Enrichment_Score"] = d["DrugBank_ID"].map(best)
        d = (d.assign(_a=d["Normalized_Enrichment_Score"].abs().fillna(0))
               .sort_values("_a", ascending=False, ignore_index=True)
               .drop(columns="_a"))
        d.to_csv(prom, index=False)
        n += 1
    return n


if __name__ == "__main__":
    print(f"Re-ordering iGSEA promising lists by |NES| in {RESULTS} ...")
    n = reorder_igsea_by_nes(RESULTS)
    print(f"  re-ordered {n} iGSEA candidate lists from cache (no re-analysis)")
    print("Rebuilding Borda consensus (tool_consensus -> per-cell-type -> master)...")
    ConsensusRanker(RESULTS).rank()
    print("Done. MASTER_DRUG_RANKINGS.csv regenerated with the |NES| ranking.")
