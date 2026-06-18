#!/usr/bin/env python3
"""
Ensemble vs. single-tool recovery of HD clinical-trial true positives.
Question: at each (dataset, cell_type), does the tool_consensus recover the
43 HD true positives better than the BEST single component?

Works per-stratum (same dataset + cell type), so only tool-count varies.
Run from repo root.
"""
from pathlib import Path
import pandas as pd
from scipy.stats import hypergeom

RESULTS = Path("data/results")
EXCLUDE = {"osktosky", "oskotsky"}

TRUE_POS = {"DB17924","DB00470","DB00148","DB21549","DB16975","DB04844","DB11947",
            "DB06819","DB11725","DB01017","DB01043","DB14509","DB00313","DB06685",
            "DB08387","DB09270","DB13978","DB15155","DB01156","DB00915","DB12161",
            "DB11340","DB02709","DB18165","DB17870","DB00334","DB13025","DB21645",
            "DB01065","DB05565","DB16977","DB01039","DB12116","DB11677","DB00740",
            "DB16968","DB08887","DB00734","DB00514","DB00908","DB00289","DB00215","DB11915"}

COMPONENTS = {
    "iGSEA": "promising_drug_candidates_igsea.csv",
    "L2S2": "promising_drug_candidates_l2s2.csv",
    "L2S2_dir": "promising_drug_candidates_l2s2_updown.csv",
}
CONSENSUS = "tool_consensus.csv"

def ranked_ids(path):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "DrugBank_ID" not in df.columns or df.empty:
        return None
    return list(dict.fromkeys(df["DrugBank_ID"].astype(str).tolist()))  # dedupe, keep order

def tp_in_topN(ranked, known, N):
    return len(set(ranked[:N]) & known)

def hyper_p(ranked, known, N):
    U = len(ranked); m = len(set(ranked) & known)
    if U == 0 or m == 0 or N > U: return None
    q = tp_in_topN(ranked, known, N)
    return hypergeom.sf(q-1, U, m, N)

def main():
    datasets = [d for d in RESULTS.iterdir()
                if d.is_dir() and d.name not in EXCLUDE
                and any((c/COMPONENTS["iGSEA"]).exists() for c in d.iterdir() if c.is_dir())]

    rows = []
    for ds in sorted(datasets):
        for ct in sorted([c for c in ds.iterdir() if c.is_dir()]):
            lists = {}
            for comp, fname in COMPONENTS.items():
                r = ranked_ids(ct / fname)
                if r: lists[comp] = r
            cons = ranked_ids(ct / CONSENSUS)
            if cons is None or not lists:
                continue

            # Fair cutoff: cap N at the shortest list present so all are comparable.
            shortest = min([len(r) for r in lists.values()] + [len(cons)])
            for N in [n for n in (20, 50) if n <= shortest]:
                # TP recovery for each single tool and the consensus
                tool_tp = {comp: tp_in_topN(r, TRUE_POS, N) for comp, r in lists.items()}
                cons_tp = tp_in_topN(cons, TRUE_POS, N)
                best_single = max(tool_tp.values())
                best_tool = max(tool_tp, key=tool_tp.get)

                rows.append({
                    "dataset": ds.name, "cell_type": ct.name, "N": N,
                    **{f"{c}_tp": tool_tp.get(c) for c in COMPONENTS},
                    "best_single_tp": best_single, "best_tool": best_tool,
                    "consensus_tp": cons_tp,
                    "consensus_beats_best": cons_tp > best_single,
                    "consensus_ties_best": cons_tp == best_single,
                    "consensus_p": hyper_p(cons, TRUE_POS, N),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No comparable strata found."); return

    df.to_csv("ensemble_vs_components.csv", index=False)
    print(f"Wrote ensemble_vs_components.csv ({len(df)} stratum-cutoff rows)\n")

    # ---- headline: win rate at each cutoff ----
    for N in sorted(df["N"].unique()):
        sub = df[df["N"] == N]
        beats = sub["consensus_beats_best"].sum()
        ties  = sub["consensus_ties_best"].sum()
        loses = len(sub) - beats - ties
        print(f"=== top-{N} (n={len(sub)} strata) ===")
        print(f"  consensus BEATS best single tool: {beats} ({100*beats/len(sub):.0f}%)")
        print(f"  consensus TIES best single tool:  {ties} ({100*ties/len(sub):.0f}%)")
        print(f"  consensus LOSES to best single:   {loses} ({100*loses/len(sub):.0f}%)")
        print(f"  mean TP recovered — consensus: {sub['consensus_tp'].mean():.2f}, "
              f"best single: {sub['best_single_tp'].mean():.2f}, "
              f"iGSEA: {sub['iGSEA_tp'].mean():.2f}, L2S2: {sub['L2S2_tp'].mean():.2f}"
              f"L2S2_dir: {sub['L2S2_dir_tp'].mean():.2f}")
        
        print()

if __name__ == "__main__":
    main()