#!/usr/bin/env python3
"""
coverage_vs_discrimination.py

Why iGSEA recovers ~4x fewer HD true positives than the proximity+L2S2 arm:
is it a DATA CEILING (iGSEA can't even see most TPs) or a DISCRIMINATION
failure (iGSEA sees them but ranks them poorly)? These need different fixes
(LINCS2020 upgrade vs. method/threshold change), so separate them before
concluding anything.

Two layers:
  Layer 1 (always runs, off the promising-candidate lists you already have):
    - selection coverage: how many of the 43 TPs each tool's list contains at all
    - recovery@N: TPs in top-20 / top-50 (matches the headline eval)
    - conditional recovery: of the TPs a tool DID select, how many reach top-N
      -> if conditional recovery is comparable across tools but raw recovery
         isn't, the gap is upstream (fewer TPs selected), not ranking.
  Layer 2 (runs only if raw pre-threshold results are present): splits the
    "not selected" TPs into NOT-IN-DATA vs SCORED-BUT-REJECTED. This is the
    clean coverage-vs-discrimination cut. Skips cleanly if files/columns absent.

Run from repo root. Writes coverage_vs_discrimination.csv and prints a verdict.
"""
from pathlib import Path
import pandas as pd

RESULTS = Path("data/results")
EXCLUDE = {"osktosky", "oskotsky"}      # HD-only diagnostic; non-HD strata would score against the wrong TPs
TOPNS   = [20, 50]

# promising (post-threshold) candidate lists -- guaranteed to exist
COMPONENTS = {
    "iGSEA":    "promising_drug_candidates_igsea.csv",
    "L2S2":     "promising_drug_candidates_l2s2.csv",        # proximity + L2S2 (your novel arm)
    "L2S2_dir": "promising_drug_candidates_l2s2_updown.csv", # directional L2S2
}

# OPTIONAL raw, pre-threshold results = the universe each tool actually SCORED.
# Edit filenames to match your repo; leave as-is to attempt auto-find, or set to
# None to skip Layer 2 for a tool. Must contain a DrugBank ID column.
RAW_RESULTS = {
    "iGSEA":    "IGSEA_results.tsv",
    "L2S2":     None,    # set to the full L2S2 results filename if you have it
    "L2S2_dir": None,
}

TRUE_POS = {"DB17924","DB00470","DB00148","DB21549","DB16975","DB04844","DB11947",
            "DB06819","DB11725","DB01017","DB01043","DB14509","DB00313","DB06685",
            "DB08387","DB09270","DB13978","DB15155","DB01156","DB00915","DB12161",
            "DB11340","DB02709","DB18165","DB17870","DB00334","DB13025","DB21645",
            "DB01065","DB05565","DB16977","DB01039","DB12116","DB11677","DB00740",
            "DB16968","DB08887","DB00734","DB00514","DB00908","DB00289","DB00215","DB11915"}
N_TP = len(TRUE_POS)


def ranked_ids(path):
    """Ordered, de-duplicated DrugBank IDs from a promising-candidate list."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "DrugBank_ID" not in df.columns or df.empty:
        return None
    return list(dict.fromkeys(df["DrugBank_ID"].astype(str).tolist()))


def find_id_col(df):
    """Locate a DrugBank-ID column by name (case-insensitive)."""
    for c in df.columns:
        if "drugbank" in c.lower().replace(" ", "").replace("_", ""):
            return c
    return None


def scored_universe(ct_dir, ds_dir, fname):
    """
    Set of DrugBank IDs a tool actually scored (pre-threshold), for Layer 2.
    Looks in the cell-type dir first, then the dataset dir. Returns None if the
    file or a usable ID column is missing (Layer 2 then skips for this tool).
    """
    if not fname:
        return None
    for base in (ct_dir, ds_dir):
        p = base / fname
        if p.exists():
            try:
                sep = "\t" if p.suffix in (".tsv", ".txt") else ","
                df = pd.read_csv(p, sep=sep)
            except Exception:
                continue
            col = find_id_col(df)
            if col is None or df.empty:
                continue
            return set(df[col].astype(str))
    return None


def main():
    datasets = [d for d in RESULTS.iterdir()
                if d.is_dir() and d.name not in EXCLUDE
                and any((c / COMPONENTS["iGSEA"]).exists() for c in d.iterdir() if c.is_dir())]

    rows = []
    layer2_used = {comp: False for comp in COMPONENTS}

    for ds in sorted(datasets):
        for ct in sorted([c for c in ds.iterdir() if c.is_dir()]):
            present = {}
            for comp, fn in COMPONENTS.items():
                r = ranked_ids(ct / fn)
                if r:
                    present[comp] = r
            if not present:
                continue

            for comp, ranked in present.items():
                L = len(ranked)
                selected = [d for d in ranked if d in TRUE_POS]          # TPs in the list, in order
                n_sel = len(selected)
                # rank percentile of each selected TP (1 = top, ->0 worse); lower mean = better ranking
                pctiles = [(ranked.index(d) + 1) / L for d in selected]
                mean_pct = sum(pctiles) / n_sel if n_sel else None

                row = {"dataset": ds.name, "cell_type": ct.name, "tool": comp,
                       "list_len": L, "sel_TP": n_sel}
                for N in TOPNS:
                    top = set(ranked[:N]) & TRUE_POS
                    row[f"top{N}_TP"] = len(top)
                    # conditional: of the TPs this tool selected, how many reached top-N
                    row[f"cond_top{N}"] = (len(top) / n_sel) if n_sel else None
                row["mean_rank_pctile_selTP"] = mean_pct

                # ---- Layer 2: split "not selected" into no-data vs rejected ----
                uni = scored_universe(ct, ds, RAW_RESULTS.get(comp))
                if uni is not None:
                    layer2_used[comp] = True
                    scored_tp = TRUE_POS & uni
                    row["scored_TP"]   = len(scored_tp)               # had data + got a score
                    row["notdata_TP"]  = N_TP - len(scored_tp)        # never scored = coverage ceiling
                    row["rejected_TP"] = len(scored_tp) - n_sel       # scored but filtered out = discrimination/threshold
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No comparable strata found."); return
    df.to_csv("coverage_vs_discrimination.csv", index=False)
    print(f"Wrote coverage_vs_discrimination.csv ({len(df)} tool-stratum rows), {N_TP} HD true positives\n")

    # ---- per-dataset selection coverage (a non-HD dataset would show ~0 here) ----
    print("Selection coverage by dataset (mean TPs in list, iGSEA):")
    ig = df[df["tool"] == "iGSEA"]
    for ds_name, g in ig.groupby("dataset"):
        print(f"  {ds_name:<24} sel_TP mean={g['sel_TP'].mean():.2f}  (strata={len(g)})")
    print()

    # ---- per-tool summary across strata ----
    print("Per-tool summary (mean across strata):")
    hdr = f"  {'tool':<10} {'sel_TP':>7} {'top20':>6} {'top50':>6} {'cond20':>7} {'cond50':>7} {'rankPct':>8}"
    extra = f" {'scored':>7} {'rejected':>9} {'notdata':>8}"
    print(hdr + (extra if any(layer2_used.values()) else ""))
    for comp in COMPONENTS:
        g = df[df["tool"] == comp]
        if g.empty:
            continue
        line = (f"  {comp:<10} {g['sel_TP'].mean():>7.2f} "
                f"{g['top20_TP'].mean():>6.2f} {g['top50_TP'].mean():>6.2f} "
                f"{g['cond_top20'].mean():>7.2f} {g['cond_top50'].mean():>7.2f} "
                f"{g['mean_rank_pctile_selTP'].mean():>8.2f}")
        if layer2_used[comp]:
            line += (f" {g['scored_TP'].mean():>7.2f} "
                     f"{g['rejected_TP'].mean():>9.2f} {g['notdata_TP'].mean():>8.2f}")
        print(line)
    print()

    # ---- plain-language verdict: iGSEA vs the stronger L2S2 arm ----
    best_l2 = max(["L2S2", "L2S2_dir"],
                  key=lambda c: df[df["tool"] == c]["top20_TP"].mean() if not df[df["tool"] == c].empty else -1)
    ig_, l2_ = df[df["tool"] == "iGSEA"], df[df["tool"] == best_l2]
    print(f"VERDICT (iGSEA vs {best_l2}):")
    print(f"  selection coverage: iGSEA {ig_['sel_TP'].mean():.2f} vs {l2_['sel_TP'].mean():.2f} TPs/stratum")
    print(f"  rank quality of selected TPs (lower=better): "
          f"iGSEA {ig_['mean_rank_pctile_selTP'].mean():.2f} vs {l2_['mean_rank_pctile_selTP'].mean():.2f}")
    print("  How to read it:")
    print("   - iGSEA selects far fewer TPs but ranks the ones it selects comparably")
    print("     => COVERAGE-limited. iGSEA is not 'bad'; the LINCS2020 upgrade is the fix.")
    print("   - iGSEA selects a similar number but ranks them worse (higher rankPct)")
    print("     => DISCRIMINATION gap. Data upgrade won't fully fix it.")
    if any(layer2_used.values()):
        print("   - Layer 2: high notdata_TP confirms the data ceiling; high rejected_TP")
        print("     points instead to the FDR<0.25 threshold dropping scored TPs.")
    else:
        print("   - Layer 2 skipped (no raw pre-threshold files found): selection coverage")
        print("     still conflates 'no data' with 'scored but rejected'. The sig_info")
        print("     check (next step) is the clean test of the data ceiling.")


if __name__ == "__main__":
    main()