#!/usr/bin/env python3
"""
Complementarity test: can the ensemble EVER beat L2S2 alone?

An ensemble of rankers can only outperform its best member if the weaker member
recovers true positives the strong member misses. Here we ask, for the 43 HD
clinical-trial true positives (TPs): does RePurposeNet-iGSEA recover any TP that
RePurposeNet-L2S2 does not?

Two levels per (dataset, cell_type) stratum:
  SELECTION level - the final promising-candidate lists (post proximity + FDR).
                    This is what actually feeds Borda.
  SCORED level    - the raw enrichment results (every drug each arm evaluated).
                    Tells us whether a missed TP was never seen vs. seen-but-filtered.

Decisive numbers:
  - per stratum: TPs selected by iGSEA-only (i.e. L2S2 missed it there)
  - pooled:      TPs in iGSEA's union NOT in L2S2's union  -> global complementarity
If both are ~0, no aggregation of these two arms can beat L2S2, and the ensemble
framing does not hold for this pair.

Run from repo root.
"""
from pathlib import Path
import pandas as pd

RESULTS = Path("data/results")
EXCLUDE = {"osktosky", "oskotsky"}  # non-HD; would score against the wrong TPs

TRUE_POS = {"DB17924","DB00470","DB00148","DB21549","DB16975","DB04844","DB11947",
            "DB06819","DB11725","DB01017","DB01043","DB14509","DB00313","DB06685",
            "DB08387","DB09270","DB13978","DB15155","DB01156","DB00915","DB12161",
            "DB11340","DB02709","DB18165","DB17870","DB00334","DB13025","DB21645",
            "DB01065","DB05565","DB16977","DB01039","DB12116","DB11677","DB00740",
            "DB16968","DB08887","DB00734","DB00514","DB00908","DB00289","DB00215","DB11915"}

SEL = {  # selection-level (post-threshold) lists that feed Borda
    "iGSEA": "promising_drug_candidates_igsea.csv",
    "L2S2":  "promising_drug_candidates_l2s2.csv",
}
RAW = {  # scored-level (pre-threshold) universes
    "iGSEA": "IGSEA_results.csv",
    "L2S2":  "l2s2_results.csv",
}


def ids(path):
    """Set of DrugBank IDs in a result file (str), or None if absent/unreadable."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "DrugBank_ID" not in df.columns or df.empty:
        return set()
    return set(df["DrugBank_ID"].astype(str))


def main():
    datasets = [d for d in RESULTS.iterdir()
                if d.is_dir() and d.name not in EXCLUDE
                and any((c / SEL["iGSEA"]).exists() for c in d.iterdir() if c.is_dir())]

    rows = []
    # pooled unions across all strata
    pool = {lvl: {arm: set() for arm in SEL} for lvl in ("sel", "raw")}

    for ds in sorted(datasets):
        for ct in sorted([c for c in ds.iterdir() if c.is_dir()]):
            sel = {arm: ids(ct / fn) for arm, fn in SEL.items()}
            raw = {arm: ids(ct / fn) for arm, fn in RAW.items()}
            if sel["iGSEA"] is None and sel["L2S2"] is None:
                continue

            ig_sel = (sel["iGSEA"] or set()) & TRUE_POS
            l2_sel = (sel["L2S2"] or set()) & TRUE_POS
            ig_raw = (raw["iGSEA"] or set()) & TRUE_POS
            l2_raw = (raw["L2S2"] or set()) & TRUE_POS

            for arm in SEL:
                pool["sel"][arm] |= (sel[arm] or set()) & TRUE_POS
                pool["raw"][arm] |= (raw[arm] or set()) & TRUE_POS

            igsea_only_sel = ig_sel - l2_sel   # <-- TPs iGSEA adds that L2S2 missed here
            rows.append({
                "dataset": ds.name, "cell_type": ct.name,
                "iGSEA_sel": len(ig_sel), "L2S2_sel": len(l2_sel),
                "both_sel": len(ig_sel & l2_sel),
                "iGSEA_only_sel": len(igsea_only_sel),
                "L2S2_only_sel": len(l2_sel - ig_sel),
                "iGSEA_only_ids": ",".join(sorted(igsea_only_sel)),
                "iGSEA_scored": len(ig_raw), "L2S2_scored": len(l2_raw),
                "iGSEA_only_scored": len(ig_raw - l2_raw),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No HD strata found."); return
    df.to_csv("complementarity.csv", index=False)
    n = len(df)
    print(f"Wrote complementarity.csv ({n} HD strata)\n")

    print("=== SELECTION level (what feeds Borda) ===")
    strata_ig_adds = (df["iGSEA_only_sel"] > 0).sum()
    print(f"  strata where iGSEA selects a TP that L2S2 missed there: {strata_ig_adds}/{n}")
    print(f"  mean TPs selected -- iGSEA: {df['iGSEA_sel'].mean():.2f}  "
          f"L2S2: {df['L2S2_sel'].mean():.2f}  both: {df['both_sel'].mean():.2f}")
    print(f"  mean iGSEA-only TPs/stratum: {df['iGSEA_only_sel'].mean():.2f}   "
          f"L2S2-only: {df['L2S2_only_sel'].mean():.2f}")

    ig_pool, l2_pool = pool["sel"]["iGSEA"], pool["sel"]["L2S2"]
    print(f"\n  POOLED across all strata (unique TPs ever recovered):")
    print(f"    iGSEA union: {len(ig_pool)}   L2S2 union: {len(l2_pool)}")
    only_ig = ig_pool - l2_pool
    print(f"    TPs ONLY iGSEA ever recovered (global complementarity): {len(only_ig)}"
          + (f"  -> {sorted(only_ig)}" if only_ig else ""))
    print(f"    TPs ONLY L2S2 ever recovered: {len(l2_pool - ig_pool)}")
    print(f"    L2S2 union as % of combined union: "
          f"{100*len(l2_pool)/len(ig_pool|l2_pool):.0f}%" if (ig_pool|l2_pool) else "n/a")

    print("\n=== SCORED level (did iGSEA even evaluate TPs L2S2 missed?) ===")
    ig_raw_pool, l2_raw_pool = pool["raw"]["iGSEA"], pool["raw"]["L2S2"]
    print(f"  iGSEA scored union: {len(ig_raw_pool)}   L2S2 scored union: {len(l2_raw_pool)}")
    print(f"  TPs iGSEA scored that L2S2 never scored: {len(ig_raw_pool - l2_raw_pool)}"
          + (f"  -> {sorted(ig_raw_pool - l2_raw_pool)}" if (ig_raw_pool - l2_raw_pool) else ""))

    print("\n=== VERDICT ===")
    if len(only_ig) == 0 and strata_ig_adds == 0:
        print("  NO complementarity: iGSEA never recovers a TP L2S2 misses.")
        print("  => No aggregation of these two arms can beat L2S2 alone. The ensemble")
        print("     of iGSEA+L2S2 cannot improve TP recovery; reframe or reweight.")
    elif len(only_ig) <= 2:
        print(f"  MARGINAL complementarity: iGSEA uniquely adds only {len(only_ig)} TP(s) overall.")
        print("  => Unlikely to overcome the dilution cost of unweighted Borda.")
    else:
        print(f"  REAL complementarity: iGSEA uniquely recovers {len(only_ig)} TPs L2S2 misses.")
        print("  => An ensemble CAN help, but needs a combiner that preserves them")
        print("     (performance-weighting or union), not unweighted Borda.")


if __name__ == "__main__":
    main()
