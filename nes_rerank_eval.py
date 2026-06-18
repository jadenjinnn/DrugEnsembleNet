#!/usr/bin/env python3
"""
Evaluate the iGSEA |NES| re-ranking fix end-to-end against the current pipeline.

Rebuilds the full 3-stage Borda master ranking under three component orderings:
  OLD   = current pipeline order (iGSEA by FDR, L2S2 by FDR)
  NEW   = iGSEA candidates re-ranked by |NES| (effect size); L2S2/dir unchanged
  NEW+  = NEW, plus L2S2 by oddsRatio and directional-L2S2 by oddsRatioReverse

Scores each master ranking against the HD true positives with:
  - top-N recovery + fold-enrichment + hypergeometric (Fisher one-sided) p
  - AUROC (= Mann-Whitney U; uses the WHOLE ranking, no arbitrary cutoff) + p
  - AUPRC (average precision; robust to the heavy positive/negative imbalance)
Reports for the full 43 TPs and for an amended "predictable" TP set
(TPs with an interactome target AND scored by >=1 enrichment tool).

Run from repo root:  ./.venv/bin/python nes_rerank_eval.py
"""
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score

RESULTS = Path("data/results")
EXCLUDE = {"osktosky", "oskotsky"}
TP43 = {"DB17924","DB00470","DB00148","DB21549","DB16975","DB04844","DB11947","DB06819",
        "DB11725","DB01017","DB01043","DB14509","DB00313","DB06685","DB08387","DB09270",
        "DB13978","DB15155","DB01156","DB00915","DB12161","DB11340","DB02709","DB18165",
        "DB17870","DB00334","DB13025","DB21645","DB01065","DB05565","DB16977","DB01039",
        "DB12116","DB11677","DB00740","DB16968","DB08887","DB00734","DB00514","DB00908",
        "DB00289","DB00215","DB11915"}


def pf(v):
    s = str(v); return float(s[1:]) if s.startswith("<") else float(s)


def borda(lists):
    """Per-list-normalized Borda (matches ConsensusRanker._borda_rank). -> {id: score}."""
    sc = defaultdict(float)
    for lst in lists:
        L = len(lst)
        for i, x in enumerate(lst):
            sc[x] += (L - i) / L
    return sc


def ranked(scores):
    return [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def col_order(path, csv, idcol, sortcol, ascending, new, agg_from=None, agg_col=None):
    """Return ordered DrugBank_IDs for one component; OLD=file order, NEW=by effect size."""
    p = path / csv
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if "DrugBank_ID" not in d.columns or d.empty:
        return None
    d["DrugBank_ID"] = d["DrugBank_ID"].astype(str)
    if not new:
        return d["DrugBank_ID"].tolist()
    if agg_from:                                    # pull effect size from raw results
        rp = path / agg_from
        if rp.exists():
            r = pd.read_csv(rp); r["DrugBank_ID"] = r["DrugBank_ID"].astype(str)
            r[agg_col] = pd.to_numeric(r[agg_col], errors="coerce").abs()
            a = r.groupby("DrugBank_ID")[agg_col].max()
            return d.assign(k=d["DrugBank_ID"].map(a).fillna(0)).sort_values(
                "k", ascending=False)["DrugBank_ID"].tolist()
    if sortcol in d.columns:                        # effect size already in the file
        d[sortcol] = pd.to_numeric(d[sortcol], errors="coerce").abs().fillna(0)
        return d.sort_values(sortcol, ascending=ascending)["DrugBank_ID"].tolist()
    return d["DrugBank_ID"].tolist()


def components(ctdir, mode):
    """mode in {old, new, newplus}. Returns the list of component ranked-lists present."""
    new_ig = mode in ("new", "newplus")
    new_l2 = mode == "newplus"
    ig = col_order(ctdir, "promising_drug_candidates_igsea.csv", "DrugBank_ID", None, False,
                   new_ig, agg_from="IGSEA_results.csv", agg_col="Normalized_Enrichment_Score")
    l2 = col_order(ctdir, "promising_drug_candidates_l2s2.csv", "DrugBank_ID", None, False,
                   new_l2, agg_from="l2s2_results.csv", agg_col="oddsRatio")
    dr = col_order(ctdir, "promising_drug_candidates_l2s2_updown.csv", "DrugBank_ID",
                   "oddsRatioReverse", False, new_l2)
    return [x for x in (ig, l2, dr) if x]


def build_master(mode):
    """Replicate the 3-stage Borda: within-stratum -> per-cell-type -> master."""
    cell_groups = defaultdict(list)
    for ds in RESULTS.iterdir():
        if not ds.is_dir() or ds.name in EXCLUDE:
            continue
        cts = [c for c in ds.iterdir() if c.is_dir()]
        if not any((c / "promising_drug_candidates_igsea.csv").exists() for c in cts):
            continue
        for ct in cts:
            comps = components(ct, mode)
            if comps:
                cell_groups[ct.name].append(ranked(borda(comps)))
    percell = [ranked(borda(lists)) for lists in cell_groups.values() if lists]
    return borda(percell)            # {id: master Borda score}


def evaluate(master, tp, label):
    ids = list(master.keys())
    y = np.array([1 if i in tp else 0 for i in ids])
    s = np.array([master[i] for i in ids])
    M, n = len(ids), int(y.sum())
    order = np.argsort(-s)
    yo = y[order]
    out = {"label": label, "M": M, "pos": n}
    for N in (20, 50, 100):
        k = int(yo[:N].sum()); exp = N * n / M
        out[f"top{N}"] = (k, k / exp if exp else 0, hypergeom.sf(k - 1, M, n, N))
    out["auroc"] = roc_auc_score(y, s) if 0 < n < M else float("nan")
    out["auprc"] = average_precision_score(y, s) if n else float("nan")
    pos, neg = s[y == 1], s[y == 0]
    out["mwu_p"] = mannwhitneyu(pos, neg, alternative="greater").pvalue if n and (M - n) else float("nan")
    return out


# predictable TP sets ------------------------------------------------------
def union_ids(pattern, idcol="DrugBank_ID"):
    out = set()
    for f in glob.glob(pattern):
        if any(e in f for e in EXCLUDE):
            continue
        try:
            d = pd.read_csv(f, comment="#")
        except Exception:
            continue
        if idcol in d.columns:
            out |= set(d[idcol].astype(str))
    return out


has_target = union_ids("data/results/*/*/network_proximities.csv") & TP43
scored = (union_ids("data/results/*/*/IGSEA_results.csv")
          | union_ids("data/results/*/*/l2s2_results.csv")) & TP43
predictable = has_target & scored        # has a target AND was evaluated by a tool

print(f"TP sets:  full=43 | has interactome target={len(has_target)} | "
      f"scored by a tool={len(scored)} | predictable (both)={len(predictable)}\n")

masters = {m: build_master(m) for m in ("old", "new", "newplus")}
print(f"master universe size (same for all): {len(masters['old'])} drugs\n")

for tpset, tpname in [(TP43, "FULL 43 TPs"), (predictable, f"PREDICTABLE {len(predictable)} TPs")]:
    print(f"================  scored against {tpname}  ================")
    hdr = f"{'master':<8}{'pos':>4}{'top20':>14}{'top50':>14}{'top100':>15}{'AUROC':>8}{'AUPRC':>8}{'MWU p':>10}"
    print(hdr)
    for m in ("old", "new", "newplus"):
        e = evaluate(masters[m], tpset, m)
        def fmt(t):
            k, fold, p = t; return f"{k}|{fold:.1f}x|p={p:.1g}"
        print(f"{m:<8}{e['pos']:>4}  {fmt(e['top20']):>12}  {fmt(e['top50']):>12}  "
              f"{fmt(e['top100']):>13}{e['auroc']:>8.3f}{e['auprc']:>8.3f}{e['mwu_p']:>10.1g}")
    print()
