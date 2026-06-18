#!/usr/bin/env python3
"""
Pipeline funnel as a Sankey: how DrugEnsembleNet narrows the drug universe.
Real flow numbers from a representative stratum (RL_cdHD, astrocytes).
  All drugs -> network-proximal -> per-method enriched candidates (+ directional arm)
  -> ensemble consensus.
Renders to figures/fig_pipeline_sankey.png via plotly + kaleido.
Run:  ./.venv/bin/python fig_pipeline_sankey.py
"""
import re
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

CT = Path("data/results/RL_cdHD_E_CTRL/ast")

# --- real numbers ---------------------------------------------------------
with open(CT / "network_proximities.csv") as f:
    f.readline()
    thr = float(re.search(r"Threshold:\s*([0-9.\-]+)", f.readline()).group(1))
prox = pd.read_csv(CT / "network_proximities.csv", comment="#")
total = len(prox)
proximal = int((prox["Distance"] < thr).sum())
not_prox = total - proximal


def ids(name):
    p = CT / name
    return set(pd.read_csv(p)["DrugBank_ID"].astype(str)) if p.exists() else set()


IG, L2, DR = ids("promising_drug_candidates_igsea.csv"), \
    ids("promising_drug_candidates_l2s2.csv"), \
    ids("promising_drug_candidates_l2s2_updown.csv")
ig_only = len(IG - L2)
l2_only = len(L2 - IG)
both = len(IG & L2)
enriched = len(IG | L2)                 # all proximal+significant
not_enriched = proximal - enriched
dir_unique = len(DR - (IG | L2))        # directional arm's net-new candidates
consensus = enriched + dir_unique

# --- nodes / colors -------------------------------------------------------
C = {"all": "#264653", "prox": "#2a9d8f", "drop": "#cbd2d9",
     "ig": "#2a9d8f", "both": "#8d6cab", "l2": "#e76f51", "dir": "#5b6ee1",
     "cons": "#e9a000"}
labels = [
    f"All drugs screened<br>{total:,}",            # 0
    f"Not proximal<br>{not_prox:,}",               # 1
    f"Network-proximal<br>{proximal:,}",           # 2
    f"Not enriched<br>{not_enriched:,}",           # 3
    f"iGSEA only<br>{ig_only}",                    # 4
    f"Both methods<br>{both}",                     # 5
    f"L2S2 only<br>{l2_only}",                     # 6
    f"Directional L2S2<br>+{dir_unique} (no proximity)",  # 7
    f"Ensemble consensus<br>{consensus}",          # 8
]
node_color = [C["all"], C["drop"], C["prox"], C["drop"], C["ig"], C["both"],
              C["l2"], C["dir"], C["cons"]]
# columns: keep a clean left-to-right funnel
node_x = [0.06, 0.34, 0.34, 0.63, 0.63, 0.63, 0.63, 0.63, 0.90]
node_y = [0.50, 0.88, 0.40, 0.92, 0.30, 0.55, 0.70, 0.10, 0.45]

links = [
    (0, 1, not_prox, C["drop"]),
    (0, 2, proximal, C["prox"]),
    (2, 3, not_enriched, C["drop"]),
    (2, 4, ig_only, C["ig"]),
    (2, 5, both, C["both"]),
    (2, 6, l2_only, C["l2"]),
    (4, 8, ig_only, C["ig"]),
    (5, 8, both, C["both"]),
    (6, 8, l2_only, C["l2"]),
    (7, 8, dir_unique, C["dir"]),
]


def rgba(hex_color, a=0.40):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


fig = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(
        label=labels, color=node_color, x=node_x, y=node_y,
        pad=22, thickness=26, line=dict(color="white", width=1),
    ),
    link=dict(
        source=[s for s, *_ in links],
        target=[t for _, t, *_ in links],
        value=[v for _, _, v, _ in links],
        color=[rgba(c) for *_, c in links],
    ),
))
fig.update_layout(
    title=dict(text="<b>DrugEnsembleNet pipeline: from 7,992 drugs to a ranked shortlist</b>"
                    "<br><span style='font-size:13px;color:#666'>"
                    "representative stratum — RL_cdHD, astrocytes</span>",
               x=0.5, xanchor="center", font=dict(size=20)),
    font=dict(family="Arial", size=13, color="#1d2433"),
    paper_bgcolor="white", plot_bgcolor="white",
    margin=dict(l=130, r=180, t=90, b=30),
)
out = "figures/fig_pipeline_sankey.png"
fig.write_image(out, width=1550, height=680, scale=2)
print(f"saved {out}")
print(f"flows: all={total} | not_prox={not_prox} prox={proximal} | "
      f"not_enriched={not_enriched} ig_only={ig_only} both={both} l2_only={l2_only} | "
      f"dir_unique={dir_unique} consensus={consensus}")
