"""
SigCom LINCS directional (reverser) signature search.

A high-coverage replacement/supplement for the low-yield L2S2 paired up/down query.
Submits up- and down-regulated gene sets to SigCom LINCS, keeps the top-N strongest
REVERSER signatures (drugs whose signature opposes the disease signature), collapses
to distinct drugs, maps them to DrugBank IDs, and writes a promising-candidate list in
the same format the Borda ranker consumes.

API: metadata-api/entities/find (symbol->UUID) + data-api/.../enrich/ranktwosided.
"""
import json
import logging

import pandas as pd
import requests

log = logging.getLogger("sigcom")

METADATA_API = "https://maayanlab.cloud/sigcom-lincs/metadata-api/"
DATA_API = "https://maayanlab.cloud/sigcom-lincs/data-api/api/v1/"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

# gene-level artifacts that contaminate snRNA-seq signatures and mislead any
# reversal search (mitochondrial, ribosomal, sex-linked).
_SEX_GENES = {
    "XIST", "TSIX", "RPS4Y1", "RPS4Y2", "DDX3Y", "UTY", "USP9Y", "KDM5D",
    "EIF1AY", "ZFY", "ZFY-AS1", "TXLNGY", "NLGN4Y", "TMSB4Y", "PRKY", "TBL1Y", "AMELY",
}
_DROP_PREFIX = ("MT-", "MT.", "RPL", "RPS", "MRPL", "MRPS", "TTTY", "LINC", "MIR")


def clean_genes(genes):
    """Drop mitochondrial / ribosomal / sex-linked / lncRNA-artifact genes."""
    out = []
    for g in genes:
        gu = str(g).upper()
        if gu in _SEX_GENES or gu.startswith(_DROP_PREFIX):
            continue
        out.append(g)
    return out


def build_name2id(drugbank):
    """One-time exact (lowercased) drug-name -> DrugBank ID map (no slow fuzzy search)."""
    db = drugbank.database
    name2id = {}
    for did in db._fields:
        drug = getattr(db, did)
        nm = getattr(drug, "name", None)
        if nm:
            name2id[str(nm).lower().strip()] = did
    return name2id


def _resolve_symbols(genes):
    r = requests.post(
        METADATA_API + "entities/find", headers=HEADERS, timeout=60,
        data=json.dumps({"filter": {"where": {"meta.symbol": {"inq": list(genes)}},
                                    "fields": ["id", "meta.symbol"]}}))
    r.raise_for_status()
    return [e["id"] for e in r.json()]


def _signature_drug_names(uuids):
    out = {}
    for i in range(0, len(uuids), 200):
        r = requests.post(
            METADATA_API + "signatures/find", headers=HEADERS, timeout=120,
            data=json.dumps({"filter": {"where": {"id": {"inq": uuids[i:i + 200]}}}}))
        r.raise_for_status()
        for s in r.json():
            out[s["id"]] = s.get("meta", {}).get("pert_name")
    return out


def enrich_sigcom_directional(genes_up, genes_down, name2id, run_name, disease_name,
                              top_n=100, database="l1000_cp", clean=True):
    """
    Query SigCom LINCS for reversers and write promising_drug_candidates_sigcom.csv.
    name2id: dict {drug_name_lower -> DrugBank_ID} from build_name2id().
    Returns the candidate DataFrame (possibly empty).
    """
    if clean:
        genes_up, genes_down = clean_genes(genes_up), clean_genes(genes_down)
    up = _resolve_symbols(genes_up)
    down = _resolve_symbols(genes_down)
    if not up or not down:
        log.warning("SigCom: no genes resolved for %s (up=%d down=%d)", disease_name, len(up), len(down))
        return pd.DataFrame()

    q = {"up_entities": up, "down_entities": down, "limit": 5000, "database": database}
    r = requests.post(DATA_API + "enrich/ranktwosided", headers=HEADERS, timeout=300, data=json.dumps(q))
    r.raise_for_status()
    rev = [x for x in r.json().get("results", []) if x.get("type") == "reversers"]
    if not rev:
        return pd.DataFrame()

    df = pd.DataFrame(rev)
    # strongest reverser first = most negative combined z
    df = df.sort_values("z-sum", ascending=True, ignore_index=True)
    # only resolve drug names for the strongest signatures (enough to reach top_n drugs)
    df = df.head(max(top_n * 10, 1000))
    df["DrugBank_Name"] = df["uuid"].map(_signature_drug_names(df["uuid"].tolist()))
    df = df.dropna(subset=["DrugBank_Name"])
    df = df.drop_duplicates(subset=["DrugBank_Name"], keep="first")          # best signature per drug
    df["DrugBank_ID"] = df["DrugBank_Name"].map(lambda n: name2id.get(str(n).lower().strip()))
    df = df.dropna(subset=["DrugBank_ID"]).head(top_n)

    keep = [c for c in ["DrugBank_ID", "DrugBank_Name", "z-sum", "z-up", "z-down",
                        "fdr-up", "fdr-down"] if c in df.columns]
    df = df[keep].reset_index(drop=True)
    out = f"data/results/{run_name}/{disease_name.replace(' ', '')}/promising_drug_candidates_sigcom.csv"
    df.to_csv(out, index=False)
    log.info("SigCom: %d reverser drugs for %s -> %s", len(df), disease_name, out)
    return df
