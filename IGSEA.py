import logging
from collections import namedtuple

import gseapy
import gseapy.algorithm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logging_name = "IGSEA"
log = logging.getLogger(logging_name)


################################################################################
#
#                   Inverted Gene Set Enrichment Analysis
#
################################################################################


prerank_gsea_results = namedtuple(
    typename="IGSEA_results",
    field_names=["ES", "NES", "pvalue"],
)


# @profile()
def prerank_gsea(
    expression_df, disease_name, gene_set, sigid2DBid, nperm=int(1e4), seed=12345
):
    """
    Gens Set Enrichment Analysis for a Preranked Single Signature
    """
    log = logging.getLogger("IGSEA:prerank_gsea")
    rank = expression_df.sort_values(ascending=False)
    gsea_results, _, _, _ = gseapy.algorithm.gsea_compute(
        data=rank,
        n=nperm,
        gmt={f"{sigid2DBid[rank.name]}_{disease_name.replace(' ', '')}": gene_set},
        weighted_score_type=1,
        permutation_type="gene_set",
        method=None,
        pheno_pos="Pos",
        pheno_neg="Neg",
        classes=None,
        ascending=False,
        processes=1,
        seed=seed,
    )

    ES, nES, pvalue, _ = tuple(*gsea_results)

    return prerank_gsea_results(ES, nES, pvalue)


# @profile()


def _run_single_gsea(
    col_name, signature_data, disease_name, gene_set, sigid2DBid, nperm, seed
):
    try:
        if signature_data.isna().all():
            return col_name, None

        res = prerank_gsea(
            signature_data, disease_name, gene_set, sigid2DBid, nperm, seed
        )
        return col_name, res
    except Exception:
        return col_name, None


def IGSEA(
    run_name,
    disease_name,
    gene_set,
    lincs,
    job_id,
    nperm=int(1e4),
    alpha=0.25,
    seed=12345,
):
    """
    Inverted Gene Set Enrichment Analysis
    As in the paper https://academic.oup.com/bioinformatics/article/36/17/4626/5855131
    """
    log = logging.getLogger("IGSEA:igsea")

    gene_set = tuple({str(id) for id in gene_set} & lincs.BING_genes)

    igsea_results = {}

    for database_batch in lincs.database:
        # THE MAGIC: joblib Parallel block replacing your dict comprehension
        # Set n_jobs to the number of cores you want this specific worker to use (e.g., 10)
        batch_results = Parallel(n_jobs=10, backend="loky")(
            delayed(_run_single_gsea)(
                col_name,
                signature_data,
                disease_name,
                gene_set,
                lincs.sigid2DBid,
                nperm,
                seed,
            )
            # Use .items() instead of .iloc for vastly faster iteration
            for col_name, signature_data in tqdm(
                database_batch.items(),
                total=len(database_batch.columns),
                position=job_id,
                desc=f"Job {job_id} ({disease_name})",
                leave=False,  # Cleans up terminal when done
            )
        )

        # Unpack the parallel results into your dictionary
        for col_name, res in batch_results:
            if res is not None:
                igsea_results[col_name] = res

    igsea_results_cleaned = {
        k: v for k, v in igsea_results.items() if not np.isnan(v.pvalue)
    }

    # Significance analysis
    pvalues = np.array([result.pvalue for result in igsea_results_cleaned.values()])
    mask, pvalues_corrected, _, _ = multipletests(
        pvals=pvalues, alpha=alpha, method="fdr_bh"
    )

    results_df = pd.DataFrame(
        {
            "Signature": igsea_results_cleaned.keys(),
            "DrugBank_ID": [
                lincs.sigid2DBid.get(id) for id in igsea_results_cleaned.keys()
            ],
            "DrugBank_Name": [
                lincs.sigid2DBname.get(id) for id in igsea_results_cleaned.keys()
            ],
            "Enrichment_Score": [
                result.ES for result in igsea_results_cleaned.values()
            ],
            "Normalized_Enrichment_Score": [
                result.NES for result in igsea_results_cleaned.values()
            ],
            "p-value": [pvalue for pvalue in pvalues],
            "FDR": [
                pvalue for pvalue in pvalues_corrected
            ],  # corrected p-value for multiple tests (False Discovery Rate)
            "Cell_Line": [
                lincs.sigid2cell.get(id) for id in igsea_results_cleaned.keys()
            ],
        }
    )

    results_df.sort_values(
        by=["p-value", "Normalized_Enrichment_Score"],
        ascending=[True, False],
        inplace=True,
        key=abs,
        ignore_index=True,
    )

    results_df["p-value"] = [
        pvalue if pvalue != 0 else f"<{1 / nperm}" for pvalue in results_df["p-value"]
    ]
    results_df["FDR"] = [
        fdr if fdr != 0 else f"<{1 / nperm}" for fdr in results_df["FDR"]
    ]

    results_df.to_csv(
        f"data/results/{run_name}/{disease_name.replace(' ', '')}/IGSEA_results.tsv",
        sep="\t",
        index=False,
    )

    return results_df
