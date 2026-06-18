"""
usage:
"""

import argparse
import concurrent.futures
from html import escape
import logging
import os
import re
from pathlib import Path
from collections import defaultdict


import numpy as np
import pandas as pd
from IGSEA import IGSEA
import l2s2
import sigcom
from networkx import from_pandas_edgelist
from tqdm import tqdm

import databases
from network_proximity_improved import get_proximities
from interactome import PPI

WORKER_DATABASES = {}

HD_CLINICAL_TRIAL_DRUGBANK_IDS = {
    "DB17924",
    "DB00470",
    "DB00148",
    "DB21549",
    "DB16975",
    "DB04844",
    "DB11947",
    "DB06819",
    "DB11725",
    "DB01017",
    "DB01043",
    "DB14509",
    "DB00313",
    "DB06685",
    "DB08387",
    "DB09270",
    "DB13978",
    "DB15155",
    "DB01156",
    "DB00915",
    "DB12161",
    "DB11340",
    "DB02709",
    "DB18165",
    "DB17870",
    "DB00334",
    "DB13025",
    "DB21645",
    "DB01065",
    "DB05565",
    "DB16977",
    "DB01039",
    "DB12116",
    "DB11677",
    "DB00740",
    "DB16968",
    "DB08887",
    "DB00734",
    "DB00514",
    "DB00908",
    "DB00289",
    "DB00215",
    "DB11915",
}


class MultitoolAnalysis:
    """
    Ensemble Method for Drug Prioritization
    """

    def __init__(
        self,
        run_name,
        directional,
        cell_type_name,
        cell_lines,
        dataset_path: Path,
        lincs_db,
        drugbank_db,
        ncbi_db,
        interactome,
        job_id,
    ):
        self.run_name = run_name
        self.cell_type_name = cell_type_name
        self.cell_lines = cell_lines

        if directional == "y":
            self.directional = True
        else:
            self.directional = False

        self.lincs = lincs_db
        self.drugbank = drugbank_db
        self.ncbi = ncbi_db

        self.interactome = interactome
        self.network_proximity_results = None
        self.distance_threshold = None

        self.igsea_results = None
        self.l2s2_results = None

        self.valid_drugbank_ids = None
        self.igsea_promising_drug_candidates = None
        self.l2s2_promising_drug_candidates = None

        self.job_id = job_id

        self.log = logging.getLogger(f"{self.run_name}.{self.cell_type_name}")

        self.disease_genes = self._get_genes_from_dataset_file(dataset_path)

        if self.directional:
            self.disease_genes_updown = self._get_genes_from_directory_updown(dataset_path)

    def _get_promising_drug_candidates(self, results_df, drugbank, rank_by_nes=False):
        """
        Applies PPI network generated to find likely drug candidates.

        rank_by_nes: rank candidates by enrichment effect size |NES| instead of FDR.
        Used for iGSEA, whose permutation FDR saturates (~70% of candidates tie at the
        floor), so FDR cannot order them; |NES| is continuous and recovers the ranking.
        FDR < 0.25 is still applied as the inclusion filter regardless.
        """

        if results_df is None or results_df.empty:
            self.log.error("No significant results found for %s", self.cell_type_name)
            return None

        proximal_results = results_df[
            results_df["DrugBank_ID"].isin(self.valid_drugbank_ids)
        ].copy()

        significant_proximal_results = proximal_results[
            proximal_results["FDR"].apply(
                lambda fdr: (
                    True if isinstance(fdr, str) and "<" in fdr else float(fdr) < 0.25
                )
            )
        ]

        if self.network_proximity_results is None:
            self.log.error("Netwrok proximity empty for %s", self.cell_type_name)
            return

        significant_proximal_results = significant_proximal_results.merge(
            self.network_proximity_results[
                ["DrugBank_ID", "Distance", "Proximity"]
            ],
            how="left",
            on="DrugBank_ID",
        )

        promising_drug_candidates = significant_proximal_results.drop_duplicates(
            subset=["DrugBank_ID"], ignore_index=True
        ).copy()

        promising_drug_candidates["Targets"] = promising_drug_candidates[
            "DrugBank_ID"
        ].apply(
            lambda id: ", ".join(
                {
                    target.symbol
                    for target in drugbank.get(id).targets
                    if target.type == "protein"
                    and target.organism == "Humans"
                    and target.symbol is not None
                }
            )
        )

        promising_drug_candidates["Indication"] = promising_drug_candidates[
            "DrugBank_ID"
        ].apply(lambda id: drugbank.get(id).indication)

        if promising_drug_candidates.empty:
            self.log.warning("No promising drug candidates found for %s", self.cell_type_name)
            return None

        ordered_cols = [
            "DrugBank_ID",
            "DrugBank_Name",
            "Indication",
            "Targets",
            "Proximity",
            "FDR",
        ]

        if rank_by_nes and "Normalized_Enrichment_Score" in promising_drug_candidates.columns:
            # FDR is saturated at the permutation floor and cannot order candidates;
            # sort by enrichment effect size (|NES|) instead (FDR<0.25 already filtered).
            ordered_cols.insert(5, "Normalized_Enrichment_Score")
            promising_drug_candidates = (
                promising_drug_candidates[ordered_cols]
                .assign(_abs_nes=lambda df: df["Normalized_Enrichment_Score"].abs())
                .sort_values(by="_abs_nes", ascending=False, ignore_index=True)
                .drop(columns="_abs_nes")
            )
        else:
            promising_drug_candidates = promising_drug_candidates[ordered_cols].sort_values(
                by=["FDR", "Proximity"],
                ascending=True,
                key=np.vectorize(
                    lambda value: (
                        0.0 if isinstance(value, str) and "<" in value else float(value)
                    )
                ),
                ignore_index=True,
            )

        return promising_drug_candidates

    def _get_genes_from_dataset_file(self, file_path):
        if (
            not file_path.exists()
            or not file_path.is_file()
            or file_path.suffix != ".csv"
        ):
            self.log.error(
                "%s either does not exist, is not a file, or is not a csv file",
                file_path,
            )

            return None

        genes = pd.read_csv(file_path).iloc[:, 0].tolist()

        genes_dict = {
            self.ncbi.check_symbol(gene): self.ncbi.get_id_by_symbol(gene)
            for gene in genes
            if self.ncbi.get_id_by_symbol(gene) and self.ncbi.check_symbol(gene)
        }

        return genes_dict

    def _get_genes_from_directory_updown(self, file_path):
        if (
            not file_path.exists()
            or not file_path.is_file()
            or file_path.suffix != ".csv"
        ):
            self.log.error(
                "%s either does not exist, is not a file, or is not a csv file",
                file_path,
            )

            return None

        up_genes = set()
        down_genes = set()

        genes = pd.read_csv(file_path).iloc[:, 0].tolist()
        # log2FC = pd.read_csv(file_path).iloc[:, 2].tolist()
        updown = pd.read_csv(file_path).iloc[:, -1].tolist()

        for gene, fc in zip(genes, log2FC):
            if fc > 1.1:
                up_genes.add(gene)
            elif fc < -1.1:
                down_genes.add(gene)


        # for gene, ud in zip(genes, updown):
        #     if ud == "up":
        #         up_genes.add(gene)
        #     else:
        #         down_genes.add(gene)

        up_genes_dict = {
            self.ncbi.check_symbol(gene): self.ncbi.get_id_by_symbol(gene)
            for gene in up_genes
            if self.ncbi.get_id_by_symbol(gene) and self.ncbi.check_symbol(gene)
        }

        down_genes_dict = {
            self.ncbi.check_symbol(gene): self.ncbi.get_id_by_symbol(gene)
            for gene in down_genes
            if self.ncbi.get_id_by_symbol(gene) and self.ncbi.check_symbol(gene)
        }

        return up_genes_dict, down_genes_dict

    def analyze(self):
        """
        Drug Candidates Identification by Integrated Network Analysis Optimized for Multiple Datasets
        """
        if self.disease_genes is None:
            return None

        os.makedirs(
            f"data/results/{self.run_name}/{self.cell_type_name.replace(' ', '')}",
            exist_ok=True,
        )

        self.network_proximity_results = get_proximities(
            self.run_name, self.cell_type_name, self.disease_genes.keys(
            ), self.drugbank, self.interactome
        )

        # self.network_proximity_results = pd.read_csv(
        #     f"data/results/{self.run_name}/{self.cell_type_name.replace(' ', '')}/network_proximities.csv",
        #     comment="#",
        #     # index_col=0,
        # )

        self.log.info("Successfully Computed Drug Proximities")

        with open(
            f"data/results/{self.run_name}/{self.cell_type_name.replace(' ', '')}/network_proximities.csv",
            "r",
            encoding="utf-8",
        ) as infile:
            infile.readline()
            self.distance_threshold = float(
                re.findall(r"# Threshold: (-*[0-9]\.[0-9]{1,2})", infile.readline())[0]
            )

        self.log.info("Performing Inverted Gene Set Enrichment Analysis")

        self.igsea_results = IGSEA(
            self.run_name,
            self.cell_type_name,
            {str(id) for id in self.disease_genes.values()},
            self.lincs,
            self.job_id,
        )

        # self.igsea_results = pd.read_csv(
        #         f"data/results/{self.run_name}/{self.cell_type_name.replace(' ', '')}/IGSEA_results.tsv", sep="\t", comment="#")

        self.log.info("Successfully Performed Inverted Gene Set Enrichment Analysis")
        self.log.info("Performing Analysis with Single Set L2S2")

        self.l2s2_results = l2s2.enrich_l2s2_single_set(
            list(self.disease_genes.keys()),
            self.drugbank,
            self.cell_type_name,
            self.run_name,
        )

        self.log.info("Successfully Performed L2S2 Single Set Analysis")

        self.valid_drugbank_ids = set(
            self.network_proximity_results[
                self.network_proximity_results["Distance"] < self.distance_threshold
            ]["DrugBank_ID"]
        )

        self.log.info("Selecting Promising Drug Candidates using IGSEA Results")
        self.igsea_promising_drug_candidates = self._get_promising_drug_candidates(
            self.igsea_results, self.drugbank, rank_by_nes=True
        )

        if self.igsea_promising_drug_candidates is None:
            self.log.error(
                "No promising drug candidates found with IGSEA for %s",
                self.cell_type_name,
            )
        else:
            self.igsea_promising_drug_candidates.to_csv(
                f"data/results/{self.run_name}/{self.cell_type_name.replace(' ', '')}/promising_drug_candidates_igsea.csv"
            )

        self.log.info("Selecting Promising Drug Candidates using L2S2 Results")
        self.l2s2_promising_drug_candidates = self._get_promising_drug_candidates(
            self.l2s2_results, self.drugbank
        )

        if self.l2s2_promising_drug_candidates is None:
            self.log.error(
                "No promising drug candidates found with L2S2 for %s",
                self.cell_type_name,
            )
        else:
            self.l2s2_promising_drug_candidates.to_csv(
                f"data/results/{self.run_name}/{self.cell_type_name.replace(' ', '')}/promising_drug_candidates_l2s2.csv"
            )

        self.log.info("Selecting Promising Drug Candidates using L2S2 w/ Directionality")

        if self.directional:
            if self.disease_genes_updown[0] and self.disease_genes_updown[1]:
                up_symbols = list(self.disease_genes_updown[0].keys())
                down_symbols = list(self.disease_genes_updown[1].keys())
                l2s2.enrich_l2s2_up_down(up_symbols, down_symbols, self.drugbank, self.run_name, self.cell_type_name,)

                # High-coverage directional (reverser) arm via SigCom LINCS.
                self.log.info("Selecting Promising Drug Candidates using SigCom LINCS (directional)")
                try:
                    name2id = sigcom.build_name2id(self.drugbank)
                    sigcom.enrich_sigcom_directional(
                        up_symbols, down_symbols, name2id, self.run_name, self.cell_type_name,
                    )
                except Exception as exc:
                    self.log.error("SigCom directional step failed for %s: %s", self.cell_type_name, exc)


def _initialize_worker_databases(cell_lines=None):
    """
    Method ran once prior to analysis to load databases for all workers
    """
    if cell_lines is None:
        cell_lines = []

    WORKER_DATABASES["lincs"] = databases.LINCS(
        base_cell_lines=cell_lines, batch_size=2500
    )
    WORKER_DATABASES["drugbank"] = databases.DrugBank()
    WORKER_DATABASES["ncbi"] = databases.NCBI()

    interactome = from_pandas_edgelist(
        pd.read_csv(
            "data/sources/interactome_slim.tsv.gz", sep="\t", compression="gzip"
        ),
        source="source",
        target="target",
    )

    interactome = interactome.to_undirected()

    WORKER_DATABASES["interactome"] = interactome


def _run_worker(job_id, job_config):
    job_config["lincs_db"] = WORKER_DATABASES["lincs"]
    job_config["drugbank_db"] = WORKER_DATABASES["drugbank"]
    job_config["ncbi_db"] = WORKER_DATABASES["ncbi"]
    job_config["interactome"] = WORKER_DATABASES["interactome"]

    analyzer = MultitoolAnalysis(**job_config, job_id=job_id)
    analyzer.analyze()

    return job_config["cell_type_name"]


class EnsemblePipeline:
    """
    Pipeline for Parallel Execution of Analysis + Ranking Results
    """

    def __init__(self, all_jobs: list):
        self.all_jobs = all_jobs
        self.completed_cell_types = set()

        self.log = logging.getLogger("Ensemble.Pipline")

    def run_parallel_analysis(self, max_workers=4):
        """
        Run Analysis in Parallel for All Cell-Types
        """

        self.log.info("Starting parallel workers...")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_initialize_worker_databases,
            initargs=(self.all_jobs[0]["cell_lines"],),
        ) as executor:
            job_ids = range(1, len(self.all_jobs) + 1)

            results = list(
                tqdm(
                    executor.map(_run_worker, job_ids, self.all_jobs),
                    total=len(self.all_jobs),
                    desc="Processing Cell Types",
                    position=0,
                )
            )

            for cell_type in results:
                self.completed_cell_types.add(cell_type)
                self.log.info("Finished processing %s", cell_type)

class ConsensusRanker:
    def __init__(self, dataset_directory):
        self.dir_path = Path(dataset_directory)

        self.log = logging.getLogger("Consensus.Rank")

    def _borda_rank(self, ranked_lists):
        if not ranked_lists:
            return pd.DataFrame(columns=["DrugBank_ID", "Borda_Score"])
    
        borda_scores = {}
    
        for lst in ranked_lists:
            L = len(lst)
            if L == 0:
                continue
            for rank_index, drug_id in enumerate(lst):
                points = (L - rank_index) / L          
                borda_scores[drug_id] = borda_scores.get(drug_id, 0.0) + points
    
        if not borda_scores:
            return pd.DataFrame(columns=["DrugBank_ID", "Borda_Score"])
    
        results_df = pd.DataFrame(
            list(borda_scores.items()), columns=["DrugBank_ID", "Borda_Score"]
        )
        results_df.sort_values(
            by="Borda_Score", ascending=False, inplace=True, ignore_index=True
        )
        return results_df

    def _collect_cell_type_predictions(self):
        """
        Build reverse index:
        DrugBank_ID -> set(cell_type_names_that_predicted_it)
        """
        predictions = defaultdict(set)

        for consensus_file in self.dir_path.glob("consensus_ALL_DATASETS_*.csv"):
            cell_type = consensus_file.stem.replace("consensus_ALL_DATASETS_", "")
            cell_consensus_df = pd.read_csv(consensus_file)

            if "DrugBank_ID" not in cell_consensus_df.columns:
                continue

            for drug_id in cell_consensus_df["DrugBank_ID"].dropna():
                predictions[str(drug_id)].add(cell_type)

        return predictions

    @staticmethod
    def _serialize_targets(drug):
        target_rows = []
        raw_targets = getattr(drug, "targets", ()) if drug is not None else ()

        for target in raw_targets:
            target_type = getattr(target, "type", "")
            organism = getattr(target, "organism", "")
            actions = tuple(action for action in getattr(target, "drug_actions", ()) if action)
            actions_text = ", ".join(actions) if actions else "N/A"
            symbol = getattr(target, "symbol", None) or "N/A"
            name = getattr(target, "name", None) or "N/A"
            cellular_location = getattr(target, "cellular_location", None) or "N/A"
            target_id = getattr(target, "id", None) or "N/A"
            swiss_prot_id = getattr(target, "swiss_prot_id", None) or "N/A"

            target_rows.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "actions": actions_text,
                    "organism": organism if organism else "N/A",
                    "type": target_type if target_type else "N/A",
                    "cellular_location": cellular_location,
                    "target_id": target_id,
                    "swiss_prot_id": swiss_prot_id,
                }
            )

        human_protein_rows = [
            row
            for row in target_rows
            if row["type"] == "protein" and row["organism"] == "Humans"
        ]

        return target_rows, human_protein_rows

    @staticmethod
    def _format_targets_summary(human_protein_rows, max_items=12):
        if not human_protein_rows:
            return ""

        summary_items = []
        for row in human_protein_rows:
            symbol = row["symbol"]
            actions = row["actions"]
            summary_items.append(f"{symbol} ({actions})")

        unique_items = list(dict.fromkeys(summary_items))
        if len(unique_items) > max_items:
            return ", ".join(unique_items[:max_items]) + ", ..."

        return ", ".join(unique_items)

    @staticmethod
    def _format_targets_full_text(target_rows):
        if not target_rows:
            return ""

        lines = []
        for idx, row in enumerate(target_rows, start=1):
            lines.append(
                (
                    f"{idx}. {row['symbol']} | {row['name']} | "
                    f"type={row['type']} | organism={row['organism']} | "
                    f"actions={row['actions']} | location={row['cellular_location']} | "
                    f"target_id={row['target_id']} | swiss_prot_id={row['swiss_prot_id']}"
                )
            )

        return "\n".join(lines)

    @staticmethod
    def _format_targets_html(target_rows):
        if not target_rows:
            return "<div class='muted'>No targets available</div>"

        cards = []
        for row in target_rows:
            cards.append(
                (
                    "<div class='target-card'>"
                    f"<div><b>{escape(str(row['symbol']))}</b> - {escape(str(row['name']))}</div>"
                    f"<div>Actions: {escape(str(row['actions']))}</div>"
                    f"<div>Type: {escape(str(row['type']))} | Organism: {escape(str(row['organism']))}</div>"
                    f"<div>Location: {escape(str(row['cellular_location']))}</div>"
                    f"<div>Target ID: {escape(str(row['target_id']))} | UniProt: {escape(str(row['swiss_prot_id']))}</div>"
                    "</div>"
                )
            )

        return "".join(cards)

    def generate_master_report(self, hd_trial_ids=None):
        """
        Build enriched CSV and HTML report for MASTER_DRUG_RANKINGS.csv.
        """
        if hd_trial_ids is None:
            hd_trial_ids = HD_CLINICAL_TRIAL_DRUGBANK_IDS

        master_path = self.dir_path / "MASTER_DRUG_RANKINGS.csv"
        if not master_path.exists():
            self.log.error("Cannot render report: %s does not exist", master_path)
            return

        master_df = pd.read_csv(master_path).copy()
        if "DrugBank_ID" not in master_df.columns or "Borda_Score" not in master_df.columns:
            self.log.error("MASTER_DRUG_RANKINGS.csv missing required columns")
            return

        cell_type_predictions = self._collect_cell_type_predictions()

        try:
            drugbank = databases.DrugBank()
        except Exception as exc:
            self.log.error("Could not initialize DrugBank for enrichment: %s", exc)
            return

        enriched_rows = []
        for rank, row in enumerate(master_df.itertuples(index=False), start=1):
            drug_id = str(row.DrugBank_ID)
            borda_score = row.Borda_Score

            try:
                drug = drugbank.search(drug_id)
            except Exception:
                drug = None

            drug_name = getattr(drug, "name", "Unknown")
            indication = getattr(drug, "indication", "")
            mechanism = getattr(drug, "mechanism_of_action", "")
            drug_groups = tuple(group for group in getattr(drug, "groups", ()) if group)
            drug_groups_text = ", ".join(drug_groups)
            is_approved = "approved" in {group.lower() for group in drug_groups}

            target_rows, human_protein_rows = self._serialize_targets(drug)
            targets_summary = self._format_targets_summary(human_protein_rows)
            targets_full_text = self._format_targets_full_text(target_rows)
            targets_html = self._format_targets_html(target_rows)

            predicted_cell_types = sorted(cell_type_predictions.get(drug_id, set()))
            predicted_cell_types_text = ", ".join(predicted_cell_types)

            enriched_rows.append(
                {
                    "Rank": rank,
                    "DrugBank_ID": drug_id,
                    "DrugBank_Name": drug_name,
                    "Borda_Score": borda_score,
                    "Approved": is_approved,
                    "Drug_Groups": drug_groups_text,
                    "HD_Clinical_Trial": drug_id in hd_trial_ids,
                    "Predicted_Cell_Types": predicted_cell_types_text,
                    "Predicted_Cell_Type_Count": len(predicted_cell_types),
                    "Target_Count": len(target_rows),
                    "Human_Protein_Target_Count": len(human_protein_rows),
                    "Targets_Summary": targets_summary,
                    "Indication": indication,
                    "Mechanism_Of_Action": mechanism,
                    "Targets_Full": targets_full_text,
                    "_targets_html": targets_html,
                }
            )

        enriched_df = pd.DataFrame(enriched_rows)
        enriched_csv_path = self.dir_path / "MASTER_DRUG_RANKINGS_enriched.csv"
        html_report_path = self.dir_path / "MASTER_DRUG_RANKINGS_report.html"

        csv_columns = [column for column in enriched_df.columns if not column.startswith("_")]
        enriched_df[csv_columns].to_csv(enriched_csv_path, index=False)

        total_drugs = len(enriched_df)
        hd_trial_count = int(enriched_df["HD_Clinical_Trial"].sum())
        hd_trial_percent = (100.0 * hd_trial_count / total_drugs) if total_drugs else 0.0

        html_rows = []
        for row in enriched_rows:
            details_block = (
                "<details><summary>View target details</summary>"
                f"{row['_targets_html']}"
                "</details>"
            )
            html_rows.append(
                (
                    "<tr>"
                    f"<td>{row['Rank']}</td>"
                    f"<td>{escape(str(row['DrugBank_ID']))}</td>"
                    f"<td>{escape(str(row['DrugBank_Name']))}</td>"
                    f"<td>{row['Borda_Score']}</td>"
                    f"<td>{'Yes' if row['Approved'] else 'No'}</td>"
                    f"<td>{escape(str(row['Drug_Groups']))}</td>"
                    f"<td>{'Yes' if row['HD_Clinical_Trial'] else 'No'}</td>"
                    f"<td>{row['Predicted_Cell_Type_Count']}</td>"
                    f"<td>{escape(str(row['Predicted_Cell_Types']))}</td>"
                    f"<td>{row['Target_Count']}</td>"
                    f"<td>{escape(str(row['Targets_Summary']))}</td>"
                    f"<td>{details_block}</td>"
                    "</tr>"
                )
            )

        html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MASTER Drug Rankings Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; line-height: 1.4; }}
    h1 {{ margin-bottom: 8px; }}
    .summary {{ display: flex; gap: 24px; margin-bottom: 16px; flex-wrap: wrap; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px 12px; min-width: 180px; }}
    .label {{ color: #555; font-size: 0.9rem; }}
    .value {{ font-size: 1.25rem; font-weight: bold; }}
    #searchBox {{ width: 100%; max-width: 600px; padding: 8px; margin: 8px 0 16px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f7f7f7; z-index: 1; text-align: left; }}
    tr:nth-child(even) {{ background: #fcfcfc; }}
    .target-card {{ border: 1px solid #eee; border-radius: 6px; padding: 8px; margin: 6px 0; }}
    .muted {{ color: #666; }}
  </style>
</head>
<body>
  <h1>MASTER Drug Rankings</h1>
  <div class="summary">
    <div class="card">
      <div class="label">Total Drugs</div>
      <div class="value">{total_drugs}</div>
    </div>
    <div class="card">
      <div class="label">HD Clinical-Trial Drugs</div>
      <div class="value">{hd_trial_count}</div>
    </div>
    <div class="card">
      <div class="label">HD Clinical-Trial %</div>
      <div class="value">{hd_trial_percent:.1f}%</div>
    </div>
  </div>

  <input id="searchBox" type="text" placeholder="Filter rows (DrugBank ID, name, targets, or cell types)..." />

  <table id="resultsTable">
    <thead>
      <tr>
        <th>Rank</th>
        <th>DrugBank ID</th>
        <th>Drug Name</th>
        <th>Borda Score</th>
        <th>Approved</th>
        <th>Drug Groups</th>
        <th>HD Trial</th>
        <th>Cell Type Count</th>
        <th>Predicted Cell Types</th>
        <th>Target Count</th>
        <th>Target Summary</th>
        <th>Target Details</th>
      </tr>
    </thead>
    <tbody>
      {''.join(html_rows)}
    </tbody>
  </table>

  <script>
    const searchBox = document.getElementById("searchBox");
    const table = document.getElementById("resultsTable");
    const rows = Array.from(table.querySelectorAll("tbody tr"));

    searchBox.addEventListener("input", function () {{
      const query = this.value.toLowerCase();
      rows.forEach((row) => {{
        const text = row.innerText.toLowerCase();
        row.style.display = text.includes(query) ? "" : "none";
      }});
    }});
  </script>
</body>
</html>
"""
        html_report_path.write_text(html_page, encoding="utf-8")

        self.log.info("Wrote enriched report CSV: %s", enriched_csv_path)
        self.log.info("Wrote enriched report HTML: %s", html_report_path)

    def rank(self):
        dataset_folders = [dataset for dataset in self.dir_path.iterdir() if dataset.is_dir()]

        cell_type_groups = defaultdict(list)

        print(dataset_folders)

        for dataset in dataset_folders:
            cell_type_folders = [cell_type for cell_type in dataset.iterdir() if cell_type.is_dir()]
            dataset_cell_type_list = []

            for cell_type_dir in cell_type_folders:
                cell_type_name = cell_type_dir.name

                igsea_path = cell_type_dir / "promising_drug_candidates_igsea.csv"
                l2s2_path = cell_type_dir / "promising_drug_candidates_l2s2.csv"
                l2s2_updown_path = cell_type_dir / "promising_drug_candidates_l2s2_updown.csv"
                sigcom_path = cell_type_dir / "promising_drug_candidates_sigcom.csv"

                cell_tool_lists = []

                if igsea_path.exists():
                    cell_tool_lists.append(pd.read_csv(igsea_path)["DrugBank_ID"].tolist())
                if l2s2_path.exists():
                    cell_tool_lists.append(pd.read_csv(l2s2_path)["DrugBank_ID"].tolist())
                if l2s2_updown_path.exists():
                    cell_tool_lists.append(pd.read_csv(l2s2_updown_path)["DrugBank_ID"].tolist())
                if sigcom_path.exists():
                    cell_tool_lists.append(pd.read_csv(sigcom_path)["DrugBank_ID"].tolist())

                if cell_tool_lists:
                    cell_type_consensus = self._borda_rank(cell_tool_lists,)
                    cell_type_consensus_path = cell_type_dir / "tool_consensus.csv"
                    cell_type_consensus.to_csv(cell_type_consensus_path, index=False)

                    cell_type_groups[cell_type_name].append(cell_type_consensus_path)

        all_cell_type_consensus_lists = []

        for cell_name, consensus_paths in cell_type_groups.items():
            dataset_lists = []
            
            for path in consensus_paths:
                dataset_lists.append(pd.read_csv(path)["DrugBank_ID"].tolist())
                
            if dataset_lists:
                cell_consensus = self._borda_rank(dataset_lists,)

                cell_consensus.to_csv(self.dir_path / f"consensus_ALL_DATASETS_{cell_name}.csv", index=False)

                all_cell_type_consensus_lists.append(cell_consensus["DrugBank_ID"].tolist())

        self.log.info("Calculating final Master Consensus across all Cell Types...")
        if all_cell_type_consensus_lists:
            master_consensus = self._borda_rank(all_cell_type_consensus_lists)
            master_consensus.to_csv(self.dir_path / "MASTER_DRUG_RANKINGS.csv", index=False)
            self.log.info(f"Ranking complete! Check {self.dir_path / 'MASTER_DRUG_RANKINGS.csv'}")
            self.generate_master_report()
        else:
            self.log.error("No data found to rank.")


def gather_run_configs():
    """
    Helper Method for Gather Run Configurations Before Analysis
    """

    first_time = input(
        "Do you want to download the required databases ? This only needs to be done if never done before. (y/n) "
    ).lower()

    while first_time not in ["y", "n"]:
        first_time = input("Response must be y or n ").lower()

    mode = input("Do you want to rank results (r) or analyze a new dataset (a)? (r/a) ").lower()

    while mode not in ["a", "r"]:
        first_time = input("Response must be r (ranking) or a (analysis) ").lower()

    

    if mode == "r":
        ranking_data_directory = input(
        "What is the directory of the folder containing the results for ranking? "
        )

        return {
            "first_time": first_time,
            "dataset_directory": ranking_data_directory,
            "mode": mode,
        }

    run_name = "_".join(
        input(
            "What is the name of the run? (This is the name of the folder that your results will be outputted to) "
        ).split()
    )
    cell_lines = input(
        "What cell lines do you want to use for analysis? Seperate each with a space "
    ).split()
    dataset_directory = input(
        "What is the directory of the folder containing the datasets for analysis? "
    )

    directional = input("Are you using directional genes as input? (y/n)")

    while directional not in ["y", "n"]:
        directional = input("Are you using directional genes as input? (y/n)")

    return {
        "first_time": first_time,
        "run_name": run_name,
        "cell_lines": cell_lines,
        "dataset_directory": dataset_directory,
        "mode": mode,
        "directional": directional,
    }


def generate_worker_jobs(run_name, cell_lines, datasets, directional):
    """
    Helper Method for Generating Worker Job Dictionaries
    """

    jobs = []

    for dataset in datasets:
        jobs.append(
            {
                "run_name": run_name,
                "cell_type_name": dataset.stem,
                "cell_lines": cell_lines,
                "dataset_path": dataset,
                "directional": directional,
            }
        )

    return jobs


if __name__ == "__main__":
    # Configure the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("run.log", mode="a"), logging.StreamHandler()],
    )

    log = logging.getLogger("Setup")

    # Configure Parser for Command Line Flags
    parser = argparse.ArgumentParser(
        description="Drug Framework Yielding Neurodegenerative Drugs via Ensemble Repurposing",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--init",
        action="store_true",
        help="Use interactive console to initialize analysis",
        dest="init",
    )
    parser.add_argument(
        "--render-report",
        action="store_true",
        help="Generate MASTER_DRUG_RANKINGS enriched CSV + HTML report",
        dest="render_report",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/results",
        help="Directory containing MASTER_DRUG_RANKINGS.csv and consensus_ALL_DATASETS_*.csv",
        dest="results_dir",
    )

    args = parser.parse_args()

    if args.render_report:
        ranker = ConsensusRanker(Path(args.results_dir))
        ranker.generate_master_report()
        raise SystemExit(0)

    if args.init:
        run_config = gather_run_configs()

        if run_config["first_time"] == "y":
            log.info("Downloading required resources")

            databases.NCBI(update=True)
            databases.DrugBank(update=True)
            databases.LINCS(update=True)
            databases.APID(update=True)
            databases.BioGRID(update=True)
            databases.HuRI(update=True)
            databases.InnateDB(update=True)
            databases.IntAct(update=True)
            databases.SignaLink(update=True)
            databases.STRING(update=True)
            PPI()
            
            log.info("Finished downloading required resources")

        dir_path = Path(run_config["dataset_directory"])

        if (
            not dir_path.exists()
            or not dir_path.is_dir()
            or not any(dir_path.iterdir())
        ):
            log.error("%s is not a directory, does not exist, or is empty", dir_path)

        if run_config["mode"] == "r":
            ranker = ConsensusRanker(dir_path)

            ranker.rank()

        else:
            os.makedirs(f"data/results/{run_config['run_name']}", exist_ok=True)

            datasets_paths = []

            for gene_file_path in dir_path.iterdir():
                if gene_file_path.is_file():
                    datasets_paths.append(gene_file_path)

            worker_jobs = generate_worker_jobs(
                run_config["run_name"], run_config["cell_lines"], datasets_paths, run_config["directional"],
            )

            pipeline = EnsemblePipeline(worker_jobs)
            pipeline.run_parallel_analysis()
