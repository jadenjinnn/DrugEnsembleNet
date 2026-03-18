"""
usage:
"""

import argparse
import concurrent.futures
import logging
import os
import re
from pathlib import Path


import numpy as np
import pandas as pd
from IGSEA import IGSEA
from l2s2 import enrich_l2s2_single_set
from networkx import from_pandas_edgelist
from tqdm import tqdm

import databases
from network_proximity_improved import get_proximities
from ineractome import PPI

WORKER_DATABASES = {}


class MultitoolAnalysis:
    """
    Ensemble Method for Drug Prioritization
    """

    def __init__(
        self,
        run_name,
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

    def _get_promising_drug_candidates(self, results_df, drugbank):
        """
        Applies PPI network generated to find likely drug candidates
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
            self.network_proximity_results.reset_index()[
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

        promising_drug_candidates = promising_drug_candidates[
            [
                "DrugBank_ID",
                "DrugBank_Name",
                "Indication",
                "Targets",
                "Proximity",
                # "Normalized_Enrichment_Score",
                "FDR",
            ]
        ].sort_values(
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
        #     index_col=0,
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

        self.log.info("Successfully Performed Inverted Gene Set Enrichment Analysis")
        self.log.info("Performing Analysis with Single Set L2S2")

        self.l2s2_results = enrich_l2s2_single_set(
            list(self.disease_genes.keys()),
            self.drugbank,
            self.cell_type_name,
            self.run_name,
        )

        self.log.info("Successfully Performed L2S2 Single Set Analysis")

        self.valid_drugbank_ids = set(
            self.network_proximity_results[
                self.network_proximity_results["Distance"] < self.distance_threshold
            ].index
        )

        self.log.info("Selecting Promising Drug Candidates using IGSEA Results")
        self.igsea_promising_drug_candidates = self._get_promising_drug_candidates(
            self.igsea_results, self.drugbank
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

    def rank(self):
        """
        Ranks all Datasets by Cell-Type then Overall, Outputting Both Results
        """

        pass


def gather_run_configs():
    """
    Helper Method for Gather Run Configurations Before Analysis
    """

    first_time = input(
        "Do you want to download the required databases ? This only needs to be done if never done before. (y/n) "
    ).lower()

    while first_time not in ["y", "n"]:
        first_time = input("Response must be y or n ").lower()

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

    return {
        "first_time": first_time,
        "run_name": run_name,
        "cell_lines": cell_lines,
        "dataset_directory": dataset_directory,
    }


def generate_worker_jobs(run_name, cell_lines, datasets):
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

    args = parser.parse_args()

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
            databases.INstruct(update=True)
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

        os.makedirs(f"data/results/{run_config['run_name']}", exist_ok=True)

        datasets_paths = []

        for gene_file_path in dir_path.iterdir():
            if gene_file_path.is_file():
                datasets_paths.append(gene_file_path)

        worker_jobs = generate_worker_jobs(
            run_config["run_name"], run_config["cell_lines"], datasets_paths
        )

        pipeline = EnsemblePipeline(worker_jobs)
        pipeline.run_parallel_analysis()
