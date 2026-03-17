import logging

import networkx as nx
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.bandwidths import bw_silverman


def get_proximities(
    run_name,
    cell_type_name,
    disease_genes,
    drugbank_db,
    undirected_interactome,
    reps=1e4,
    seed=12345,
):
    """
    Calculates Drug-Disease Proximity
    """

    log = logging.getLogger("network_proximity.get_proximities")

    drug_traget_data = []

    nodes = set(undirected_interactome.nodes())

    for drug in drugbank_db.database:
        valid_symbols = {
            target.symbol
            for target in drug.targets
            if target.type == "protein"
            and target.organism == "Humans"
            and target.symbol is not None
        }

        # only keep targets in the interactome
        valid_symbols = valid_symbols & nodes

        if valid_symbols:
            drug_traget_data.append(
                {
                    "DrugBank_ID": drug.id,
                    "DrugBank_Name": drug.name,
                    "Targets": list(valid_symbols),
                    "Target_Count": len(valid_symbols),
                }
            )

    drug_target_df = pd.DataFrame(drug_traget_data)

    # only keep genes in the interactome
    valid_disease_genes = set(disease_genes) & nodes

    log.info("Computing SSSP for %d disease genes... ", len(disease_genes))

    raw_distances = nx.multi_source_dijkstra_path_length(
        undirected_interactome, valid_disease_genes
    )

    degrees = dict(undirected_interactome.degree(valid_disease_genes))
    node_distances = {}

    for node in nodes:
        dist = raw_distances.get(node, np.inf)

        if dist != np.inf:
            w = -np.log(degrees[node] + 1) if node in valid_disease_genes else 0
            node_distances[node] = dist + w
        else:
            node_distances[node] = np.inf

    log.info("Successfully computed SSSP for disease genes")

    valid_distances = np.array(
        [distance for distance in node_distances.values() if distance != np.inf]
    )

    drug_target_df["Distance"] = drug_target_df["Targets"].apply(
        lambda targets: np.mean([node_distances.get(t, np.inf) for t in targets])
    )

    drug_target_df = drug_target_df[drug_target_df["Distance"] != np.inf].copy()

    log.info("Computing reference distribution with %d permutations", reps)

    rng = np.random.default_rng(seed)

    unique_target_counts = drug_target_df["Target_Count"].unique()

    ref_dist_list = []

    for count in unique_target_counts:
        num_drugs = (drug_target_df["Target_Count"] == count).sum()
        total_samples = reps * num_drugs

        rand_indices = rng.integers(
            0, len(valid_distances), size=(int(total_samples), int(count))
        )

        means = valid_distances[rand_indices].mean(axis=1)
        ref_dist_list.append(means)

    reference_distribution = np.concatenate(ref_dist_list)
    mean_ref = np.mean(reference_distribution)
    std_ref = np.std(reference_distribution)

    log.info("Reference Dist - Mean: %.3f, Std: %.3f", mean_ref, std_ref)

    real_distances = drug_target_df["Distance"].values

    # if len(reference_distribution) > 1000000:
    #     kde_ref_data = rng.choice(
    #         reference_distribution, size=1000000, replace=False)
    # else:

    log.info("Calculating threshold, this may take some time...")

    kde_ref_data = reference_distribution

    bw_real = bw_silverman(real_distances)
    bw_ref = bw_silverman(kde_ref_data)
    bandwidth = np.mean([bw_real, bw_ref])

    grid_size = 2000
    grid_min = min(real_distances.min(), reference_distribution.min())
    grid_max = max(real_distances.max(), reference_distribution.max())
    grid = np.linspace(grid_min, grid_max, grid_size)

    ref_kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian", rtol=1 / grid_size)
    reference_pdf = np.exp(
        ref_kde.fit(kde_ref_data[:, np.newaxis]).score_samples(grid[:, np.newaxis])
    )

    peaks, peaks_properties = find_peaks(
        reference_pdf, height=(np.mean(reference_pdf), None)
    )

    rel_height = 1 - (np.mean(reference_pdf) / peaks_properties["peak_heights"][0])
    widths, width_heights, left_ips, right_ips = peak_widths(
        reference_pdf, peaks[[0]], rel_height=rel_height
    )
    threshold = round(grid[left_ips.astype(int)].item(), 2)

    ref_below_thresh = (
        (reference_distribution < threshold).sum() / len(reference_distribution) * 100
    )
    drugs_below_thresh = (
        (drug_target_df["Distance"] < threshold).sum() / len(drug_target_df) * 100
    )

    drug_target_df["Proximity"] = (drug_target_df["Distance"] - mean_ref) / std_ref

    drug_target_df.sort_values(by="Proximity", inplace=True)

    with open(
        f"data/results/{run_name}/{cell_type_name.replace(' ', '')}/network_proximities.csv",
        "w",
    ) as outfile:
        outfile.write(
            f"# Reference distribution mean: {mean_ref:.3f} and standard deviation: {std_ref:.3f}\n"
        )
        outfile.write(f"# Threshold: {threshold}\n")
        outfile.write(
            f"# {ref_below_thresh:.2f}% of reference distribution below threshold, {drugs_below_thresh:.2f}% of distances distribution below threshold\n"
        )

        drug_target_df.to_csv(outfile, index=False)

    return drug_target_df
