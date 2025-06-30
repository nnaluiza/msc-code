"""Imports necessary modules"""

import random

import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from sklearn import metrics
from sklearn.datasets import load_iris, make_blobs, make_circles, make_moons
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from params import list_limits, list_params
from rules import adjust_parameters_based_on_rule, get_positive_rules


def get_data_training(dataset_name):
    """Reads and processes dataset based on the provided dataset name.
    Returns standardized data as a numpy array and true labels if available."""
    true_labels = None

    if dataset_name.endswith(".arff"):
        data = arff.loadarff("datasets/" + dataset_name)
        base = pd.DataFrame(data[0])

        if "z" in base.columns:
            dataTrain = base[["x", "y", "z"]]
        else:
            dataTrain = base[["x", "y"]]

        if "class" in base.columns:
            true_labels = base["class"].apply(lambda x: float(x.decode("utf-8")) if isinstance(x, bytes) else float(x)).values

        data = StandardScaler().fit_transform(dataTrain)
        return data, true_labels

    else:
        if dataset_name == "iris":
            dataset = load_iris()
            data = dataset.data
            true_labels = dataset.target
        elif dataset_name == "moons":
            data, true_labels = make_moons(n_samples=1000, noise=0.1, random_state=42)
        elif dataset_name == "blobs":
            data, true_labels = make_blobs(n_samples=1000, centers=3, random_state=42)
        elif dataset_name == "circles":
            data, true_labels = make_circles(n_samples=2000, noise=0.05, factor=0.3, random_state=42)
        else:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. Supported datasets: .arff files, 'iris', 'moons', 'blobs', 'circles'"
            )

        data = StandardScaler().fit_transform(data)
        return data, true_labels


def random_value_generator(limit, is_discrete=False):
    """Generates a random value between the lower and upper limits of a given param"""
    lower_limit, upper_limit = limit
    if is_discrete:
        return random.randint(int(lower_limit), int(upper_limit))
    else:
        random_value = random.uniform(lower_limit, upper_limit)
        return float(format(random_value, ".4f"))


def create_working_memory(seed, size, file_limit_path, distance_metric):
    """Generates size random parameter sets and stores them in the working_memory list.
    Each parameter set is a dictionary that maps parameter names to their values."""
    limits = list_limits(file_limit_path, distance_metric)
    params = list_params()

    if len(limits) != len(params):
        raise ValueError(
            f"Number of limits ({len(limits)}) does not match number of parameters ({len(params)}). "
            f"Parameters: {params}, Limits: {limits}"
        )

    discrete_params = ["a_max", "l", "passes"]
    working_memory = []

    for i in range(int(size)):
        random_values = []
        for j, limit in enumerate(limits):
            param = params[j]
            is_discrete = param in discrete_params
            random_value = random_value_generator(limit, is_discrete=is_discrete)
            random_values.append(random_value)

        dict_value = dict(zip(params, random_values))
        working_memory.append(dict_value)

    return working_memory


def create_knowledge_base(clustered_data, instance, start, end, global_error, num_clusters, rep_number, true_labels=None):
    """Creates a knowledge base entry with clustering evaluation metrics, global error, and number of clusters."""
    labels = []
    data = []

    for item in clustered_data:
        data.append(item[0])
        label = item[1]
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    silhouette_avg = metrics.silhouette_score(data, labels, metric="euclidean")
    davies_bouldin = metrics.davies_bouldin_score(data, labels)
    calinski_harabasz = metrics.calinski_harabasz_score(data, labels)

    adjusted_rand = None
    rand_index = None
    if true_labels is not None:
        if len(true_labels) == len(labels):
            adjusted_rand = metrics.adjusted_rand_score(true_labels, labels)
            rand_index = metrics.rand_score(true_labels, labels)
        else:
            print("Warning: True labels length does not match clustered data length. ARI and RI will be None.")

    total_time = end - start
    if isinstance(total_time, float):
        total_seconds = total_time
    else:
        total_seconds = total_time.total_seconds()

    execution_time = total_seconds
    execution_time_str = f"{execution_time:.4f}"
    execution_time_str = str(execution_time)[: str(execution_time).find(".") + 5]
    execution_time = float(execution_time_str)

    return {
        "rep_number": rep_number,
        "e_b": instance["e_b"],
        "e_n": instance["e_n"],
        "a_max": instance["a_max"],
        "l": instance["l"],
        "a": instance["a"],
        "d": instance["d"],
        "passes": instance["passes"],
        "clusters_number": int(num_clusters),
        # "silhouette_avg": float(format(silhouette_avg, ".4f")),
        # "davies_bouldin_index": float(format(davies_bouldin, ".4f")),
        # "calinski_harabasz_index": float(format(calinski_harabasz, ".4f")),
        "adjusted_rand_index": float(format(adjusted_rand, ".4f")) if adjusted_rand is not None else None,
        # "rand_index": float(format(rand_index, ".4f")) if rand_index is not None else None,
        "dunn_index": float(format(dunn_index(data, labels), ".4f")),
        "global_error": float(format(global_error, ".4f")),
        "execution_time": float(format(execution_time, ".4f")),
        "objective_function": None,  # Objective function will be assigned after sorting
        "class": None,  # Class will be assigned after sorting
    }


def connectivity_index(data, labels, k=10):
    """Computes the Connectivity Index for clustering validation.
    Lower values indicate better clustering (neighbors are in the same cluster)."""
    n_samples = data.shape[0]
    connectivity = 0.0
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n_samples)).fit(data)
    # For each point, get its k nearest neighbors (excluding itself)
    distances, indices = nbrs.kneighbors(data)
    for i in range(n_samples):
        for j in range(1, min(k + 1, n_samples)):  # skip the first neighbor (itself)
            if labels[i] != labels[indices[i, j]]:
                connectivity += 1.0 / j
    return connectivity


def dunn_index(data, labels, metric="euclidean"):
    """Calculates the Dunn Index for clustering"""

    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)

    if n_clusters < 2:
        return 0.0

    intra_dists = []
    inter_dists = []

    for i in unique_clusters:
        cluster_i = data[labels == i]
        if len(cluster_i) < 2:
            intra_dists.append(0.0)
        else:
            dists = cdist(cluster_i, cluster_i, metric=metric)
            intra_dists.append(np.max(dists))

        for j in unique_clusters:
            if j > i:
                cluster_j = data[labels == j]
                dists = cdist(cluster_i, cluster_j, metric=metric)
                inter_dists.append(np.min(dists))

    max_intra = np.max(intra_dists)
    min_inter = np.min(inter_dists)

    if max_intra == 0:
        return 0.0

    return min_inter / max_intra


def compactness_index(data, labels):
    """Computes the average intra-cluster distance (compactness). Lower is better."""
    unique_labels = np.unique(labels)
    compactness = 0.0
    count = 0
    for label in unique_labels:
        cluster_points = data[labels == label]
        if len(cluster_points) > 1:
            dists = cdist(cluster_points, cluster_points)
            # sum of upper triangle distances (excluding diagonal)
            sum_dists = np.sum(np.triu(dists, 1))
            num_pairs = (len(cluster_points) * (len(cluster_points) - 1)) / 2
            compactness += sum_dists / num_pairs
            count += 1
    if count == 0:
        return 0.0
    return compactness / count


def separation_index(data, labels):
    """Computes the minimum distance between cluster centroids (separation). Higher is better."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    centroids = np.array([data[labels == label].mean(axis=0) for label in unique_labels])
    dists = cdist(centroids, centroids)
    # get the minimum nonzero distance
    min_dist = np.min(dists[np.nonzero(dists)])
    return min_dist


def classify_knowledge_base(entries, rep, reps, weights=None):
    """
    Classifies and ranks knowledge base entries using a weighted sum of normalized metrics.
    Metrics:
        - Dunn index (higher is better)
        - Separation (higher is better)
        - Global error (lower is better)
        - Compactness (lower is better)
    The objective function is a weighted sum of these normalized values.
    Entries are ranked by objective function (higher is better).
    """
    if not entries:
        return None

    # Default weights if not provided
    if weights is None:
        weights = {"dunn": 0.25, "separation": 0.20, "error": 0.35, "compactness": 0.20}

    # Extract metrics
    dunn_indices = np.array([entry.get("dunn_index", 0.0) for entry in entries], dtype=np.float64)
    global_errors = np.array([entry.get("global_error", 0.0) for entry in entries], dtype=np.float64)
    separation_indices = np.array([entry.get("separation_index", 0.0) for entry in entries], dtype=np.float64)
    compactness_indices = np.array([entry.get("compactness_index", 0.0) for entry in entries], dtype=np.float64)

    # Normalization helper
    def normalize(arr):
        min_v, max_v = np.min(arr), np.max(arr)
        if max_v != min_v:
            return (arr - min_v) / (max_v - min_v)
        else:
            return np.zeros_like(arr)

    norm_dunn = normalize(dunn_indices)
    norm_error = normalize(global_errors)
    norm_separation = normalize(separation_indices)
    norm_compactness = normalize(compactness_indices)

    # Compute objective function (higher is better)
    objectives = (
        weights["dunn"] * norm_dunn
        + weights["separation"] * norm_separation
        + weights["error"] * (1 - norm_error)
        + weights["compactness"] * (1 - norm_compactness)
    )

    for i, entry in enumerate(entries):
        entry["objective_function"] = float(format(objectives[i], ".4f"))
        # Optionally, set class 1 for top 30% (or any threshold)
        entry["class"] = 1 if objectives[i] >= np.percentile(objectives, 70) else 0

    # Sort entries by objective function (descending)
    sorted_entries = [x for _, x in sorted(zip(objectives, entries), key=lambda pair: pair[0], reverse=True)]
    return sorted_entries


def split_knowledge_base(rules, knowledge_base_file, file_limit_path, distance):
    """Processes knowledge base and rules to update parameter limits."""
    knowledge_base = pd.read_csv(knowledge_base_file, delimiter=",", skiprows=4).to_dict("records")
    if knowledge_base:
        limits = list_limits(file_limit_path, distance)
        positive_conditions = get_positive_rules(rules)
        if positive_conditions:
            new_limits = adjust_parameters_based_on_rule(positive_conditions, limits)
            return new_limits


def get_real_labels(obj, df):
    """Extracts the actual labels from the data"""
    obj_tuple = [(tuple(array), value) for array, value in obj]
    messy_classes = []

    for array_tuple, value in obj_tuple:
        for i, row in df.iterrows():
            if "z_t" in df.columns:
                if (row["x_t"], row["y_t"], row["z_t"]) == array_tuple:
                    messy_classes.append(float(row["class"].decode("utf-8")))
                    break
            elif (row["x_t"], row["y_t"]) == array_tuple:
                messy_classes.append(float(row["class"].decode("utf-8")))
                break
    return messy_classes
