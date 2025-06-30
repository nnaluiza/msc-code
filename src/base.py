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


def create_working_memory(seed, size, file_limit_path, distance):
    """Generates size random parameter sets using Latin Hypercube Sampling."""
    limits = list_limits(file_limit_path, distance)
    params = list_params()
    if len(limits) != len(params):
        raise ValueError(
            f"Number of limits ({len(limits)}) does not match number of parameters ({len(params)}). "
            f"Parameters: {params}, Limits: {limits}"
        )
    discrete_params = ["a_max", "l", "passes"]
    sampler = qmc.LatinHypercube(d=len(limits), seed=seed)
    sample = sampler.random(n=size)
    working_memory = []
    for i in range(size):
        random_values = []
        for j, limit in enumerate(limits):
            lower_limit, upper_limit = limit
            param = params[j]
            is_discrete = param in discrete_params
            scaled_value = sample[i][j] * (upper_limit - lower_limit) + lower_limit
            if is_discrete:
                random_value = int(round(scaled_value))
            else:
                random_value = float(format(scaled_value, ".4f"))
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


def classify_knowledge_base(entries, rep, reps, normalize=True):
    """Processes knowledge base entries by sorting them by a composite score based on global_error, dunn_index, and execution_time.
    Classifies entries with threshold based on percentile. For rep=1, 80% of solutions are classified as good (class=1).
    Threshold decreases by 5% per rep until reaching 10%. Normalized metrics are used internally but not stored."""
    if not entries:
        return None

    sorted_entries = sorted(entries, key=lambda x: x["global_error"])
    global_errors = [entry["global_error"] for entry in entries]
    if normalize:
        min_error = min(global_errors)
        max_error = max(global_errors)
        if max_error != min_error:
            normalized_errors = [(error - min_error) / (max_error - min_error) for error in global_errors]
        else:
            normalized_errors = [0.0] * len(global_errors)
    else:
        normalized_errors = global_errors

    dunn_indices = [entry["dunn_index"] for entry in entries]
    min_dunn = min(dunn_indices)
    max_dunn = max(dunn_indices)
    if max_dunn != min_dunn:
        normalized_dunn = [(max_dunn - di) / (max_dunn - min_dunn) for di in dunn_indices]
    else:
        normalized_dunn = [0.0] * len(dunn_indices)

    execution_times = [entry["execution_time"] for entry in entries]
    min_time = min(execution_times)
    max_time = max(execution_times)
    if max_time != min_time:
        normalized_times = [(et - min_time) / (max_time - min_time) for et in execution_times]
    else:
        normalized_times = [0.0] * len(execution_times)

    # Add connectivity index to the composite score (optional, here as 0.2 weight)
    connectivity_indices = [entry.get("connectivity_index", 0.0) for entry in entries]
    min_conn = min(connectivity_indices)
    max_conn = max(connectivity_indices)
    if max_conn != min_conn:
        normalized_conn = [(ci - min_conn) / (max_conn - min_conn) for ci in connectivity_indices]
    else:
        normalized_conn = [0.0] * len(connectivity_indices)

    composite_scores = [
        normalized_errors[i] * 0.4 + normalized_dunn[i] * 0.25 + normalized_times[i] * 0.15 + normalized_conn[i] * 0.2
        for i in range(len(entries))
    ]

    # Normalize Dunn index and global error
    dunn_indices = np.array(dunn_indices)
    global_errors = np.array(global_errors)
    norm_dunn = (dunn_indices - np.min(dunn_indices)) / (np.max(dunn_indices) - np.min(dunn_indices) + 1e-8)
    norm_error = (global_errors - np.min(global_errors)) / (np.max(global_errors) - np.min(global_errors) + 1e-8)

    # Dynamic reward/penalty system based on Dunn index mean and std
    mean_dunn = np.mean(dunn_indices)
    std_dunn = np.std(dunn_indices)
    dunn_reward = 0.1  # Subtract from score (reward)
    dunn_penalty = 0.15  # Add to score (penalty)

    # Use MAD (Median Absolute Deviation) for reward/penalty thresholds
    def mad(arr):
        med = np.median(arr)
        return np.median(np.abs(arr - med))

    dunn_median = np.median(norm_dunn)
    dunn_mad = mad(norm_dunn)
    error_median = np.median(norm_error)
    error_mad = mad(norm_error)
    error_reward = 0.1
    error_penalty = 0.15

    adjusted_scores = []
    for i, score in enumerate(composite_scores):
        dunn = norm_dunn[i]
        error = norm_error[i]
        adjusted_score = score
        # Reward/penalize based on MAD
        if dunn > dunn_median + dunn_mad:
            adjusted_score -= dunn_reward
        elif dunn < dunn_median - dunn_mad:
            adjusted_score += dunn_penalty
        if error < error_median - error_mad:
            adjusted_score -= error_reward
        elif error > error_median + error_mad:
            adjusted_score += error_penalty
        adjusted_scores.append(adjusted_score)

    initial_percentile = 80
    min_percentile = 10
    decrement = 10
    current_percentile = max(min_percentile, initial_percentile - (rep - 1) * decrement)
    threshold = np.percentile(adjusted_scores, current_percentile)

    for entry, score in zip(sorted_entries, adjusted_scores):
        entry["objective_function"] = float(format(score, ".4f"))
        entry["class"] = 1 if score <= threshold else 0

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
