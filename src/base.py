"""Imports necessary modules"""

import random

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import metrics
from sklearn.datasets import load_iris, make_blobs, make_circles, make_moons
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


def create_working_memory(seed, size, file_limit_path):
    """Generates size random parameter sets and stores them in the working_memory list.
    Each parameter set is a dictionary that maps parameter names to their values."""

    limits = list_limits(file_limit_path)
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


def compute_clustering_score(silhouette_avg, davies_bouldin_index, calinski_harabasz_index, adjusted_rand_index, global_error):
    """Computes a combined clustering score based on five metrics and classifies the parameter set as good (1) or bad (0)."""
    # Normalize the metrics to [0, 1] where higher is better

    # Silhouette Score: -1 to 1 -> 0 to 1
    normalized_silhouette = (silhouette_avg + 1) / 2

    # Davies-Bouldin Index: 0 to infinity (lower is better) -> 0 to 1 (higher is better)
    normalized_davies_bouldin = np.exp(-davies_bouldin_index)

    # Calinski-Harabasz Index: 0 to infinity (higher is better) -> 0 to 1
    max_calinski = 300
    log_calinski = np.log(calinski_harabasz_index + 1)
    log_max = np.log(max_calinski + 1)
    log_min = np.log(1 + 1)
    normalized_calinski_harabasz = (log_calinski - log_min) / (log_max - log_min)
    normalized_calinski_harabasz = np.clip(normalized_calinski_harabasz, 0, 1)

    # Adjusted Rand Index: -1 to 1 -> 0 to 1 (or None)
    normalized_adjusted_rand = None
    if adjusted_rand_index is not None:
        normalized_adjusted_rand = (adjusted_rand_index + 1) / 2

    # Global Error: 0 to infinity (lower is better) -> 0 to 1 (higher is better)
    max_global_error = 500
    normalized_global_error = np.exp(-global_error / max_global_error)
    normalized_global_error = np.clip(normalized_global_error, 0, 1)

    weights = {
        "silhouette": 0.20,
        "davies_bouldin": 0.15,
        "calinski_harabasz": 0.15,
        "adjusted_rand": 0.25,
        "global_error": 0.25,
    }

    if adjusted_rand_index is None:
        total_weight = weights["silhouette"] + weights["davies_bouldin"] + weights["calinski_harabasz"] + weights["global_error"]
        scale_factor = 1 / total_weight
        weight_silhouette = weights["silhouette"] * scale_factor
        weight_davies_bouldin = weights["davies_bouldin"] * scale_factor
        weight_calinski_harabasz = weights["calinski_harabasz"] * scale_factor
        weight_global_error = weights["global_error"] * scale_factor

        combined_score = (
            weight_silhouette * normalized_silhouette
            + weight_davies_bouldin * normalized_davies_bouldin
            + weight_calinski_harabasz * normalized_calinski_harabasz
            + weight_global_error * normalized_global_error
        )
    else:
        combined_score = (
            weights["silhouette"] * normalized_silhouette
            + weights["davies_bouldin"] * normalized_davies_bouldin
            + weights["calinski_harabasz"] * normalized_calinski_harabasz
            + weights["adjusted_rand"] * normalized_adjusted_rand
            + weights["global_error"] * normalized_global_error
        )

    threshold = 0.65
    class_label = 1 if combined_score >= threshold else 0

    return combined_score, class_label


def create_knowledge_base(clustered_data, instance, start, end, global_error, true_labels=None):
    """Creates a knowledge base entry with clustering evaluation metrics and global error."""

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
    if true_labels is not None:
        if len(true_labels) == len(labels):
            adjusted_rand = metrics.adjusted_rand_score(true_labels, labels)
        else:
            print("Warning: True labels length does not match clustered data length. ARI will be None.")

    combined_score, class_label = compute_clustering_score(
        silhouette_avg, davies_bouldin, calinski_harabasz, adjusted_rand, global_error
    )

    return {
        "e_b": instance["e_b"],
        "e_n": instance["e_n"],
        "a_max": instance["a_max"],
        "l": instance["l"],
        "a": instance["a"],
        "d": instance["d"],
        "passes": instance["passes"],
        "silhouette_avg": float(format(silhouette_avg, ".4f")),
        "davies_bouldin_index": float(format(davies_bouldin, ".4f")),
        "calinski_harabasz_index": float(format(calinski_harabasz, ".4f")),
        "adjusted_rand_index": float(format(adjusted_rand, ".4f")) if adjusted_rand is not None else None,
        "global_error": float(format(global_error, ".4f")),
        "combined_score": float(format(combined_score, ".4f")),
        "execution_time": float(format(end - start, ".4f")),
        "class": class_label,
    }


def split_knowledge_base(rules, knowledge_base_file, file_limit_path):
    """Processes knowledge base and rules to update parameter limits."""
    knowledge_base = pd.read_csv(knowledge_base_file, delimiter=",", skiprows=2).to_dict("records")
    if knowledge_base:
        limits = list_limits(file_limit_path)
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
