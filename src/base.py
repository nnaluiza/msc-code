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


def create_knowledge_base(clustered_data, instance, start, end, global_error, num_clusters, true_labels=None):
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
    if true_labels is not None:
        if len(true_labels) == len(labels):
            adjusted_rand = metrics.adjusted_rand_score(true_labels, labels)
        else:
            print("Warning: True labels length does not match clustered data length. ARI will be None.")

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
        "e_b": instance["e_b"],
        "e_n": instance["e_n"],
        "a_max": instance["a_max"],
        "l": instance["l"],
        "a": instance["a"],
        "d": instance["d"],
        "passes": instance["passes"],
        "clusters_number": int(num_clusters),
        "silhouette_avg": float(format(silhouette_avg, ".4f")),
        "davies_bouldin_index": float(format(davies_bouldin, ".4f")),
        "calinski_harabasz_index": float(format(calinski_harabasz, ".4f")),
        "adjusted_rand_index": float(format(adjusted_rand, ".4f")) if adjusted_rand is not None else None,
        "global_error": float(format(global_error, ".4f")),
        "execution_time": float(format(execution_time, ".4f")),
        "class": None,  # Class will be assigned after sorting
    }


def classify_knowledge_base(entries, rep, reps, normalize=True):
    """Processes knowledge base entries by sorting them by global error, and classifies entries with threshold based on percentile.
    For rep=1, 80% of solutions are classified as good (class=1). Threshold decreases by 5% per rep until reaching 10%."""

    if not entries:
        return None

    sorted_entries = sorted(entries, key=lambda x: x["global_error"])
    global_errors = [entry["global_error"] for entry in sorted_entries]

    if normalize:
        min_error = min(global_errors)
        max_error = max(global_errors)
        if max_error != min_error:
            normalized_errors = [(error - min_error) / (max_error - min_error) for error in global_errors]
        else:
            normalized_errors = [0.0] * len(global_errors)
        for entry, norm_error in zip(sorted_entries, normalized_errors):
            entry["normalized_global_error"] = float(format(norm_error, ".4f"))
        error_to_compare = [entry["normalized_global_error"] for entry in sorted_entries]
    else:
        for entry in sorted_entries:
            entry["normalized_global_error"] = None
        error_to_compare = global_errors

    initial_percentile = 80
    min_percentile = 10
    decrement = 5
    current_percentile = max(min_percentile, initial_percentile - (rep - 1) * decrement)
    threshold = np.percentile(error_to_compare, current_percentile)

    for entry in sorted_entries:
        compare_value = entry["normalized_global_error"] if normalize else entry["global_error"]
        entry["class"] = 1 if compare_value <= threshold else 0

    return sorted_entries


def split_knowledge_base(rules, knowledge_base_file, file_limit_path):
    """Processes knowledge base and rules to update parameter limits."""
    knowledge_base = pd.read_csv(knowledge_base_file, delimiter=",", skiprows=4).to_dict("records")
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
