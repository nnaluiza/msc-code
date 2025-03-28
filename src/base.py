"""Imports necessary modules"""

import random

import pandas as pd
from scipy.io import arff
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from decision_tree import train_tree
from params import list_limits, list_params
from rules import adjust_parameters_based_on_rule, get_positive_rules


def get_data_training(name_file):
    """Reads and treats dataset files for network training"""

    data = arff.loadarff("datasets/" + name_file)
    base = pd.DataFrame(data[0])

    if "z" in base.columns:
        dataTrain = base[["x", "y", "z"]]
    else:
        dataTrain = base[["x", "y"]]

    data = StandardScaler().fit_transform(dataTrain)
    return data


def random_value_generator(limit):
    """Generates a random value between the lower and upper limits of a given param"""

    lower_limit, upper_limit = limit
    random_value = random.uniform(lower_limit, upper_limit)
    return random_value


def create_working_memory(seed, size):
    """Generates size random parameter sets and stores them in the working_memory list.
    Each parameter set is a dictionary that maps parameter names to their values"""

    limits = list_limits()
    params = list_params()
    working_memory = []

    """Specifies the random seed to guarantee the generation of the same values every time"""
    random.seed(seed)

    for i in range(int(size)):
        random_values = []
        for limit in limits:
            random_value = random_value_generator(limit)

            param = params[len(random_values)]
            if param in ["a_max", "l", "passes"]:
                random_value = int(format(random_value, ".0f"))
            else:
                random_value = float(format(random_value, ".4f"))

            random_values.append(random_value)
            dict_value = dict(zip(params, random_values))

        working_memory.append(dict_value)

    return working_memory


def create_knowledge_base(clustered_data, instance, start, end):
    labels = []
    data = []

    for item in clustered_data:
        data.append(item[0])
        label = item[1]
        labels.append(label)

    silhouette_avg = metrics.silhouette_score(data, labels, metric="euclidean")

    return {
        "e_b": instance["e_b"],
        "e_n": instance["e_n"],
        "a_max": instance["a_max"],
        "l": instance["l"],
        "a": instance["a"],
        "d": instance["d"],
        "passes": instance["passes"],
        "silhouette_avg": float(format(silhouette_avg, ".4f")),
        "execution_time": float(format(end - start, ".4f")),
        "class": 1 if float(format(silhouette_avg, ".4f")) >= 0.5 else 0,
    }


def split_knowledge_base(
    knowledge_base,
    rep,
    reps,
    seed,
):
    if knowledge_base and knowledge_base[-1]["class"] == 1:
        rules = train_tree(rep, seed)
        positive_conditions = get_positive_rules(rules)
        if positive_conditions:
            limits = adjust_parameters_based_on_rule(positive_conditions, limits)
            return limits


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
