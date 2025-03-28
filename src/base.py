"""Imports necessary modules"""

import random

import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler

from params import list_limits, list_params


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
