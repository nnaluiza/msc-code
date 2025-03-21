"""Imports necessary modules"""

import os
import random

import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler

from gng import GrowingNeuralGas

"""Sets the random seed for reproducibility"""
seed = 1108


"""Defines the global random seed for reproducibility"""


def train_network():
    """Specifies the random seed for maintaining repeatability in training."""
    random.seed(seed)

    """Generates the dataset from the ARFF file"""
    print("Generating data...")
    """Construct the path relative to main.py's location"""
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "chainlink.arff")
    data = arff.loadarff(dataset_path)
    base = pd.DataFrame(data[0])
    dataTrain = base[["x", "y"]]
    data = StandardScaler().fit_transform(dataTrain)
    print("Done.\n")

    """Trains the model"""
    print("Fitting neural network...\n")
    gng = GrowingNeuralGas(data, seed)
    gng.fit_network(
        e_b=0.1,
        e_n=0.006,
        a_max=10,
        l=200,
        a=0.5,
        d=0.995,
        passes=8,
        plot_evolution=False,
    )

    """Reports the number of clusters found"""
    print("Found %d clusters." % gng.number_of_clusters())

    """Indicates completion of the training process"""
    print("\nTraining all done.\n")


if __name__ == "__main__":
    train_network()
