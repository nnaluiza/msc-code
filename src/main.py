"""Imports necessary modules"""

import random

from base import create_working_memory, get_data_training
from gng import GrowingNeuralGas
from utils import aux_folders, export_working_memory_csv, show_working_memory


def run_model(seed, size, reps):
    """Runs train_network a specified number of times"""

    for i in range(1, reps + 1):
        print(f"Rep {i} of {reps} for train model")
        print("-" * 100 + "\n")
        train_network(seed, size, i, reps)
        print(f"Completed rep {i} of {reps}")
        print("-" * 100 + "\n")


def train_network(seed, size, rep, reps):
    """Specifies the random seed for maintaining repeatability in training."""

    random.seed(seed)

    print("Generating working memory...")
    working_memory = create_working_memory(seed, size)
    export_working_memory_csv(working_memory, seed, rep, reps)
    print("Done.\n")

    print("Gathering data...")
    data = get_data_training("chainlink.arff")
    print("Done.\n")

    print("Starting training...\n")
    aux_folders()

    # if len(working_memory) < 100:
    #     print(f"Error: working_memory has only {len(working_memory)} entries, need at least 100.")
    #     return

    selected_instances = random.sample(working_memory, 5)

    for i, instance in enumerate(selected_instances, 1):
        print(f"Iteration {i} of 100")
        values = list(instance.values())
        print(f"Chosen instance: {instance}")

        print("Fitting neural network...\n")
        gng = GrowingNeuralGas(data, seed)
        gng.fit_network(e_b=values[0], e_n=values[1], a_max=values[2], l=values[3], a=values[4], d=values[5], passes=values[6])

        print(f"\nFound {gng.number_of_clusters()} clusters.\n")
        print("-" * 100)

    print("\nTraining all done.\n")


def main(params):
    """Parses command-line parameters and initiates the model training process."""
    if len(params) != 3:
        print("Error: Expected 3 parameters (seed, size, reps)")
        print("Usage: python run.py <seed> <size> <reps>")
        return

    seed = int(params[0])
    size = int(params[1])
    reps = int(params[2])

    run_model(seed, size, reps)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
