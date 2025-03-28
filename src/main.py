"""Imports necessary modules"""

import random
import time

from base import create_knowledge_base, create_working_memory, get_data_training
from gng import GrowingNeuralGas
from utils import (
    aux_folders,
    export_and_upload_logs,
    export_clustered_data,
    export_knowledge_base_csv,
    export_working_memory_csv,
)


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

    knowledge_base = []
    selected_instances = random.sample(working_memory, 5)

    for i, instance in enumerate(selected_instances, 1):
        aux_folders(seed, rep, reps, i)

        print(f"Iteration {i} of x")
        values = list(instance.values())
        print(f"Chosen instance: {instance}")

        print("Fitting neural network...\n")
        start = time.time()

        gng = GrowingNeuralGas(data, seed, rep, reps, i)
        gng.fit_network(e_b=values[0], e_n=values[1], a_max=values[2], l=values[3], a=values[4], d=values[5], passes=values[6])
        export_clustered_data(gng.cluster_data(), seed, rep, reps, i)

        end = time.time()

        if gng.number_of_clusters() > 1:
            print("\nFound %d clusters.\n" % gng.number_of_clusters())
            knowledge_base.append(create_knowledge_base(gng.cluster_data(), instance, start, end))

        else:
            print("\nFound %d cluster.\n" % gng.number_of_clusters())
            print("Only one cluster found. The training and the parameter set used will be disregarded.\n")

        print("-" * 100)

    print("Generating knowledge base...")
    export_knowledge_base_csv(knowledge_base, seed, rep, reps)
    print("Done.\n")

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

    # Export and upload logs at the end
    suffix = f"_seed{seed}_reps{reps}"
    export_and_upload_logs(
        base_dir="output",  # Directory where ZIP will be saved locally
        suffix=suffix,  # Unique identifier for this run
        folder_id="YOUR_FOLDER_ID",  # Replace with your Google Drive folder ID
        key_file="service-account-key.json",  # Path to your service account key
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
