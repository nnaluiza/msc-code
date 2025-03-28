"""Imports necessary modules"""

import random
import sys
import time
from datetime import timedelta
from pathlib import Path

src_path = Path(__file__).resolve().parent
sys.path.insert(0, str(src_path))

from base import (
    create_knowledge_base,
    create_working_memory,
    get_data_training,
    split_knowledge_base,
)
from decision_tree import train_tree
from gng import GrowingNeuralGas
from params import update_limits
from utils import (
    aux_folders,
    aux_folders_tree,
    export_clustered_data,
    export_knowledge_base_csv,
    export_working_memory_csv,
)


def run_model(seed, size, reps):
    """Runs train_network a specified number of times"""

    start_time = time.time()

    for i in range(1, reps + 1):
        print(f"Rep {i} of {reps} for train model")
        print("-" * 100 + "\n")
        train_network(seed, size, i, reps)
        print(f"Completed rep {i} of {reps}")
        print("-" * 100 + "\n")

    end_time = time.time()

    total_time = end_time - start_time
    total = str(timedelta(seconds=round(total_time)))
    total_execution_time = total.split(":")

    """Indicates completion of the training process"""
    print(
        "\nTraining all done in ~"
        + total_execution_time[0]
        + "h:"
        + total_execution_time[1]
        + "m:"
        + total_execution_time[2]
        + "s.\n"
    )


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
    selected_instances = random.sample(working_memory, 100)

    for i, instance in enumerate(selected_instances, 1):
        aux_folders(seed, rep, reps, i)

        print(f"Iteration {i} of 100")
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
    knowledge_base_file = export_knowledge_base_csv(knowledge_base, seed, rep, reps)
    print("Done.\n")

    print("\nTraining all done.\n")

    print("Starting tree training...\n")
    tree_path = aux_folders_tree(seed, rep, reps)
    rules = train_tree(rep, reps, seed, knowledge_base_file, tree_path)
    for r in rules:
        print(r)
    print("Done.\n")

    print("Updating limits...\n")
    limits_to_update = split_knowledge_base(knowledge_base, rep, reps, seed)
    update_limits(limits_to_update)
    print("Done.\n")


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
