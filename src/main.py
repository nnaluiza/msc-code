"""Imports necessary modules"""

import random
import sys
import time
from datetime import timedelta

from base import (
    classify_knowledge_base,
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
    aux_folders_limits,
    aux_folders_tree,
    export_clustered_data,
    export_knowledge_base_csv,
    export_working_memory_csv,
)


def run_model(seed, size, reps, distance_metric, dataset_name):
    """Runs train_network a specified number of times"""
    random.seed(seed)

    start_time = time.time()

    for i in range(1, reps + 1):
        print(f"Rep {i} of {reps} for train model")
        print("-" * 100 + "\n")
        train_network(seed, size, i, reps, distance_metric, dataset_name)
        print(f"Completed rep {i} of {reps}")
        print("-" * 100 + "\n")

    end_time = time.time()

    total_time = end_time - start_time
    total = str(timedelta(seconds=round(total_time)))
    total_execution_time = total.split(":")

    print(
        "\nTraining all done in ~"
        + total_execution_time[0]
        + "h:"
        + total_execution_time[1]
        + "m:"
        + total_execution_time[2]
        + "s.\n"
    )


def train_network(seed, size, rep, reps, distance_metric, dataset_name):
    """Trains the network for a single repetition."""
    rep_seed = seed + rep

    print("Generating working memory...")
    files = aux_folders_limits(dataset_name, seed, rep, reps)
    working_memory = create_working_memory(rep_seed, size, files["limits_file"])
    if not working_memory:
        raise ValueError(f"create_working_memory returned an empty or None working_memory for rep {rep}")
    print(f"Working memory size: {len(working_memory)}")
    export_working_memory_csv(dataset_name, working_memory, seed, rep, reps)
    print("Done.\n")

    print("Gathering data...")
    data, true_labels = get_data_training(dataset_name)
    print("Done.\n")

    print("Starting training...\n")

    knowledge_base_file = export_knowledge_base_csv(dataset_name, [], seed, rep, reps, append=False)
    knowledge_base_entries = []

    for i, instance in enumerate(working_memory, 1):
        base_dir = aux_folders(dataset_name, seed, rep, reps, i)

        print(f"Iteration {i} of {len(working_memory)}")
        values = list(instance.values())
        print(f"Chosen instance: {instance}")

        print("Fitting neural network...\n")
        start = time.time()

        gng = GrowingNeuralGas(base_dir, data, seed, rep, reps, i, distance_metric=distance_metric)
        gng.fit_network(e_b=values[0], e_n=values[1], a_max=values[2], l=values[3], a=values[4], d=values[5], passes=values[6])
        export_clustered_data(dataset_name, gng.cluster_data(), seed, rep, reps, i)

        end = time.time()

        global_error = gng.compute_global_error()
        num_clusters = gng.number_of_clusters()

        if num_clusters > 1:
            print("\nFound %d clusters.\n" % num_clusters)
            knowledge_entry = create_knowledge_base(
                gng.cluster_data(), instance, start, end, global_error, num_clusters, true_labels
            )
            knowledge_base_entries.append(knowledge_entry)
        else:
            print("\nFound %d cluster.\n" % num_clusters)
            print("Only one cluster found. The training and the parameter set used will be disregarded.\n")

        print("-" * 100)

    if knowledge_base_entries:
        sorted_entries = classify_knowledge_base(knowledge_base_entries, rep, reps)
        if sorted_entries:
            export_knowledge_base_csv(dataset_name, sorted_entries, seed, rep, reps, append=False)
        else:
            print("Warning: No sorted entries from classify_knowledge_base, skipping export")
    else:
        print("Warning: No knowledge base entries generated, skipping classification and export")

    print("\nTraining all done.\n")

    print("Starting tree training...\n")
    tree_path = aux_folders_tree(dataset_name, seed, rep, reps)
    rules = train_tree(dataset_name, rep, reps, seed, knowledge_base_file, tree_path)
    for r in rules:
        print(r)
    print("Done.\n")

    print("Updating limits...\n")
    limits_to_update = split_knowledge_base(rules, knowledge_base_file, files["limits_file"])
    update_limits(limits_to_update, files["updated_limits_file"])
    print("Done.\n")


def main(params):
    """Parses command-line parameters and initiates the model training process."""
    print(f"Received parameters: {params}")
    if len(params) != 5:
        print("Error: Expected 5 parameters (seed, size, reps, distance_metric, dataset_name)")
        print("Usage: python run.py <seed> <size> <reps> <distance_metric> <dataset_name>")
        print("Example: python run.py 42 1000 5 euclidean iris")
        return

    seed = int(params[0])
    size = int(params[1])
    reps = int(params[2])
    distance_metric = params[3]
    dataset_name = params[4]

    run_model(seed, size, reps, distance_metric, dataset_name)


if __name__ == "__main__":
    main(sys.argv[1:])
