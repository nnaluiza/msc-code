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
    get_formatted_time,
    get_knowledge_base_file,
)


def run_model(seed, size, reps, distance_metric, dataset_name, m):
    """Runs train_network a specified number of times"""
    random.seed(seed)

    start_time = time.time()

    for i in range(1, reps + 1):
        print(f"Rep {i} of {reps} for train model")
        print("-" * 100 + "\n")
        train_network(seed, size, i, reps, distance_metric, dataset_name, m)
        print(f"Completed rep {i} of {reps}")
        print("-" * 100 + "\n")

    end_time = time.time()
    total_time = get_formatted_time(start_time, end_time)
    print(f"\nTraining all done in ~{total_time}\n")


def train_network(seed, size, rep, reps, distance_metric, dataset_name, m):
    """Trains the network for a single repetition."""
    rep_seed = seed + rep
    # rep_seed = seed

    print("Generating working memory...")
    start_time_WM = time.time()
    files = aux_folders_limits(dataset_name, seed, rep, reps, distance_metric)
    working_memory = create_working_memory(rep_seed, size, files["limits_file"], distance_metric)
    if not working_memory:
        raise ValueError(f"create_working_memory returned an empty or None working_memory for rep {rep}")
    export_working_memory_csv(dataset_name, working_memory, seed, rep, reps, distance_metric)
    end_time_WM = time.time()
    total_time_WM = get_formatted_time(start_time_WM, end_time_WM)
    print(f"\nDone in ~{total_time_WM}.\n")

    print("Gathering data...")
    start_time_GD = time.time()
    data, true_labels = get_data_training(dataset_name)
    end_time_GD = time.time()
    total_time_GD = get_formatted_time(start_time_GD, end_time_GD)
    print(f"\nDone in ~{total_time_GD}.\n")

    print("Starting training...\n")

    discarded_sets = 0
    knowledge_base_file = get_knowledge_base_file(dataset_name, seed, rep, reps, distance_metric)
    knowledge_base_entries = []

    for i, instance in enumerate(working_memory, 1):
        base_dir = aux_folders(dataset_name, seed, rep, reps, i, distance_metric)

        print(f"Iteration {i} of {len(working_memory)}")
        values = list(instance.values())
        print(f"Chosen instance: {instance}")

        print("Fitting neural network...\n")

        gng = GrowingNeuralGas(base_dir, data, seed, rep, reps, i, distance_metric)
        start, end = gng.fit_network(
            e_b=values[0], e_n=values[1], a_max=values[2], l=values[3], a=values[4], d=values[5], passes=values[6]
        )

        export_clustered_data(dataset_name, gng.cluster_data(), seed, rep, reps, i, distance_metric)
        gng.plot_clusters(gng.cluster_data())

        global_error = gng.compute_global_error()
        num_clusters = gng.number_of_clusters()

        if num_clusters > 1:
            print("\nFound %d clusters.\n" % num_clusters)
            knowledge_entry = create_knowledge_base(
                gng.cluster_data(), instance, start, end, global_error, num_clusters, i, true_labels
            )
            knowledge_base_entries.append(knowledge_entry)
        else:
            print("\nFound %d cluster.\n" % num_clusters)
            print("Only one cluster found. The training and the parameter set used will be disregarded.\n")
            discarded_sets += 1

        print("-" * 100)

    if knowledge_base_entries:
        sorted_entries = classify_knowledge_base(knowledge_base_entries, rep, reps)
        if sorted_entries:
            export_knowledge_base_csv(
                knowledge_base_file, dataset_name, distance_metric, sorted_entries, seed, rep, reps, discarded_sets, m
            )
        else:
            print("Warning: No sorted entries from classify_knowledge_base, skipping export")
    else:
        print("Warning: No knowledge base entries generated, skipping classification and export")

    print("\nTraining all done.\n")

    print("Starting tree training...\n")
    start_time_TT = time.time()
    tree_path = aux_folders_tree(dataset_name, seed, reps, distance_metric)
    rules = train_tree(dataset_name, distance_metric, rep, reps, seed, knowledge_base_file, tree_path)
    for r in rules:
        print(r)
    end_time_TT = time.time()
    total_time_TT = get_formatted_time(start_time_TT, end_time_TT)
    print(f"\nDone in ~{total_time_TT}.\n")

    print("Updating limits...\n")
    start_time_UL = time.time()
    limits_to_update = split_knowledge_base(rules, knowledge_base_file, files["limits_file"], distance_metric)
    update_limits(limits_to_update, files["updated_limits_file"])
    end_time_UL = time.time()
    total_time_UL = get_formatted_time(start_time_UL, end_time_UL)
    print(f"\nDone in ~{total_time_UL}.\n")


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

    run_model(seed, size, reps, distance_metric, dataset_name, size)


if __name__ == "__main__":
    main(sys.argv[1:])
