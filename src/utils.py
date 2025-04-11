"""Imports necessary modules"""

import csv
import io
import os

import pandas as pd
from tabulate import tabulate


def aux_folders_limits(seed, rep, reps):
    """Creates the necessary folders to store the limits if they don't already exist"""
    if not os.path.exists("logs/limits"):
        os.makedirs("logs/limits")

    dir_path = f"logs/limits/seed-{seed}_reps-{reps}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_limits_name = f"{dir_path}/limits_rep{rep}.json"
    file_updated_limits_name = f"{dir_path}/updated_limits_rep{rep}.json"

    return {"limits_file": file_limits_name, "updated_limits_file": file_updated_limits_name}


def aux_folders(seed, rep, reps, i):
    """Creates the necessary folders for visualization if they don't already exist"""
    if not os.path.exists("logs/visualization"):
        os.makedirs("logs/visualization")

    dir_path = f"logs/visualization/seed-{seed}_reps-{reps}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path_ = f"{dir_path}/rep{rep}"
    if not os.path.exists(dir_path_):
        os.makedirs(dir_path_)

    dir_path__ = f"{dir_path_}/i{i}"
    if not os.path.exists(dir_path__):
        os.makedirs(dir_path__)

    dir_path___ = f"{dir_path__}/sequence"
    if not os.path.exists(dir_path___):
        os.makedirs(dir_path___)


def show_working_memory(data):
    """Displays the working memory in a tabulated format"""

    dataframe = pd.DataFrame(data)
    print("Working memory:")
    table = tabulate(dataframe, headers="keys", tablefmt="pretty", showindex=False)
    print(table)
    print("")


def export_working_memory_csv(data, seed, rep, reps):
    """Exports the working memory to a CSV file"""

    if not os.path.exists("logs/working_memory"):
        os.makedirs("logs/working_memory")

    dir_path = f"logs/working_memory/seed-{seed}_reps-{reps}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_name = f"{dir_path}/working_memory_rep{rep}.csv"

    with open(f"{file_name}", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([f"# Working Memory - Execution: {rep} of {reps}"])
        writer.writerow(["# Seed: {}".format(seed)])
        writer.writerow(["e_b", "e_n", "a_max", "l", "a", "d", "passes"])

        for instance in data:
            writer.writerow(
                [
                    instance["e_b"],
                    instance["e_n"],
                    instance["a_max"],
                    instance["l"],
                    instance["a"],
                    instance["d"],
                    instance["passes"],
                ]
            )


def export_knowledge_base_csv(data, seed, rep, reps, append=False):
    """Exports the knowledge data to a CSV file"""
    if not os.path.exists("logs/knowledge_base"):
        os.makedirs("logs/knowledge_base")

    dir_path = f"logs/knowledge_base/seed-{seed}_reps-{reps}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_name = f"{dir_path}/knowledge_base_rep{rep}.csv"

    mode = "a" if append else "w"
    with open(file_name, mode, newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        if not append:
            writer.writerow([f"# Knowledge Base - Execution: {rep} of {reps}"])
            writer.writerow(["# Seed: {}".format(seed)])
            writer.writerow(
                [
                    "e_b",
                    "e_n",
                    "a_max",
                    "l",
                    "a",
                    "d",
                    "passes",
                    "clusters_number",
                    "silhouette_avg",
                    "davies_bouldin_index",
                    "calinski_harabasz_index",
                    "adjusted_rand_index",
                    "global_error",
                    "combined_score",
                    "execution_time",
                    "class",
                ]
            )

        if isinstance(data, list):
            for instance in data:
                writer.writerow(
                    [
                        instance["e_b"],
                        instance["e_n"],
                        instance["a_max"],
                        instance["l"],
                        instance["a"],
                        instance["d"],
                        instance["passes"],
                        instance["clusters_number"],
                        instance["silhouette_avg"],
                        instance["davies_bouldin_index"],
                        instance["calinski_harabasz_index"],
                        instance["adjusted_rand_index"],
                        instance["global_error"],
                        instance["combined_score"],
                        instance["execution_time"],
                        instance["class"],
                    ]
                )
        else:
            writer.writerow(
                [
                    data["e_b"],
                    data["e_n"],
                    data["a_max"],
                    data["l"],
                    data["a"],
                    data["d"],
                    data["passes"],
                    data["clusters_number"],
                    data["silhouette_avg"],
                    data["davies_bouldin_index"],
                    data["calinski_harabasz_index"],
                    data["adjusted_rand_index"],
                    data["global_error"],
                    data["combined_score"],
                    data["execution_time"],
                    data["class"],
                ]
            )

    return file_name


def export_clustered_data(data, seed, rep, reps, i):
    """Exports the clustered data to a text file, which is solely used for comparison of the results."""

    if not os.path.exists("logs/clusters"):
        os.makedirs("logs/clusters")

    dir_path = f"logs/clusters/seed-{seed}_reps-{reps}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path_ = f"{dir_path}/rep{rep}"
    if not os.path.exists(dir_path_):
        os.makedirs(dir_path_)

    file_name = f"{dir_path_}/clustered_data_rep{rep}_i{i}.csv"

    file = io.open(f"{file_name}", "w")
    file.write(str(data))
    file.close()


def aux_folders_rules(seed, rep, reps):
    """Creates the necessary folders for the rules if they don't already exist"""
    if not os.path.exists("logs/rules"):
        os.makedirs("logs/rules")

    dir_path = f"logs/rules/seed-{seed}_reps-{reps}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_name = f"{dir_path}/extracted_rules_rep{rep}.txt"
    return file_name


def aux_folders_tree(seed, rep, reps):
    """Creates the necessary folders for the tree images if they don't already exist"""
    if not os.path.exists("logs/tree"):
        os.makedirs("logs/tree")

    dir_path = f"logs/tree/seed-{seed}_reps-{reps}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path
