"""Imports necessary modules"""

import csv
import io
import os

import pandas as pd
from tabulate import tabulate


def aux_folders_limits(dataset_name, seed, rep, reps, distance):
    """Creates the necessary folders to store the limits if they don't already exist"""
    os.makedirs("logs/limits", exist_ok=True)
    dir_path = f"logs/limits/{dataset_name}/seed-{seed}_reps-{reps}_{distance}"
    os.makedirs(dir_path, exist_ok=True)
    file_limits_name = f"{dir_path}/limits_rep{rep}.json"
    file_updated_limits_name = f"{dir_path}/updated_limits_rep{rep}_{distance}.json"
    return {"limits_file": file_limits_name, "updated_limits_file": file_updated_limits_name}


def aux_folders(dataset_name, seed, rep, reps, i, distance):
    """Creates the necessary folders for visualization if they don't already exist"""
    os.makedirs("logs/visualization", exist_ok=True)
    dir_path = f"logs/visualization/{dataset_name}/seed-{seed}_reps-{reps}_{distance}"
    os.makedirs(dir_path, exist_ok=True)
    dir_path_ = f"{dir_path}/rep{rep}"
    os.makedirs(dir_path_, exist_ok=True)
    dir_path__ = f"{dir_path_}/i{i}"
    os.makedirs(dir_path__, exist_ok=True)
    dir_path___ = f"{dir_path__}/sequence"
    os.makedirs(dir_path___, exist_ok=True)
    return dir_path__


def show_working_memory(data):
    """Displays the working memory in a tabulated format"""

    dataframe = pd.DataFrame(data)
    print("Working memory:")
    table = tabulate(dataframe, headers="keys", tablefmt="pretty", showindex=False)
    print(table)
    print("")


def export_working_memory_csv(dataset_name, data, seed, rep, reps, distance):
    """Exports the working memory to a CSV file"""
    os.makedirs("logs/working_memory", exist_ok=True)
    dir_path = f"logs/working_memory/{dataset_name}/seed-{seed}_reps-{reps}_{distance}"
    os.makedirs(dir_path, exist_ok=True)
    file_name = f"{dir_path}/working_memory_rep{rep}.csv"

    with open(f"{file_name}", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([f"# Working Memory - Execution: {rep} of {reps}"])
        writer.writerow(["# Seed: {}".format(seed)])
        writer.writerow([f"# Dataset: {dataset_name} - Distance: {distance}"])
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


def get_knowledge_base_file(dataset_name, seed, rep, reps, distance):
    """Creates the necessary folders for the knowledge bases if they don't already exist"""
    os.makedirs("logs/knowledge_base", exist_ok=True)
    dir_path = f"logs/knowledge_base/{dataset_name}/seed-{seed}_reps-{reps}_{distance}"
    os.makedirs(dir_path, exist_ok=True)
    file_name = f"{dir_path}/knowledge_base_rep{rep}.csv"

    return file_name


def export_knowledge_base_csv(
    file_name, dataset_name, distance, data, seed, rep, reps, discarded_sets, working_memory_size, append=False
):
    """Exports the knowledge data to a CSV file"""

    mode = "a" if append else "w"
    with open(file_name, mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        if not append:
            writer.writerow([f"# Knowledge Base - Execution: {rep} of {reps}"])
            writer.writerow([f"# Working Memory generated sets: {working_memory_size} - Discarded sets: {discarded_sets}"])
            writer.writerow([f"# Seed: {seed}"])
            writer.writerow([f"# Dataset: {dataset_name} - Distance: {distance}"])
            writer.writerow(
                [
                    "rep_number",
                    "e_b",
                    "e_n",
                    "a_max",
                    "l",
                    "a",
                    "d",
                    "passes",
                    "clusters_number",
                    # "silhouette_avg",
                    # "davies_bouldin_index",
                    # "calinski_harabasz_index",
                    "adjusted_rand_index",
                    # "rand_index",
                    "dunn_index",
                    "global_error",
                    # "normalized_global_error",
                    "execution_time",
                    "objective_function",
                    "class",
                ]
            )

        if isinstance(data, list):
            for instance in data:
                writer.writerow(
                    [
                        instance["rep_number"],
                        instance["e_b"],
                        instance["e_n"],
                        instance["a_max"],
                        instance["l"],
                        instance["a"],
                        instance["d"],
                        instance["passes"],
                        instance["clusters_number"],
                        # instance["silhouette_avg"],
                        # instance["davies_bouldin_index"],
                        # instance["calinski_harabasz_index"],
                        instance["adjusted_rand_index"],
                        # instance["rand_index"],
                        instance["dunn_index"],
                        instance["global_error"],
                        # instance["normalized_global_error"],
                        instance["execution_time"],
                        instance["objective_function"],
                        instance["class"],
                    ]
                )
        else:
            writer.writerow(
                [
                    data["rep_number"],
                    data["e_b"],
                    data["e_n"],
                    data["a_max"],
                    data["l"],
                    data["a"],
                    data["d"],
                    data["passes"],
                    data["clusters_number"],
                    # data["silhouette_avg"],
                    # data["davies_bouldin_index"],
                    # data["calinski_harabasz_index"],
                    data["adjusted_rand_index"],
                    # data["rand_index"],
                    data["dunn_index"],
                    data["global_error"],
                    # data["normalized_global_error"],
                    data["execution_time"],
                    data["objective_function"],
                    data["class"],
                ]
            )


def export_clustered_data(dataset_name, data, seed, rep, reps, i, distance):
    """Exports the clustered data to a text file, which is solely used for comparison of the results."""
    os.makedirs("logs/clusters", exist_ok=True)
    dir_path = f"logs/clusters/{dataset_name}/seed-{seed}_reps-{reps}_{distance}"
    os.makedirs(dir_path, exist_ok=True)
    dir_path_ = f"{dir_path}/rep{rep}"
    os.makedirs(dir_path_, exist_ok=True)
    file_name = f"{dir_path_}/clustered_data_rep{rep}_i{i}.csv"

    file = io.open(f"{file_name}", "w")
    file.write(str(data))
    file.close()


def aux_folders_rules(dataset_name, seed, rep, reps, distance):
    """Creates the necessary folders for the rules if they don't already exist"""
    os.makedirs("logs/rules", exist_ok=True)
    dir_path = f"logs/rules/{dataset_name}/seed-{seed}_reps-{reps}_{distance}"
    os.makedirs(dir_path, exist_ok=True)
    file_name = f"{dir_path}/extracted_rules_rep{rep}.txt"
    return file_name


def aux_folders_tree(dataset_name, seed, reps, distance):
    """Creates the necessary folders for the tree images if they don't already exist"""
    os.makedirs("logs/tree", exist_ok=True)
    dir_path = f"logs/tree/{dataset_name}/seed-{seed}_reps-{reps}_{distance}"
    os.makedirs(dir_path, exist_ok=True)

    return dir_path


def get_formatted_time(start_time, end_time):
    total_time = end_time - start_time
    total_seconds = int(total_time)
    milliseconds = int((total_time % 1) * 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours}h:{minutes:02d}m:{seconds:02d}s.{milliseconds:03d}ms."
