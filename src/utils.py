"""Imports necessary modules"""

import csv
import os

import pandas as pd
from tabulate import tabulate


def aux_folders():
    """Creates the necessary folders for visualization if they don't already exist"""

    if not os.path.exists("visualization"):
        os.makedirs("visualization")
        os.makedirs("visualization/sequence")


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
