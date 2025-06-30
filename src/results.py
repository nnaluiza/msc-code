import csv
import glob
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def get_knowledge_base_files(dataset_name: str, seed: int, reps: int, distance_metric: str) -> List[str]:
    """Returns a list of all knowledge base CSV files for a given dataset, seed, reps, and distance metric."""
    dir_path = f"logs/knowledge_base/{dataset_name}/seed-{seed}_reps-{reps}_{distance_metric.lower()}"
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return []

    file_pattern = os.path.join(dir_path, "knowledge_base_rep*.csv")
    files = glob.glob(file_pattern)

    if not files:
        print(f"No knowledge base files found in: {dir_path}")
        return []

    return sorted(files)


def get_limits_files(dataset_name: str, seed: int, reps: int) -> Dict[str, List[str]]:
    """Returns lists of limits and updated limits JSON files for a given dataset, seed, and reps."""
    dir_path = f"logs/limits/{dataset_name}/seed-{seed}_reps-{reps}"
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return {"limits_files": [], "updated_limits_files": []}

    limits_pattern = os.path.join(dir_path, "limits_rep*.json")
    updated_limits_pattern = os.path.join(dir_path, "updated_limits_rep*.json")

    limits_files = glob.glob(limits_pattern)
    updated_limits_files = glob.glob(updated_limits_pattern)

    if not limits_files and not updated_limits_files:
        print(f"No limits files found in: {dir_path}")

    return {"limits_files": sorted(limits_files), "updated_limits_files": sorted(updated_limits_files)}


def read_knowledge_base_file(file_path: str) -> tuple[List[Dict[str, Any]], int]:
    """Reads a knowledge base CSV file and returns data and discarded sets count."""
    data = []
    discarded_sets = 0
    try:
        with open(file_path, mode="r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            headers = None
            for row in reader:
                if row and row[0].startswith("#"):
                    if "Discarted sets" in row[0]:
                        try:
                            discarded_sets = int(row[0].split("Discarted sets: ")[1])
                        except (IndexError, ValueError):
                            print(f"Warning: Could not parse discarded sets in {file_path}")
                    continue
                if headers is None:
                    headers = row
                    continue
                if len(row) == len(headers):
                    entry = {}
                    for key, value in zip(headers, row):
                        if value:  # Only process non-empty values
                            if key in [
                                "e_b",
                                "e_n",
                                "a_max",
                                "l",
                                "a",
                                "d",
                                "silhouette_avg",
                                "davies_bouldin_index",
                                "calinski_harabasz_index",
                                "adjusted_rand_index",
                                "global_error",
                                "normalized_global_error",
                                "execution_time",
                                "dunn_index",  # Added dunn_index to numeric keys
                            ]:
                                try:
                                    entry[key] = float(value)
                                except ValueError:
                                    print(f"Warning: Non-numeric value '{value}' for key '{key}' in {file_path}, skipping entry")
                                    entry[key] = None
                            elif key in ["passes", "clusters_number"]:
                                try:
                                    entry[key] = int(value)
                                except ValueError:
                                    print(f"Warning: Non-numeric value '{value}' for key '{key}' in {file_path}, skipping entry")
                                    entry[key] = None
                            else:
                                entry[key] = value
                        else:
                            entry[key] = None
                    data.append(entry)
                else:
                    print(f"Warning: Skipping malformed row in {file_path}: {row}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return data, discarded_sets


def read_limits_file(file_path: str) -> Dict[str, List[float]]:
    """Reads a limits JSON file and returns a dictionary mapping parameters to [min, max]."""
    param_names = ["e_b", "e_n", "a_max", "l", "a", "d", "passes"]
    limits = {}
    try:
        with open(file_path, mode="r") as jsonfile:
            data = json.load(jsonfile)
            if len(data) != len(param_names):
                print(f"Warning: Expected {len(param_names)} parameters in {file_path}, got {len(data)}")
                return limits
            for param, (min_val, max_val) in zip(param_names, data):
                limits[param] = [float(min_val), float(max_val)]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return limits


def calculate_statistics(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Calculates global min, max, mean, std, and 95% CI across all knowledge base entries."""
    numerical_cols = [
        "clusters_number",
        "silhouette_avg",
        "davies_bouldin_index",
        "calinski_harabasz_index",
        "adjusted_rand_index",
        "dunn_index",
        "normalized_global_error",
        "execution_time",
        "e_b",
        "e_n",
        "a_max",
        "l",
        "a",
        "d",
        "passes",
        "global_error",
    ]
    statistics = {
        col: {"min": None, "max": None, "mean": None, "std": None, "ci_lower": None, "ci_upper": None} for col in numerical_cols
    }

    for col in numerical_cols:
        values = [entry[col] for entry in data if entry.get(col) is not None and isinstance(entry[col], (int, float))]
        if values:
            statistics[col]["min"] = float(np.min(values))
            statistics[col]["max"] = float(np.max(values))
            statistics[col]["mean"] = float(np.mean(values))
            statistics[col]["std"] = float(np.std(values, ddof=1))
            if len(values) > 1:
                t_critical = stats.t.ppf(0.975, df=len(values) - 1)
                margin_of_error = t_critical * (statistics[col]["std"] / np.sqrt(len(values)))
                statistics[col]["ci_lower"] = float(statistics[col]["mean"] - margin_of_error)
                statistics[col]["ci_upper"] = float(statistics[col]["mean"] + margin_of_error)

    return statistics


def export_statistics_csv(
    dataset_name: str,
    seed: int,
    reps: int,
    stats_by_distance: Dict[str, Dict[str, Dict[str, float]]],
):
    """Exports statistics to a CSV file, differentiating by distance metric, formatted to 4 decimals."""
    dir_path = f"logs/results/{dataset_name}/seed-{seed}_reps-{reps}"
    os.makedirs(dir_path, exist_ok=True)
    file_path = f"{dir_path}/statistics.csv"

    headers = [
        "Distance Metric",
        "Statistic",
        "clusters_number",
        "silhouette_avg",
        "davies_bouldin_index",
        "calinski_harabasz_index",
        "adjusted_rand_index",
        "dunn_index",
        "normalized_global_error",
        "execution_time",
    ]

    with open(file_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Add a comment row to guide Google Sheets usage
        # writer.writerow([
        #     "# Note: To ensure correct decimal interpretation in Google Sheets, set the spreadsheet locale to 'United States' (or another locale that uses '.' for decimals).",
        # ])
        writer.writerow(headers)

        distance_metrics = ["euclidean", "cosine", "cityblock"]
        for distance_metric in distance_metrics:
            if distance_metric not in stats_by_distance:
                continue  # Skip if no data for this distance metric
            stats = stats_by_distance[distance_metric]

            # Min row
            min_row = [distance_metric, "min"] + [
                f"{float(stats.get(col, {}).get('min', 0)):.4f}" if stats.get(col, {}).get("min") is not None else ""
                for col in headers[2:]
            ]
            writer.writerow(min_row)

            # Max row
            max_row = [distance_metric, "max"] + [
                f"{float(stats.get(col, {}).get('max', 0)):.4f}" if stats.get(col, {}).get("max") is not None else ""
                for col in headers[2:]
            ]
            writer.writerow(max_row)

            # Mean row
            mean_row = [distance_metric, "mean"] + [
                f"{float(stats.get(col, {}).get('mean', 0)):.4f}" if stats.get(col, {}).get("mean") is not None else ""
                for col in headers[2:]
            ]
            writer.writerow(mean_row)

            # Std row
            std_row = [distance_metric, "std"] + [
                f"{float(stats.get(col, {}).get('std', 0)):.4f}" if stats.get(col, {}).get("std") is not None else ""
                for col in headers[2:]
            ]
            writer.writerow(std_row)

            # CI lower row
            ci_lower_row = [distance_metric, "ci_lower"] + [
                f"{float(stats.get(col, {}).get('ci_lower', 0)):.4f}" if stats.get(col, {}).get("ci_lower") is not None else ""
                for col in headers[2:]
            ]
            writer.writerow(ci_lower_row)

            # CI upper row
            ci_upper_row = [distance_metric, "ci_upper"] + [
                f"{float(stats.get(col, {}).get('ci_upper', 0)):.4f}" if stats.get(col, {}).get("ci_upper") is not None else ""
                for col in headers[2:]
            ]
            writer.writerow(ci_upper_row)

    print(f"Statistics exported to: {file_path}")


def create_boxplots(dataset_name: str, seed: int, reps: int, kb_data: List[Dict[str, Any]], distance_metric: str):
    """Creates a boxplot for each parameter showing variation across iterations for a specific distance metric."""
    if not kb_data:
        print(f"No knowledge base data available for boxplots for {distance_metric}")
        return

    params = ["e_b", "e_n", "a_max", "l", "a", "d", "passes"]
    output_dir = f"logs/results/{dataset_name}/seed-{seed}_reps-{reps}/{distance_metric}"
    os.makedirs(output_dir, exist_ok=True)

    # Group data by repetition (knowledge base file)
    rep_data = {}
    for entry in kb_data:
        rep = entry["rep"]
        if rep not in rep_data:
            rep_data[rep] = {param: [] for param in params}
        for param in params:
            if entry.get(param) is not None and isinstance(entry[param], (int, float)):
                rep_data[rep][param].append(entry[param])

    # Get unique repetitions
    rep_numbers = sorted(rep_data.keys())
    if not rep_numbers:
        print(f"No repetition data available for boxplots for {distance_metric}")
        return

    # Create one figure per parameter
    for param in params:
        # Collect data for each repetition
        all_data = [rep_data[rep][param] for rep in rep_numbers if rep_data[rep][param]]

        if not all_data:  # Skip if no data for this parameter
            print(f"No data for parameter {param} in {distance_metric}, skipping boxplot")
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Generate colors using a colormap
        cmap = plt.get_cmap("tab10")  # Use a colormap with 10 distinct colors
        colors = [cmap(i / len(rep_numbers)) for i in range(len(rep_numbers))]

        # Create boxplots with different colors
        box = ax.boxplot(all_data, patch_artist=True, labels=[f"Rep {r}" for r in rep_numbers])
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)

        # Find the maximum whisker value across all boxplots to set y-axis limit
        max_whisker = 0
        for i in range(len(all_data)):
            if all_data[i]:  # Skip empty data
                whiskers = box["whiskers"][i * 2 : i * 2 + 2]
                whisker_max = whiskers[1].get_ydata()[1]  # Upper whisker
                max_whisker = max(max_whisker, whisker_max)

        # Add labels for Q1 and Q3
        for i, data in enumerate(all_data):
            if not data:  # Skip empty data
                continue
            # Calculate statistics
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            # Get whiskers (min and max, excluding outliers)
            whiskers = box["whiskers"][i * 2 : i * 2 + 2]
            whisker_min = whiskers[0].get_ydata()[1]  # Lower whisker
            whisker_max = whiskers[1].get_ydata()[1]  # Upper whisker

            # Position for the boxplot (1-based index for plotting)
            x_pos = i + 1

            # Add Q1 label below the boxplot
            ax.text(
                x_pos,
                whisker_min - 0.05 * (max_whisker - whisker_min),
                f"Q1: {q1:.4f}",
                ha="center",
                va="top",
                fontsize=8,
                color="gray",
            )

            # Add Q3 label above the boxplot
            ax.text(
                x_pos,
                whisker_max + 0.05 * (max_whisker - whisker_min),
                f"Q3: {q3:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="gray",
            )

        # Set y-axis to start at 0 with padding for Q3 labels
        y_max = max_whisker * 1.1  # Add 10% padding above the highest whisker
        ax.set_ylim(0, y_max)  # Start y-axis at 0

        # Customize plot
        ax.set_title(
            f"{param} Variation Across Iterations (Seed: {seed}, Reps: {reps}, Distance: {distance_metric.capitalize()})"
        )
        ax.set_xlabel("Repetition")
        ax.set_ylabel(f"{param} Values")
        ax.grid(True, linestyle="--", alpha=0.7)

        # Adjust layout and save
        plt.tight_layout()
        output_path = f"{output_dir}/{param}_variation_across_iterations.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Boxplot for {param} ({distance_metric}) saved to: {output_path}")


def create_summary_table(dataset_name: str, seed: int, reps: int, kb_data_by_distance: Dict[str, List[Dict[str, Any]]]):
    """Creates a LaTeX table summarizing statistics across all knowledge bases for all distance measures."""
    if not kb_data_by_distance:
        print("No knowledge base data available for summary table")
        return

    # Define attributes to summarize
    attributes = [
        "clusters_number",
        "silhouette_avg",
        "davies_bouldin_index",
        "calinski_harabasz_index",
        "adjusted_rand_index",
        "dunn_index",
        "normalized_global_error",
        "execution_time",
    ]

    # Calculate statistics for each distance measure
    stats_by_distance = {}
    for distance_metric, kb_data in kb_data_by_distance.items():
        stats = {attr: {"min": None, "max": None, "median": None, "std": None} for attr in attributes}
        for attr in attributes:
            values = [entry[attr] for entry in kb_data if entry.get(attr) is not None and isinstance(entry[attr], (int, float))]
            if values:
                stats[attr]["min"] = float(np.min(values))
                stats[attr]["max"] = float(np.max(values))
                stats[attr]["median"] = float(np.median(values))
                stats[attr]["std"] = float(np.std(values, ddof=1))
        stats_by_distance[distance_metric] = stats

    # Generate LaTeX table
    output_dir = f"logs/results/{dataset_name}/seed-{seed}_reps-{reps}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/summary_table.tex"

    # Writing LaTeX table
    with open(output_path, "w") as f:
        # Setting up the LaTeX document
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{multirow}\n")
        f.write("\\usepackage{geometry}\n")
        f.write("\\geometry{a4paper, margin=0.5in}\n")  # Reduced margin for wider table
        f.write("\\begin{document}\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Summary Statistics for Knowledge Base Attributes (Dataset: "
            + dataset_name
            + ", Seed: "
            + str(seed)
            + ", Reps: "
            + str(reps)
            + ")}\n"
        )
        f.write("\\label{tab:summary_stats}\n")

        # Calculate the number of columns: 1 for Dataset/Distancia, and 4 columns (Min, Med, Max, DP) per attribute
        num_columns = 1 + len(attributes) * 4  # 4 subcolumns per attribute (Min, Med, Max, DP)
        f.write(f"\\begin{{tabular}}{{l{'c' * (num_columns - 1)}}}\n")
        f.write("\\toprule\n")

        # First header row: Dataset and merged attribute names
        f.write(
            " & ".join(
                ["\\multirow{2}{*}{\\textbf{Dataset}}"]
                + [f"\\multicolumn{{4}}{{c}}{{\\textbf{{{attr.replace('_', ' ').title()}}}}}" for attr in attributes]
            )
            + " \\\\\n"
        )

        # Second header row: Subheaders (Min, Med, Max, DP) for each attribute
        subheaders = [""]  # Empty cell under Dataset
        for attr in attributes:
            subheaders.extend(["Min", "Med", "Max", "DP"])
        f.write(" & ".join(subheaders) + " \\\\\n")
        f.write("\\midrule\n")

        # Data rows for each distance measure
        distance_metrics = ["euclidean", "cosine", "cityblock"]
        for distance_metric in distance_metrics:
            if distance_metric not in stats_by_distance:
                continue  # Skip if no data for this distance metric
            stats = stats_by_distance[distance_metric]
            row = [f"{dataset_name} ({distance_metric.capitalize()})"]
            for attr in attributes:
                row.extend(
                    [
                        f"{stats[attr]['min']:.4f}" if stats[attr]["min"] is not None else "-",
                        f"{stats[attr]['median']:.4f}" if stats[attr]["median"] is not None else "-",
                        f"{stats[attr]['max']:.4f}" if stats[attr]["max"] is not None else "-",
                        f"{stats[attr]['std']:.4f}" if stats[attr]["std"] is not None else "-",
                    ]
                )
            f.write(" & ".join(row) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
        f.write("\\end{document}\n")

    print(f"Summary table saved to: {output_path}")


def read_all_data(dataset_name: str, seed: int, reps: int, distance_metric: str) -> Dict[str, Any]:
    """Reads all knowledge base CSV and limits JSON files, returning combined data."""
    kb_files = get_knowledge_base_files(dataset_name, seed, reps, distance_metric)
    all_kb_data = []
    total_discarded = 0

    for file_path in kb_files:
        print(f"Reading knowledge base file: {file_path}")
        file_data, discarded = read_knowledge_base_file(file_path)
        if file_data:
            rep_number = int(file_path.split("rep")[-1].split(".")[0])
            for entry in file_data:
                entry["rep"] = rep_number
            all_kb_data.extend(file_data)
            total_discarded += discarded
        else:
            print(f"No data read from {file_path}")

    limits_info = get_limits_files(dataset_name, seed, reps)
    limits_data = {}

    for file_path in limits_info["limits_files"]:
        print(f"Reading limits file: {file_path}")
        rep_number = int(file_path.split("rep")[-1].replace(".json", ""))
        limits = read_limits_file(file_path)
        if limits:
            limits_data[rep_number] = {"limits": limits}
        else:
            print(f"No limits data read from {file_path}")

    for file_path in limits_info["updated_limits_files"]:
        print(f"Reading updated limits file: {file_path}")
        rep_number = int(file_path.split("rep")[-1].replace(".json", ""))
        limits = read_limits_file(file_path)
        if limits:
            if rep_number not in limits_data:
                limits_data[rep_number] = {}
            limits_data[rep_number]["updated_limits"] = limits
        else:
            print(f"No updated limits data read from {file_path}")

    return {"knowledge_base": all_kb_data, "limits": limits_data, "total_discarded": total_discarded, "total_reps": len(kb_files)}


def main(dataset_name: str, seed: int, reps: int):
    """Main function to read, process, and export statistics, boxplots, and summary table for all distance measures."""
    distance_metrics = ["euclidean", "cosine", "cityblock"]

    # Dictionary to store knowledge base data and statistics for each distance metric
    kb_data_by_distance = {}
    stats_by_distance = {}

    for distance_metric in distance_metrics:
        print(f"Processing data for dataset: {dataset_name}, seed: {seed}, reps: {reps}, distance metric: {distance_metric}")
        all_data = read_all_data(dataset_name, seed, reps, distance_metric)

        kb_data = all_data["knowledge_base"]
        total_reps = all_data["total_reps"]
        total_discarded = all_data["total_discarded"]

        if kb_data:
            print(f"Read {len(kb_data)} total knowledge base entries from {total_reps} files for {distance_metric}")
            print(f"Total sets discarded: {total_discarded}")
            kb_data_by_distance[distance_metric] = kb_data

            # Calculate statistics for this distance metric
            stats = calculate_statistics(kb_data)
            stats_by_distance[distance_metric] = stats

            # Create boxplots for this distance metric
            create_boxplots(dataset_name, seed, reps, kb_data, distance_metric)
        else:
            print(f"No knowledge base data found for {distance_metric}.")

        limits_data = all_data["limits"]
        if limits_data:
            print(f"Limits data found for {distance_metric}, but not used for iteration-based boxplots or summary table.")
        else:
            print(f"No limits data found for {distance_metric}.")

    # Export statistics for all distance metrics to a single CSV
    if stats_by_distance:
        export_statistics_csv(dataset_name, seed, reps, stats_by_distance)
    else:
        print("No statistics data found for any distance metric.")

    # Create a single summary table for all distance measures
    if kb_data_by_distance:
        create_summary_table(dataset_name, seed, reps, kb_data_by_distance)
    else:
        print("No knowledge base data found for any distance metric.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python results.py <dataset_name> <seed> <reps>")
        print("Example: python results.py iris 1 4")
        sys.exit(1)

    dataset_name = sys.argv[1]
    seed = int(sys.argv[2])
    reps = int(sys.argv[3])
    main(dataset_name, seed, reps)
