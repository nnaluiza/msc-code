import os
import csv
import glob
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def get_knowledge_base_files(dataset_name: str, seed: int, reps: int) -> List[str]:
    """Returns a list of all knowledge base CSV files for a given dataset, seed, and reps."""
    dir_path = f"logs/knowledge_base/{dataset_name}/seed-{seed}_reps-{reps}"
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

    return {
        "limits_files": sorted(limits_files),
        "updated_limits_files": sorted(updated_limits_files)
    }

def read_knowledge_base_file(file_path: str) -> tuple[List[Dict[str, Any]], int]:
    """Reads a knowledge base CSV file and returns data and discarded sets count."""
    data = []
    discarded_sets = 0
    try:
        with open(file_path, mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            headers = None
            for row in reader:
                if row and row[0].startswith('#'):
                    if 'Discarted sets' in row[0]:
                        try:
                            discarded_sets = int(row[0].split('Discarted sets: ')[1])
                        except (IndexError, ValueError):
                            print(f"Warning: Could not parse discarded sets in {file_path}")
                    continue
                if headers is None:
                    headers = row
                    continue
                if len(row) == len(headers):
                    entry = {}
                    for key, value in zip(headers, row):
                        if key in ['e_b', 'e_n', 'a_max', 'l', 'a', 'd', 'silhouette_avg',
                                 'davies_bouldin_index', 'calinski_harabasz_index',
                                 'adjusted_rand_index', 'global_error', 'normalized_global_error',
                                 'execution_time']:
                            entry[key] = float(value) if value else None
                        elif key in ['passes', 'clusters_number']:
                            entry[key] = int(value) if value else None
                        else:
                            entry[key] = value if value else None
                    data.append(entry)
                else:
                    print(f"Warning: Skipping malformed row in {file_path}: {row}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return data, discarded_sets

def read_limits_file(file_path: str) -> Dict[str, List[float]]:
    """Reads a limits JSON file and returns a dictionary mapping parameters to [min, max]."""
    param_names = ['e_b', 'e_n', 'a_max', 'l', 'a', 'd', 'passes']
    limits = {}
    try:
        with open(file_path, mode='r') as jsonfile:
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
    numerical_cols = ['e_b', 'e_n', 'a_max', 'l', 'a', 'd', 'passes', 'clusters_number',
                      'silhouette_avg', 'davies_bouldin_index', 'calinski_harabasz_index',
                      'adjusted_rand_index', 'global_error', 'execution_time']
    statistics = {col: {'min': None, 'max': None, 'mean': None, 'std': None, 'ci_lower': None, 'ci_upper': None}
                 for col in numerical_cols}

    for col in numerical_cols:
        values = [entry[col] for entry in data if entry[col] is not None]
        if values:
            statistics[col]['min'] = float(np.min(values))
            statistics[col]['max'] = float(np.max(values))
            statistics[col]['mean'] = float(np.mean(values))
            statistics[col]['std'] = float(np.std(values, ddof=1))
            if len(values) > 1:
                t_critical = stats.t.ppf(0.975, df=len(values)-1)
                margin_of_error = t_critical * (statistics[col]['std'] / np.sqrt(len(values)))
                statistics[col]['ci_lower'] = float(statistics[col]['mean'] - margin_of_error)
                statistics[col]['ci_upper'] = float(statistics[col]['mean'] + margin_of_error)

    return statistics

def export_statistics_csv(dataset_name: str, seed: int, reps: int, data: List[Dict[str, Any]],
                         total_reps: int, total_discarded: int, stats: Dict[str, Dict[str, float]]):
    """Exports statistics to a CSV file with the updated header, formatted to 4 decimals."""
    dir_path = f"logs/results/{dataset_name}/seed-{seed}_reps-{reps}"
    os.makedirs(dir_path, exist_ok=True)
    file_path = f"{dir_path}/statistics.csv"

    headers = ['e_b', 'e_n', 'a_max', 'l', 'a', 'd', 'passes', 'clusters_number',
               'silhouette_avg', 'davies_bouldin_index', 'calinski_harabasz_index',
               'adjusted_rand_index', 'global_error', 'execution_time']

    with open(file_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        min_row = [f"{float(stats.get(col, {}).get('min', 0)):.4f}" if stats.get(col, {}).get('min') is not None else '' for col in headers] + ['']
        writer.writerow(['min'] + min_row[1:])

        max_row = [f"{float(stats.get(col, {}).get('max', 0)):.4f}" if stats.get(col, {}).get('max') is not None else '' for col in headers] + ['']
        writer.writerow(['max'] + max_row[1:])

        mean_row = [f"{float(stats.get(col, {}).get('mean', 0)):.4f}" if stats.get(col, {}).get('mean') is not None else '' for col in headers] + ['']
        writer.writerow(['mean'] + mean_row[1:])

        std_row = [f"{float(stats.get(col, {}).get('std', 0)):.4f}" if stats.get(col, {}).get('std') is not None else '' for col in headers] + ['']
        writer.writerow(['std'] + std_row[1:])

        ci_lower_row = [f"{float(stats.get(col, {}).get('ci_lower', 0)):.4f}" if stats.get(col, {}).get('ci_lower') is not None else '' for col in headers] + ['']
        writer.writerow(['ci_lower'] + ci_lower_row[1:])

        ci_upper_row = [f"{float(stats.get(col, {}).get('ci_upper', 0)):.4f}" if stats.get(col, {}).get('ci_upper') is not None else '' for col in headers] + ['']
        writer.writerow(['ci_upper'] + ci_upper_row[1:])

    print(f"Statistics exported to: {file_path}")

def create_boxplots(dataset_name: str, seed: int, reps: int, limits_data: Dict[int, Dict[str, Any]]):
    """Creates separate normalized and raw boxplots for each parameter across all repetitions with data points."""
    rep_numbers = sorted(limits_data.keys())
    if not rep_numbers:
        print("No limits data available for boxplots")
        return

    params = ['e_b', 'e_n', 'a_max', 'l', 'a', 'd', 'passes']
    data = {param: [] for param in params}

    for rep in rep_numbers:
        limits = limits_data[rep].get('limits', {})
        for param in params:
            min_val, max_val = limits.get(param, [0, 0])
            data[param].append(min_val)
            data[param].append(max_val)

    # Normalized boxplots with data points
    fig_norm, axes_norm = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes_norm = axes_norm.flatten()
    for idx, param in enumerate(params):
        if max(data[param]) - min(data[param]) > 0:
            normalized = [(x - min(data[param])) / (max(data[param]) - min(data[param])) for x in data[param]]
        else:
            normalized = data[param]
        axes_norm[idx].boxplot(normalized)
        # Add scatter plot for data points
        x_positions = np.random.normal(1, 0.04, len(normalized))  # Jitter for visibility
        axes_norm[idx].scatter(x_positions, normalized, alpha=0.5, color='red', s=20)
        axes_norm[idx].set_title(f'{param} (Normalized)')
        axes_norm[idx].set_ylabel('Normalized Values (0 to 1)')
        axes_norm[idx].grid(True, linestyle='--', alpha=0.7)
    for ax in axes_norm[len(params):]:
        ax.remove()
    plt.suptitle(f'Parameter Limits Variation (Normalized) Across All Reps (Seed: {seed}, Reps: {reps})')

    output_dir = f"logs/results/{dataset_name}/seed-{seed}_reps-{reps}"
    os.makedirs(output_dir, exist_ok=True)
    output_path_norm = f"{output_dir}/parameter_limits_normalized_boxplots.png"
    plt.savefig(output_path_norm)
    plt.close()
    print(f"Normalized boxplots saved to: {output_path_norm}")

    # Raw boxplots with data points
    fig_raw, axes_raw = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes_raw = axes_raw.flatten()
    for idx, param in enumerate(params):
        values = data[param]
        axes_raw[idx].boxplot(values)
        # Add scatter plot for data points
        x_positions = np.random.normal(1, 0.04, len(values))  # Jitter for visibility
        axes_raw[idx].scatter(x_positions, values, alpha=0.5, color='red', s=20)
        axes_raw[idx].set_title(f'{param} (Raw)')
        axes_raw[idx].set_ylabel('Raw Values')
        axes_raw[idx].grid(True, linestyle='--', alpha=0.7)
        axes_raw[idx].set_ylim(min(values) * 0.9, max(values) * 1.1)  # Dynamic y-limits
    for ax in axes_raw[len(params):]:
        ax.remove()
    plt.suptitle(f'Parameter Limits Variation (Raw) Across All Reps (Seed: {seed}, Reps: {reps})')

    output_path_raw = f"{output_dir}/parameter_limits_raw_boxplots.png"
    plt.savefig(output_path_raw)
    plt.close()
    print(f"Raw boxplots saved to: {output_path_raw}")

def read_all_data(dataset_name: str, seed: int, reps: int) -> Dict[str, Any]:
    """Reads all knowledge base CSV and limits JSON files, returning combined data."""
    kb_files = get_knowledge_base_files(dataset_name, seed, reps)
    all_kb_data = []
    total_discarded = 0

    for file_path in kb_files:
        print(f"Reading knowledge base file: {file_path}")
        file_data, discarded = read_knowledge_base_file(file_path)
        if file_data:
            rep_number = int(file_path.split('rep')[-1].replace('.csv', ''))
            for entry in file_data:
                entry['rep'] = rep_number
            all_kb_data.extend(file_data)
            total_discarded += discarded
        else:
            print(f"No data read from {file_path}")

    limits_info = get_limits_files(dataset_name, seed, reps)
    limits_data = {}

    for file_path in limits_info['limits_files']:
        print(f"Reading limits file: {file_path}")
        rep_number = int(file_path.split('rep')[-1].replace('.json', ''))
        limits = read_limits_file(file_path)
        if limits:
            limits_data[rep_number] = {'limits': limits}
        else:
            print(f"No limits data read from {file_path}")

    for file_path in limits_info['updated_limits_files']:
        print(f"Reading updated limits file: {file_path}")
        rep_number = int(file_path.split('rep')[-1].replace('.json', ''))
        limits = read_limits_file(file_path)
        if limits:
            if rep_number not in limits_data:
                limits_data[rep_number] = {}
            limits_data[rep_number]['updated_limits'] = limits
        else:
            print(f"No updated limits data read from {file_path}")

    return {
        'knowledge_base': all_kb_data,
        'limits': limits_data,
        'total_discarded': total_discarded,
        'total_reps': len(kb_files)
    }

def main(dataset_name: str, seed: int, reps: int):
    """Main function to read, process, and export statistics and boxplots."""
    print(f"Processing data for dataset: {dataset_name}, seed: {seed}, reps: {reps}")
    all_data = read_all_data(dataset_name, seed, reps)

    kb_data = all_data['knowledge_base']
    total_reps = all_data['total_reps']
    total_discarded = all_data['total_discarded']

    if kb_data:
        print(f"Read {len(kb_data)} total knowledge base entries from {total_reps} files")
        print(f"Total sets discarded: {total_discarded}")

        stats = calculate_statistics(kb_data)

        export_statistics_csv(dataset_name, seed, reps, kb_data, total_reps, total_discarded, stats)
    else:
        print("No knowledge base data found.")

    limits_data = all_data['limits']
    if limits_data:
        create_boxplots(dataset_name, seed, reps, limits_data)
    else:
        print("No limits data found for boxplots.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python results.py <dataset_name> <seed> <reps>")
        print("Example: python results.py iris 1 10")
        sys.exit(1)

    dataset_name = sys.argv[1]
    seed = int(sys.argv[2])
    reps = int(sys.argv[3])
    main(dataset_name, seed, reps)