import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

COLOR_MAP_FILE = "./../data/colormaps/color_map.json"
with open(COLOR_MAP_FILE, "r") as f:
    COLOR_MAP = json.load(f)

def load_jsonl(filepath):
    """Load dataset from a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def extract_metrics(data, metrics):
    """Extract specified metric values from dataset."""
    return {metric: [line["source_metrics"].get(metric, 0) for line in data] for metric in metrics}

def remove_outliers_by_char_length(data: list, lower_percentile=3, upper_percentile=97) -> list:
    """Remove entries based on character length outliers (3rd and 97th percentiles)."""
    
    # Calculate the character length for each entry
    char_lengths = [len(line["source_text"]) for line in data]

    # Calculate the lower and upper percentiles for character length
    lower_threshold = np.percentile(char_lengths, lower_percentile)
    upper_threshold = np.percentile(char_lengths, upper_percentile)

    # Filter out entries based on the character length thresholds
    filtered_data = [
        line for line in data
        if lower_threshold <= len(line["source_text"]) <= upper_threshold
    ]

    return filtered_data

def plot_distributions(metrics, full_metric_values, train_metric_values, val_metric_values, test_metric_values, save_dir):
    """Plot the distributions of the metrics across the entire dataset, train, validation, and test splits."""
    save_dir = f"{save_dir}/visuals"
    os.makedirs(save_dir, exist_ok=True)
    
    for metric in metrics:
        plt.figure(figsize=(5, 5))

        sns.kdeplot(train_metric_values[metric], color=COLOR_MAP["splits"].get("train"), label="Train", linewidth=2, alpha=1)
        sns.kdeplot(val_metric_values[metric], color=COLOR_MAP["splits"].get("val"), label="Validation", linewidth=2, alpha=1)
        sns.kdeplot(test_metric_values[metric], color=COLOR_MAP["splits"].get("test"), label="Test", linewidth=2, alpha=1)
        sns.kdeplot(full_metric_values[metric], color=COLOR_MAP["splits"].get("full"), label="Full Dataset", linewidth=2, alpha=1)

        
        plt.xlabel(f"{metric}")
        plt.ylabel("Density")
        plt.grid(True)
        # plt.title(f"Distribution of {metric}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{metric}.png", dpi=400)

def plot_distributions_on_one_image(metrics, full_metric_values, train_metric_values, val_metric_values, test_metric_values, save_dir):
    """Plot all metric distributions in a single image while maintaining the original appearance."""
    save_dir = f"{save_dir}/visuals"
    os.makedirs(save_dir, exist_ok=True)
    
    num_metrics = len(metrics)
    cols = 3  # Number of columns in the grid layout
    rows = (num_metrics + cols - 1) // cols  # Calculate number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()  

    # Store legend handles & labels from first subplot
    legend_handles = None

    for i, metric in enumerate(metrics):
        ax = axes[i]

        kde_train = sns.kdeplot(train_metric_values[metric], color=COLOR_MAP["splits"].get("train"), label="Train", linewidth=1, alpha=1, ax=ax)
        kde_val = sns.kdeplot(val_metric_values[metric], color=COLOR_MAP["splits"].get("val"), label="Validation", linewidth=1, alpha=1, ax=ax)
        kde_test = sns.kdeplot(test_metric_values[metric], color=COLOR_MAP["splits"].get("test"), label="Test", linewidth=1, alpha=1, ax=ax)
        kde_full = sns.kdeplot(full_metric_values[metric], color=COLOR_MAP["splits"].get("full"), label="Full Dataset", linewidth=1, alpha=1, ax=ax)

        ax.set_xlabel(f"{metric}")
        ax.set_ylabel("Density")
        ax.grid(True, linewidth=0.5)

        # Capture legend elements only once
        if legend_handles is None:
            legend_handles, labels = ax.get_legend_handles_labels()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a single legend outside the subplots at the bottom
    fig.legend(
                legend_handles, 
                labels, 
                loc="upper right"
                )

    plt.tight_layout()
    save_path = f"{save_dir}/all_metrics_comparison.png"
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

def save_jsonl(data, filepath):
    """Save dataset to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

def generate_splits(data, metric_values, strat_metric, num_bins=35):
    """Generate stratified train, validation, and test splits."""
    values = np.array(metric_values[strat_metric])
    bins = np.histogram_bin_edges(values, bins=num_bins)
    bin_indices = np.digitize(values, bins)
    
    df = pd.DataFrame({"data": data, "bin": bin_indices})
    
    # Split into train, validation, and test (80%, 10%, 10%)
    train_data, val_data, test_data = [], [], []

    # Loop through each bin and create splits
    for bin_id in np.unique(bin_indices):
        bin_data = df[df["bin"] == bin_id]["data"].tolist()
        
        # Shuffle the bin data and split it
        np.random.shuffle(bin_data)
        
        # Add 80% to train, 10% to validation, and 10% to test
        train_data.extend(bin_data[:int(len(bin_data) * 0.8)])  # 80% train
        val_data.extend(bin_data[int(len(bin_data) * 0.8):int(len(bin_data) * 0.9)])  # 10% validation
        test_data.extend(bin_data[int(len(bin_data) * 0.9):])  # 10% test

    # Return the non-overlapping splits
    return train_data, val_data, test_data


def main():
    DATASETS = [
        "medeasi",
        "newsela",
        "simpa",
        # "wikilarge"
        ]
    DATA_DIR = "./../data"
    BINS = [15,
            25,
            35,
            45
            ]
    
    EXPERIMENT_RESULTS = []

    for NUM_BINS in BINS:
        # Metrics to be used for stratification and visualization
        METRICS = ["char_count", "word_count", "sentence_count", "FKGL", "ARI", "FRE", "Dale-Chall"]
        STRAT_METRIC = "ARI"

        for dataset_name in DATASETS:
            print(f"Processing {dataset_name}...")

            SAVE_DIR = f"./../experiments/splits/stratified_by_{STRAT_METRIC}/num_bins_{NUM_BINS}/{dataset_name}"
            os.makedirs(SAVE_DIR, exist_ok=True)
            SPLIT_SAVE_DIR = f"./../data/splits/split_by_{STRAT_METRIC}/num_bins_{NUM_BINS}/{dataset_name}"
            os.makedirs(SPLIT_SAVE_DIR, exist_ok=True)
            
            dataset_path = f"{DATA_DIR}/datasets/{dataset_name}/dataset.jsonl"
            data = load_jsonl(dataset_path)
            data = remove_outliers_by_char_length(
                                                    data,
                                                    lower_percentile=3,
                                                    upper_percentile=97
                                                    )
            
            # Extract metric values
            full_metric_values = extract_metrics(data, METRICS)
            
            # Generate stratified splits based on stratification metric
            train_data, val_data, test_data = generate_splits(data, full_metric_values, STRAT_METRIC, num_bins=NUM_BINS)

            # # Save splits
            save_jsonl(train_data, f"{SPLIT_SAVE_DIR}/train.jsonl")
            save_jsonl(val_data, f"{SPLIT_SAVE_DIR}/val.jsonl")
            save_jsonl(test_data, f"{SPLIT_SAVE_DIR}/test.jsonl")

            # Plot distributions of metrics for full dataset, train, val, and test
            full_metric_values = extract_metrics(data, METRICS)
            train_metric_values = extract_metrics(train_data, METRICS)
            val_metric_values = extract_metrics(val_data, METRICS)
            test_metric_values = extract_metrics(test_data, METRICS)

            plot_distributions(METRICS, full_metric_values, train_metric_values, val_metric_values, test_metric_values, SAVE_DIR)
            plot_distributions_on_one_image(METRICS, full_metric_values, train_metric_values, val_metric_values, test_metric_values, SAVE_DIR)

            print(f"Splits for {dataset_name} generated and saved.")

            # Calculate KL Divergence between distributions
            ks_train,_ = ks_2samp(full_metric_values[STRAT_METRIC], train_metric_values[STRAT_METRIC])
            ks_val,_ = ks_2samp(full_metric_values[STRAT_METRIC], val_metric_values[STRAT_METRIC])
            ks_test,_ = ks_2samp(full_metric_values[STRAT_METRIC], test_metric_values[STRAT_METRIC])

            # Store results for each experiment
            EXPERIMENT_RESULTS.append({
                "dataset": dataset_name,
                "strat_metric": STRAT_METRIC,
                "num_bins": NUM_BINS,
                "KS_full_train": ks_train,
                "KS_full_val": ks_val,
                "KS_full_test": ks_test,
                "average_KS": np.mean([ks_train, ks_val, ks_test])
            })

    with open(f"./../experiments/splits/stratified_by_{STRAT_METRIC}/all_results.json", "w") as f:
        json.dump(EXPERIMENT_RESULTS, f, indent=4)
        
if __name__ == "__main__":
    main()
