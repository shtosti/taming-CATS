import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_jsonl(filepath):
    """Load dataset from a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def extract_metrics(data, metrics):
    """Extract specified metric values from dataset."""
    return {metric: [line["source_metrics"].get(metric, 0) for line in data] for metric in metrics}

def stratified_sampling(data, metric_values, metric, num_bins=20, subset_size=2000):
    """Perform stratified sampling based on a single metric."""
    values = np.array(metric_values[metric])
    bins = np.histogram_bin_edges(values, bins=num_bins)
    bin_indices = np.digitize(values, bins)
    df = pd.DataFrame({"data": data, "bin": bin_indices})
    subset = df.groupby("bin", group_keys=False).apply(lambda x: x.sample(frac=subset_size / len(df), random_state=42))
    return subset["data"].tolist()

def plot_distributions(metrics, full_metric_values, train_metric_values, val_metric_values, test_metric_values, save_dir):
    """Plot the distributions of the metrics across the entire dataset, train, validation, and test splits."""
    save_dir = f"{save_dir}/visuals"
    os.makedirs(save_dir, exist_ok=True)
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        sns.kdeplot(train_metric_values[metric], color="green", label="Train", linewidth=2, bw_adjust=1, alpha=0.7)
        sns.kdeplot(val_metric_values[metric], color="orange", label="Validation", linewidth=2, bw_adjust=0.9, alpha=0.7)
        sns.kdeplot(test_metric_values[metric], color="red", label="Test", linewidth=2, bw_adjust=1, alpha=0.7)
        sns.kdeplot(full_metric_values[metric], color="blue", label="Full Dataset", linewidth=2, bw_adjust=0.9, alpha=0.7)

        
        plt.xlabel(f"{metric}")
        plt.ylabel("Density")
        # plt.title(f"Distribution of {metric}")
        plt.legend()

        plt.savefig(f"{save_dir}/{metric}.png", dpi=400)
        plt.close()

def save_jsonl(data, filepath):
    """Save dataset to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

def generate_splits(data, metric_values, strat_metric, num_bins=20, subset_size=2000):
    """Generate stratified train, validation, and test splits."""
    values = np.array(metric_values[strat_metric])
    bins = np.histogram_bin_edges(values, bins=num_bins)
    bin_indices = np.digitize(values, bins)
    
    df = pd.DataFrame({"data": data, "bin": bin_indices})
    
    # Split into train, validation, and test (80%, 10%, 10%)
    train_size = int(subset_size * 0.8)
    val_size = int(subset_size * 0.1)
    test_size = int(subset_size * 0.1)

    train_subset = df.groupby("bin", group_keys=False).apply(lambda x: x.sample(frac=train_size / len(df), random_state=42))
    val_subset = df.groupby("bin", group_keys=False).apply(lambda x: x.sample(frac=val_size / len(df), random_state=42))
    test_subset = df.groupby("bin", group_keys=False).apply(lambda x: x.sample(frac=test_size / len(df), random_state=42))

    return train_subset["data"].tolist(), val_subset["data"].tolist(), test_subset["data"].tolist()

def main():
    DATASETS = [
        "medeasi",
        "newsela",
        "simpa_lexical",
        "simpa_syntactic",
        "wikilarge_1000", 
        "wikilarge_1000_from_splits",
        "wikilarge_2000", 
        "wikilarge_2000_from_splits",
        "wikilarge_3000",
        "wikilarge_3000_from_splits"
        ]
    DATA_DIR = "./../data"

    # Metrics to be used for stratification and visualization
    metrics = ["char_count", "word_count", "sentence_count", "FKGL", "ARI", "FRE", "Dale-Chall"]

    for dataset_name in DATASETS:
        print(f"Processing {dataset_name}...")

        SAVE_DIR = f"./../data/splits/{dataset_name}"
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        dataset_path = f"{DATA_DIR}/datasets/{dataset_name}/dataset.jsonl"
        data = load_jsonl(dataset_path)
        
        # Extract metric values
        metric_values = extract_metrics(data, metrics)
        
        # Generate stratified splits based on character count
        strat_metric = "char_count"
        subset_size = len(data)
        train_data, val_data, test_data = generate_splits(data, metric_values, strat_metric, subset_size=subset_size)

        # Save splits
        save_jsonl(train_data, f"{SAVE_DIR}/train.jsonl")
        save_jsonl(val_data, f"{SAVE_DIR}/val.jsonl")
        save_jsonl(test_data, f"{SAVE_DIR}/test.jsonl")

        # Plot distributions of metrics for full dataset, train, val, and test
        full_metric_values = extract_metrics(data, metrics)
        train_metric_values = extract_metrics(train_data, metrics)
        val_metric_values = extract_metrics(val_data, metrics)
        test_metric_values = extract_metrics(test_data, metrics)

        plot_distributions(metrics, full_metric_values, train_metric_values, val_metric_values, test_metric_values, SAVE_DIR)
        
        print(f"Splits for {dataset_name} generated and saved.")
        
if __name__ == "__main__":
    main()
