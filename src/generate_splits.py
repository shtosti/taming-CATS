"""
Script: generate_splits.py

Purpose: Generate stratified train, validation, and test splits for the datasets.

Usage: python generate_splits.py
- This script generates stratified train, validation, and test splits
- stratification metric (FKGL) and number of bins (35). 
- The splits are saved as JSONL files in the data/splits directory.

Parameters:
- STRAT_METRIC: Stratification metric (FKGL)
- NUM_BINS: Number of bins (35)

Output:
- train.jsonl: Train split
- val.jsonl: Validation split
- test.jsonl: Test split

- dev.jsonl: Dev split (for experimentation) - used in some experiments
"""


import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp


COLOR_MAP_FILE = "./../data/colormap/color_map.json"
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
    char_lengths = [len(line["source_text"]) for line in data]
    lower_threshold = np.percentile(char_lengths, lower_percentile)
    upper_threshold = np.percentile(char_lengths, upper_percentile)
    filtered_data = [
        line for line in data
        if lower_threshold <= len(line["source_text"]) <= upper_threshold
    ]
    return filtered_data

def save_jsonl(data, filepath):
    """Save dataset to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in data:
            # TODO remove "original_split" from metadata on the dataset preprocessing level
            # Remove original_split from metadata
            if "metadata" in line and "original_split" in line["metadata"]:
                del line["metadata"]["original_split"]

            f.write(json.dumps(line) + "\n")

def generate_splits(data, metric_values, strat_metric, num_bins=35, seed=None):
    """Generate stratified train, validation, and test splits."""
    values = np.array(metric_values[strat_metric])
    bins = np.histogram_bin_edges(values, bins=num_bins)
    bin_indices = np.digitize(values, bins)
    
    df = pd.DataFrame({"data": data, "bin": bin_indices})
    
    train_data, val_data, test_data = [], [], []

    np.random.seed(seed)

    for bin_id in np.unique(bin_indices):
        bin_data = df[df["bin"] == bin_id]["data"].tolist()
        np.random.shuffle(bin_data)
        train_data.extend(bin_data[:int(len(bin_data) * 0.8)])  # 80% train
        val_data.extend(bin_data[int(len(bin_data) * 0.8):int(len(bin_data) * 0.9)])  # 10% validation
        test_data.extend(bin_data[int(len(bin_data) * 0.9):])  # 10% test

    return train_data, val_data, test_data

def generate_splits_with_dev(data, metric_values, strat_metric, num_bins=35, seed=None):
    """Generate stratified train, validation, and test splits + dev for experimentation."""
    values = np.array(metric_values[strat_metric])
    bins = np.histogram_bin_edges(values, bins=num_bins)
    bin_indices = np.digitize(values, bins)
    
    df = pd.DataFrame({"data": data, "bin": bin_indices})
    
    train_data, val_data, test_data, dev_data = [], [], [], []

    np.random.seed(seed)

    for bin_id in np.unique(bin_indices):
        bin_data = df[df["bin"] == bin_id]["data"].tolist()

        np.random.shuffle(bin_data)

        train_data.extend(bin_data[:int(len(bin_data) * 0.7)])  # 70% train
        dev_data.extend(bin_data[int(len(bin_data) * 0.7):int(len(bin_data) * 0.8)])  # 10% dev
        val_data.extend(bin_data[int(len(bin_data) * 0.8):int(len(bin_data) * 0.9)])  # 10% val
        test_data.extend(bin_data[int(len(bin_data) * 0.9):])  # 10% test

    return train_data, dev_data, val_data, test_data


def main():
    DATASETS = [
        "medeasi",
        "newsela",
        "simpa",
        "wikilarge_global"
    ]
    DATA_DIR = "./../data"

    # parameters selected through experimentation with distributions (hyperparameter tuning)
    STRAT_METRIC = "FKGL"
    NUM_BINS = 25
    SEED = 42
    WITH_DEV = False
    
    for dataset_name in DATASETS:
        print(f"Processing {dataset_name}...")

        if WITH_DEV:
            SPLIT_SAVE_DIR = f"./../data/splits_w_dev/{dataset_name}"
        else:
            SPLIT_SAVE_DIR = f"./../data/splits/{dataset_name}"
        os.makedirs(SPLIT_SAVE_DIR, exist_ok=True)
        
        dataset_path = f"{DATA_DIR}/datasets/{dataset_name}/dataset.jsonl"
        data = load_jsonl(dataset_path)
        data = remove_outliers_by_char_length(data, lower_percentile=3, upper_percentile=97)

        # Extract metric values
        full_metric_values = extract_metrics(data, ["char_count", "word_count", "sentence_count", "FKGL", "ARI", "FRE", "Dale-Chall"])
        
        if WITH_DEV:
            train_data, dev_data, val_data, test_data = generate_splits_with_dev(data, full_metric_values, STRAT_METRIC, num_bins=NUM_BINS, seed=SEED)
            save_jsonl(train_data, f"{SPLIT_SAVE_DIR}/train.jsonl")
            save_jsonl(dev_data, f"{SPLIT_SAVE_DIR}/dev.jsonl")
            save_jsonl(val_data, f"{SPLIT_SAVE_DIR}/val.jsonl")
            save_jsonl(test_data, f"{SPLIT_SAVE_DIR}/test.jsonl")
        else:
            train_data, val_data, test_data = generate_splits(data, full_metric_values, STRAT_METRIC, num_bins=NUM_BINS, seed=SEED)
            save_jsonl(train_data, f"{SPLIT_SAVE_DIR}/train.jsonl")
            save_jsonl(val_data, f"{SPLIT_SAVE_DIR}/val.jsonl")
            save_jsonl(test_data, f"{SPLIT_SAVE_DIR}/test.jsonl")

        print(f"Splits for {dataset_name} generated and saved.")

if __name__ == "__main__":
    main()
