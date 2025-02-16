import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import numpy as np
import seaborn as sns # to fit hist 


def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def plot_compression(data, metric, dataset_dir):

    source_lengths, target_lengths = get_compression_values(data, metric)
    # ensure the arrays have equal lengths
    # NB: the original is repeated for every corresponding simplification instance 
    print("Equal length:", len(source_lengths) == len(target_lengths))

    # Plotting the distribution shift
    # Calculate the common bin edges for source and target distributions
    min_val = min(min(source_lengths), min(target_lengths))
    max_val = max(max(source_lengths), max(target_lengths))
    bins = np.linspace(min_val, max_val, 25)  # TODO adjust n bins to ompimize the visual effect 

    # Plotting the distribution shift
    plt.figure(figsize=(12, 6))
    
    ############ histograms with aligned bins ############
    plt.subplot(1, 2, 1)
    plt.hist(source_lengths, bins=bins, alpha=0.4, label="Original", color='blue', density=True)
    plt.hist(target_lengths, bins=bins, alpha=0.4, label="Simplification", color='green', density=True)
    # Add KDE for smooth distribution curves
    if len(set(source_lengths)) > 1:  # More than one unique value
        sns.kdeplot(source_lengths, color='blue', linewidth=1, label="Original KDE")
    if len(set(target_lengths)) > 1:
        sns.kdeplot(target_lengths, color='green', linewidth=1, label="Simplification KDE")
    if metric in ["char_count", "sentence_count", "word_count"]:
        plt.xlabel(f'{metric} Count')
    else:
        plt.xlabel(f'{metric}')
    plt.ylabel('Frequency')
    plt.legend()

    ############ scatter plot ############
    plt.subplot(1, 2, 2)
    plt.scatter(target_lengths, source_lengths, alpha=0.4, color='green', label="Source vs Target")

    plt.subplot(1,2,2)
    plt.scatter(target_lengths, source_lengths, alpha=0.4, color='green', label="Source vs Target")
    # Set the same scale for both axes
    # min_val = min(min(source_lengths), min(target_lengths))
    min_val = 0
    max_val = max(max(source_lengths), max(target_lengths))
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel(f'Simplification {metric} Count')
    plt.ylabel(f'Original {metric} Count')
    
    # save plots
    output_dir = f"{dataset_dir}/stats/visuals"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric}.png", dpi=400)
    

def get_compression_values(data, metric):
    source_lengths = []
    target_lengths = []

    for line in data:

        simplifications = line["simplifications"]
        for simplification in simplifications:
            if metric in ["char", "sentence", "word"]:
                target_value = simplification["target_metrics"][f"{metric}_count"]
                target_lengths.append(target_value)
                source_value = line["source_metrics"][f"{metric}_count"]
                source_lengths.append(source_value)
            else:
                target_value = simplification["target_metrics"][f"{metric}"]
                target_lengths.append(target_value)
                source_value = line["source_metrics"][f"{metric}"]
                source_lengths.append(source_value)
    
    return source_lengths, target_lengths
    

def main():

    DATASETS = [
        "simpa_lexical",
        "simpa_syntactic",
        "newsela",
        "medeasi",
        # "wikilarge"
    ]
    DATA_DIR = "./../data" 

    for DATASET in DATASETS:
        DATASET_DIR = f"{DATA_DIR}/datasets/{DATASET}"
        DATASET_PATH = f"{DATASET_DIR}/dataset.jsonl"

        dataset = load_jsonl(DATASET_PATH)
        # dataset = dataset[:500]
        metrics_to_plot = [
                            "char", 
                            "word", 
                            "sentence", 
                            "FRE", 
                            "ARI", 
                            "FKGL", 
                            "Dale-Chall"
                            ]
        for metric in metrics_to_plot:
            plot_compression(data=dataset, metric=metric, dataset_dir=DATASET_DIR)


if __name__ == "__main__":
    main()