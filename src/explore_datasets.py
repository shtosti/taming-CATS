import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr



def load_jsonl(filepath: str) -> list:
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def plot_compression(data: list, metric: str, dataset_dir: str) -> None:

    source_lengths, target_lengths = get_compression_values(data, metric)

    # Check if either source_lengths or target_lengths is empty
    if not source_lengths or not target_lengths:
        print(f"Warning: No valid values for metric '{metric}' found. Skipping plot for this metric.")
        return  # Skip plotting for this metric if no values are found

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
    plt.hist(source_lengths, bins=bins, alpha=0.3, label="Original", color='blue', density=True, edgecolor="black")
    plt.hist(target_lengths, bins=bins, alpha=0.3, label="Simplification", color='green', density=True, edgecolor="black")
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
    plt.subplot(1,2,2)
    plt.scatter(target_lengths, source_lengths, alpha=0.3, color='orchid', label="Source vs Target")

    if len(source_lengths) > 1 and len(target_lengths) > 1:

        corr_coeff, _ = pearsonr(source_lengths, target_lengths)

        # Calculate average direction (slope) using the mean of the target values
        avg_source = np.mean(source_lengths)
        avg_target = np.mean(target_lengths)

        # Calculate slope (rise over run)
        slope = avg_source / avg_target if avg_target != 0 else 0

        # Plot the line starting from the origin
        line_x = np.linspace(0, max(target_lengths), 100)  # Line range
        line_y = slope * line_x  # y = mx (Line passing through origin)
        plt.plot(line_x, line_y, color='orchid', linestyle='-', label="Mean value")

        # Display the Pearson correlation on the plot
        plt.text(0.05, 0.95, f"Pearson: {corr_coeff:.4f}", transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', color='black')

    # Set the same scale for both axes
    # min_val = min(min(source_lengths), min(target_lengths))
    min_val = 0
    max_val = max(max(source_lengths), max(target_lengths))
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel(f'Simplification {metric} Count')
    plt.ylabel(f'Original {metric} Count')
    
    # save plots
    output_dir = f"{dataset_dir}/stats/comparison_source_target"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric}.png", dpi=400)
    

def get_compression_values(data: list, metric: str) -> tuple:
    source_val_arr = []
    target_val_arr = []

    for line in data:

        simplifications = line["simplifications"]
        for simplification in simplifications:
            if metric in ["char", "sentence", "word"]:
                target_value = simplification["target_metrics"][f"{metric}_count"]
                target_val_arr.append(target_value)
                source_value = line["source_metrics"][f"{metric}_count"]
                source_val_arr.append(source_value)
            else:
                target_value = simplification["target_metrics"][f"{metric}"]
                target_val_arr.append(target_value)
                source_value = line["source_metrics"][f"{metric}"]
                source_val_arr.append(source_value)
    
    return source_val_arr, target_val_arr

def get_eval_values(data: list, metric: str) -> list:
    target_val_arr = []

    for line in data:
        simplifications = line["simplifications"]
        for simplification in simplifications:
            if metric in ["BLEU", "BERTScore"]:
                target_value = simplification["target_metrics"][f"{metric}"]
                target_val_arr.append(target_value)
    
    return target_val_arr

def plot_eval_values(data: list, metric: str, dataset_dir: str) -> None:
    """ Plot BLEU and BERTScore. """

    target_vals = get_eval_values(data, metric)

    # Plotting the distribution shift
    # Calculate the common bin edges for source and target distributions
    min_val = min(target_vals)
    max_val = max(target_vals)
    bins = np.linspace(min_val, max_val, 25)  # TODO adjust n bins to ompimize the visual effect 

    # Plotting the distribution shift
    plt.figure(figsize=(12, 6))

    ############ histograms with aligned bins ############
    plt.subplot(1, 2, 1)
    plt.hist(target_vals, bins=bins, alpha=0.4, label="Simplification", color='green', density=True, edgecolor="black")
    # Add KDE for smooth distribution curves
    if len(set(target_vals)) > 1:
        sns.kdeplot(target_vals, color='green', linewidth=1, label="Simplification KDE")
    plt.xlabel(f'{metric}')
    plt.ylabel('Frequency')
    plt.legend()

    ############ violin plot ############
    plt.subplot(1, 2, 2)
    sns.violinplot(x=target_vals, color='orchid', inner="quartile")
    sns.boxplot(x=target_vals, color='black', width=0.15, fliersize=3)  # Add box plot for summary stats
    plt.xlabel(f'{metric} Score')
    plt.title(f'Distribution of {metric} Scores')
    
    # save plots
    output_dir = f"{dataset_dir}/stats/comparison_source_target"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric}.png", dpi=400)
    
def save_log(data: list, dataset_name: str, dataset_dir: str, comparison_metrics: list, similarity_metrics: list) -> None:
    log_data = {}
    
    for metric in comparison_metrics:
        source_vals, target_vals = get_compression_values(data, metric)
        if source_vals and target_vals:
            log_data[metric] = {
                "source_mean": np.mean(source_vals),
                "source_std": np.std(source_vals),
                "source_median": np.median(source_vals),
                "target_mean": np.mean(target_vals),
                "target_std": np.std(target_vals),
                "target_median": np.median(target_vals)
            }
    
    for metric in similarity_metrics:
        target_vals = get_eval_values(data, metric)
        if target_vals:
            log_data[metric] = {
                "mean": np.mean(target_vals),
                "std": np.std(target_vals),
                "median": np.median(target_vals)
            }
    
    log_path = os.path.join(dataset_dir, "stats", "dataset_stats.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)
    
    print(f"Saved log for {dataset_name} at {log_path}")

def plot_source_target_comparison(source_vals: list, target_vals: list, metric: str, dataset_dir: str) -> None:
    df = pd.DataFrame({"Source": source_vals, "Target": target_vals})
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, palette=["orchid", "green"], width=0.3)
    
    # Plot trajectory from median to median and mean to mean
    source_mean, target_mean = np.mean(source_vals), np.mean(target_vals)
    source_median, target_median = np.median(source_vals), np.median(target_vals)
    
    plt.plot([0, 1], [source_mean, target_mean], "r-o", label="Mean trajectory")
    plt.plot([0, 1], [source_median, target_median], "b-o", label="Median trajectory")
    
    plt.xticks([0, 1], ["Original", "Simplification"])
    plt.ylabel(metric)
    # plt.title(f"Comparison of Source and Target for {metric}")
    plt.legend()
    plt.grid()
    
    output_dir = f"{dataset_dir}/stats/trajectory"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{metric}.png", dpi=400)
    plt.close()

def main():

    DATASETS = [
        "simpa"
        # "simpa_lexical",
        # "simpa_syntactic",
        # "newsela",
        # "medeasi",
        # "wikilarge_1000",
        # "wikilarge_1000_from_splits",
        # "wikilarge_2000",
        # "wikilarge_2000_from_splits",
        # "wikilarge_3000",
        # "wikilarge_3000_from_splits",
        # "wikilarge"
    ]
    DATA_DIR = "./../data" 

    for DATASET in DATASETS:
        DATASET_DIR = f"{DATA_DIR}/datasets/{DATASET}"
        DATASET_PATH = f"{DATASET_DIR}/dataset.jsonl"

        dataset = load_jsonl(DATASET_PATH)
        # dataset = dataset[:500] # for dev
        # metrics_to_plot = [
        #                     "char", 
        #                     "word", 
        #                     "sentence", 
        #                     "FRE", 
        #                     "ARI", 
        #                     "FKGL", 
        #                     "Dale-Chall"
        #                     ]
        # # for metric in metrics_to_plot:
        # #     plot_compression(data=dataset, metric=metric, dataset_dir=DATASET_DIR)

        # eval_metrics_to_plot = [
        #                         "BLEU",
        #                         "BERTScore"
        #                         ]
        # for metric in eval_metrics_to_plot:
        #     plot_eval_values(data=dataset, metric=metric, dataset_dir=DATASET_DIR)

        comparison_metrics = ["char", "word", "sentence", "FRE", "ARI", "FKGL", "Dale-Chall"]
        similarity_metrics = ["BLEU", "BERTScore"]
        
        for metric in comparison_metrics:
            plot_compression(data=dataset, metric=metric, dataset_dir=DATASET_DIR)
            source_vals, target_vals = get_compression_values(dataset, metric)
            plot_source_target_comparison(source_vals, target_vals, metric, DATASET_DIR)
        
        for metric in similarity_metrics:
            plot_eval_values(data=dataset, metric=metric, dataset_dir=DATASET_DIR)

        save_log(data=dataset, dataset_name=DATASET, dataset_dir=DATASET_DIR, 
                 comparison_metrics=comparison_metrics, similarity_metrics=similarity_metrics)


if __name__ == "__main__":
    main()