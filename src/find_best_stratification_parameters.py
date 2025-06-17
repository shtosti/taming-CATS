import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

results = {}

EXPERIMENT_NAME = "splits_sampling"
RESULTS_FILE = f"./../experiments/{EXPERIMENT_NAME}/all_results.json"
LOG_FILE_PER_DATASET = f"./../experiments/{EXPERIMENT_NAME}/best_stratification_results_per_dataset.txt"
LOG_FILE_ACROSS_DATASETS = f"./../experiments/{EXPERIMENT_NAME}/best_stratification_results_across_datasets.txt"
with open("./../data/colormap/color_map.json", "r") as f:
    COLOR_MAP = json.load(f)

with open(RESULTS_FILE, "r") as f:
    experiment_results = json.load(f)

df = pd.DataFrame(experiment_results)


with open(LOG_FILE_PER_DATASET, "w") as log_file:
    log_file.write("Best Stratification Results (Per Dataset)\n")
    log_file.write("="*50 + "\n\n")
    log_file.write("Note: For each dataset, the best stratification metric and number of bins are "
                   "determined based on the lowest average KS divergence. All seed runs (10 seeds)"
                   "and averaged the KS divergence values. The metric with the lowest average KS divergence "
                   "across all combinations of stratification metric and number of bins is chosen.\n")
    log_file.write("="*50 + "\n\n")
    
    # Per dataset: Find the best result for each stratification metric and number of bins, for each dataset
    for dataset in df["dataset"].unique():
        log_file.write(f"\nProcessing dataset: {dataset}\n")
        log_file.write("-"*50 + "\n")
        
        dataset_df = df[df["dataset"] == dataset]
        
        # Find the best result for each stratification metric and number of bins
        best_results_per_metric = dataset_df.groupby(["strat_metric", "num_bins"]).agg(
            mean_KS=("average_KS", "mean"),
            std_KS=("average_KS", "std")
        ).reset_index()
        
        # Find the best stratification metric (lowest mean KS) and best num_bins for that metric
        best_result = best_results_per_metric.loc[best_results_per_metric["mean_KS"].idxmin()]
        
        best_strat_metric = best_result["strat_metric"]
        best_num_bins = int(best_result["num_bins"])
        best_avg_ks = best_result["mean_KS"]
        
        # Write best results for this dataset to log file
        log_file.write(f"Best Stratification Metric: {best_strat_metric}\n")
        log_file.write(f"Best Number of Bins: {best_num_bins}\n")
        log_file.write(f"Lowest Average KS Divergence: {best_avg_ks:.6f}\n")
        log_file.write("-"*50 + "\n")
        
        # Save the best stratification results for this dataset
        results[dataset] = {
            "best_strat_metric": best_strat_metric,
            "best_num_bins": best_num_bins,
            "best_avg_ks": best_avg_ks
        }

with open(f"./../experiments/{EXPERIMENT_NAME}/best_stratification_results_per_dataset.json", "w") as f:
    json.dump(results, f, indent=4)

with open(LOG_FILE_ACROSS_DATASETS, "w") as log_file:
    log_file.write("Best Stratification Results (Across All Datasets)\n")
    log_file.write("="*50 + "\n\n")
    log_file.write("Note: The best stratification metric and number of bins across all datasets were "
                   "determined by aggregating the average KS divergence values across all datasets. "
                   "The combination with the lowest average KS divergence across all datasets is considered the best.\n")
    log_file.write("="*50 + "\n\n")
    
    all_results = df.groupby(["strat_metric", "num_bins"]).agg(
        mean_KS=("average_KS", "mean"),
        std_KS=("average_KS", "std")
    ).reset_index()

    # best stratification metric (lowest mean KS) across all datasets
    best_result = all_results.loc[all_results["mean_KS"].idxmin()]
    
    best_strat_metric = best_result["strat_metric"]
    best_num_bins = int(best_result["num_bins"])
    best_avg_ks = best_result["mean_KS"]
    
    log_file.write(f"Best Stratification Metric (Across All Datasets): {best_strat_metric}\n")
    log_file.write(f"Best Number of Bins: {best_num_bins}\n")
    log_file.write(f"Lowest Average KS Divergence: {best_avg_ks:.6f}\n")
    log_file.write("-"*50 + "\n")
    
    results["best_strat_metric_across_datasets"] = best_strat_metric
    results["best_num_bins_across_datasets"] = best_num_bins
    results["best_avg_ks_across_datasets"] = best_avg_ks

with open(f"./../experiments/{EXPERIMENT_NAME}/best_stratification_results_across_datasets.json", "w") as f:
    json.dump(results, f, indent=4)
print(f"Best stratification results per dataset saved to {LOG_FILE_PER_DATASET}")
print(f"Best stratification results across all datasets saved to {LOG_FILE_ACROSS_DATASETS}")


# visualize
visuals_dir = f"./../experiments/{EXPERIMENT_NAME}/visuals"
os.makedirs(visuals_dir, exist_ok=True)

def jitter(values, strength=0.25):
    return [v + np.random.uniform(-strength, strength) for v in values]

metric_order = df["strat_metric"].unique()
metric_map = {metric: i for i, metric in enumerate(metric_order)}
df["metric_pos"] = df["strat_metric"].map(metric_map)
df["metric_jittered"] = jitter(df["metric_pos"])


dataset_colors = COLOR_MAP["datasets"]
shape_palette = {25: "o", 35: "s", 45: "X"}  # Circle, square, fat cross

plt.rcParams.update({
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12
})
plt.figure(figsize=(10, 5))
for dataset in df["dataset"].unique():
    dataset_df = df[df["dataset"] == dataset]
    for bin_count in dataset_df["num_bins"].unique():
        subset = dataset_df[dataset_df["num_bins"] == bin_count]
        plt.scatter(
            subset["metric_jittered"],
            subset["average_KS"],
            label=f"{dataset} - {bin_count} bins",
            color=dataset_colors.get(dataset, "gray"),
            marker=shape_palette.get(bin_count, "o"),
            s=100,
            alpha=0.7,
            edgecolor="black"
        )

# Final formatting
plt.xticks(list(metric_map.values()), metric_order)
plt.xlabel("Stratification Metric")
plt.ylabel("Average KS Divergence")
# plt.title("KS Divergence Across Seeds, Metrics, Bin Counts, and Datasets")
plt.grid(True, linewidth=0.5)
plt.legend(title="Dataset / Bins", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{visuals_dir}/ks_scatter_customcolor_jittered.png", dpi=300)
plt.show()