import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp

def load_jsonl(filepath):
    """Load dataset from a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, filepath):
    """Save dataset to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

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

def plot_final_multi_metric(metric_values, all_subset_metric_values, strat_metrics, metrics, save_path, subset_size):
    """Create a single multi-plot where each row represents a stratification metric."""
    num_strat_metrics = len(strat_metrics)
    num_metrics = len(metrics)

    fig, axes = plt.subplots(num_strat_metrics, num_metrics, figsize=(4 * num_metrics, 4 * num_strat_metrics), constrained_layout=True)

    for row, strat_metric in enumerate(strat_metrics):
        subset_metric_values = all_subset_metric_values[strat_metric]
        
        axes[row, 0].annotate(
            f"Stratified by {strat_metric}",
            xy=(0, 0.5),
            xytext=(-axes[row, 0].yaxis.labelpad - 30, 0),
            xycoords=axes[row, 0].yaxis.label,
            textcoords="offset points",
            size=12,
            ha="right",
            va="center",
            rotation=90,
            fontweight="bold"
        )
        
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            sns.kdeplot(metric_values[metric], color="blue", label="Full Dataset", ax=ax)
            sns.kdeplot(subset_metric_values[metric], color="red", label="Subset", ax=ax)
            ax.set_title(f"{metric}")
            
            if col == 0:
                ax.set_ylabel("Density")
            if row == num_strat_metrics - 1:
                ax.set_xlabel(metric)
            if row == 0 and col == num_metrics - 1:
                ax.legend()

    plot_filepath = f"{save_path}/wikilarge_{subset_size}/stratification_metrics.png"
    os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
    plt.savefig(plot_filepath, dpi=400)
    plt.close()

def compute_similarity_scores(full_data, subset_data, metrics):
    """Compute JSD, EMD, and KS scores to quantify similarity between distributions."""
    scores = {metric: {"JSD": None, "EMD": None, "KS": None} for metric in metrics}
    
    for metric in metrics:
        full_dist = np.array(full_data[metric])
        subset_dist = np.array(subset_data[metric])

        bins = np.histogram_bin_edges(np.concatenate([full_dist, subset_dist]), bins=50)
        full_hist, _ = np.histogram(full_dist, bins=bins, density=True)
        subset_hist, _ = np.histogram(subset_dist, bins=bins, density=True)
        jsd = jensenshannon(full_hist, subset_hist)

        emd = wasserstein_distance(full_dist, subset_dist)

        ks_stat, _ = ks_2samp(full_dist, subset_dist)

        scores[metric] = {"JSD": jsd, "EMD": emd, "KS": ks_stat}

    return scores

def rank_stratifications(all_scores):
    """Aggregate similarity scores across all metrics for each stratification method."""
    strat_ranking = {}
    
    for strat_metric, metric_scores in all_scores.items():
        total_jsd = np.mean([metric_scores[m]["JSD"] for m in metric_scores])
        total_emd = np.mean([metric_scores[m]["EMD"] for m in metric_scores])
        total_ks = np.mean([metric_scores[m]["KS"] for m in metric_scores])

        final_score = (total_jsd + total_emd + total_ks) / 3
        strat_ranking[strat_metric] = final_score

    ranked_strats = sorted(strat_ranking.items(), key=lambda x: x[1])
    return ranked_strats

def save_ranking_log(ranked_strats, save_path, subset_size):
    """Save stratification ranking to a log file."""
    log_filepath = f"{save_path}/wikilarge_{subset_size}/metric_rank_log.txt"
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    with open(log_filepath, "w", encoding="utf-8") as f:
        f.write("=== Stratification Ranking (Lower Score = More Representative) ===\n")
        for rank, (strat, score) in enumerate(ranked_strats, start=1):
            f.write(f"{rank}. Stratified by {strat}: Score = {score:.4f}\n")

def main():

    subset_size = 10000
    num_bins = 25

    full_dataset_path = "./../data/datasets/wikilarge/dataset.jsonl"
    save_path = "./../data/datasets"
    subset_dir = f"{save_path}/wikilarge_{subset_size}"
    metrics = ["char_count", "word_count", "sentence_count", "FKGL", "ARI", "FRE", "Dale-Chall"]
    
    data = load_jsonl(full_dataset_path)
    metric_values = extract_metrics(data, metrics)

    all_subset_metric_values = {}
    all_similarity_scores = {}

    for strat_metric in metrics:
        subset_data = stratified_sampling(
            data, 
            metric_values, 
            strat_metric, 
            num_bins=num_bins, 
            subset_size=subset_size
        )
        subset_metric_values = extract_metrics(subset_data, metrics)

        all_subset_metric_values[strat_metric] = subset_metric_values
        all_similarity_scores[strat_metric] = compute_similarity_scores(metric_values, subset_metric_values, metrics)
    
    ranked_strats = rank_stratifications(all_similarity_scores)

    # Save ranking log
    save_ranking_log(ranked_strats, save_path, subset_size)

    # Select the best stratification method
    best_strat_metric = ranked_strats[0][0]
    best_subset_data = stratified_sampling(
        data, 
        metric_values, 
        best_strat_metric, 
        num_bins=num_bins, 
        subset_size=subset_size
    )

    # Save only the best subset
    best_subset_filepath = f"{subset_dir}/dataset.jsonl"
    save_jsonl(best_subset_data, best_subset_filepath)

    # Generate and save the final plot
    plot_final_multi_metric(metric_values, all_subset_metric_values, metrics, metrics, save_path, subset_size)

    print("\n=== Stratification Ranking (Lower Score = More Representative) ===")
    for rank, (strat, score) in enumerate(ranked_strats, start=1):
        print(f"{rank}. Stratified by {strat}: Score = {score:.4f}")

if __name__ == "__main__":
    main()
