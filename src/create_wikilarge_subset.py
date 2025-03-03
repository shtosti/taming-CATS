import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp
from Metrics import Metrics

def load_jsonl(filepath: str) -> list:
    """Load dataset from a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data: list, filepath: str) -> None:
    """Save dataset to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in data:
            source_text = line["source_text"]
            for simplification in line["simplifications"]:
                bleu = Metrics(
                    input_text=simplification["simplification_text"], 
                    source_text=source_text
                    ).compute_bleu()
                bertscore = Metrics(
                    input_text=simplification["simplification_text"], 
                    source_text=source_text
                    ).compute_bertscore()
                
                # Append the scores to the target_metrics
                simplification['target_metrics']['BLEU'] = bleu
                simplification['target_metrics']['BERTScore'] = bertscore

            f.write(json.dumps(line) + "\n")

def extract_metrics(data: list, metrics: list) -> dict:
    """Extract specified metric values from dataset."""
    return {metric: [line["source_metrics"].get(metric, 0) for line in data] for metric in metrics}

def stratified_sampling(data: list, metric_values: dict, metric: str, num_bins=20, subset_size=2000) -> list:
    """Perform stratified sampling based on a single metric."""
    values = np.array(metric_values[metric])
    bins = np.histogram_bin_edges(values, bins=num_bins)
    bin_indices = np.digitize(values, bins)
    df = pd.DataFrame({"data": data, "bin": bin_indices})
    subset = df.groupby("bin", group_keys=False).apply(lambda x: x.sample(frac=subset_size / len(df), random_state=42))
    return subset["data"].tolist()

def stratified_sampling_from_split(data: list, metric, num_bins=20, subset_size=2000):
    """Perform stratified sampling separately for train, valid, and test to maintain proportions."""
    # Split dataset into train, valid, test
    train_data = [line for line in data if line["metadata"]["original_split"] == "train"]
    valid_data = [line for line in data if line["metadata"]["original_split"] == "valid"]
    test_data = [line for line in data if line["metadata"]["original_split"] == "test"]

    train_size = int(subset_size * 0.8)
    valid_size = int(subset_size * 0.1)
    test_size = int(subset_size * 0.1)

    train_metrics = extract_metrics(train_data, [metric])
    valid_metrics = extract_metrics(valid_data, [metric])
    test_metrics = extract_metrics(test_data, [metric])

    def sample_split(split_data: list, split_metrics: dict, split_size: int) -> list:
        """Helper function to perform stratified sampling for each split."""
        if len(split_data) == 0:
            return []  # In case a split has no data (edge case)
        values = np.array(split_metrics[metric])
        bins = np.histogram_bin_edges(values, bins=num_bins)
        bin_indices = np.digitize(values, bins)
        df = pd.DataFrame({"data": split_data, "bin": bin_indices})
        subset = df.groupby("bin", group_keys=False).apply(lambda x: x.sample(frac=split_size / len(df), random_state=42))
        return subset["data"].tolist()

    # Sample from each split
    sampled_train = sample_split(train_data, train_metrics, train_size)
    sampled_valid = sample_split(valid_data, valid_metrics, valid_size)
    sampled_test = sample_split(test_data, test_metrics, test_size)

    return sampled_train + sampled_valid + sampled_test

def plot_final_multi_metric(metric_values: dict, all_subset_metric_values: dict, strat_metrics: list, metrics: list, subset_dir: str) -> None:
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

    plot_filepath = f"{subset_dir}/stratification_metrics.png"
    os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
    plt.savefig(plot_filepath, dpi=400)
    plt.close()

def compute_similarity_scores(full_data: dict, subset_data: dict, metrics: list) -> dict:
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

def rank_stratifications(all_scores: dict) -> dict:
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

def save_ranking_log(ranked_strats: list, subset_dir: str) -> None:
    """Save stratification ranking to a log file."""
    log_filepath = f"{subset_dir}/metric_rank_log.txt"
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    with open(log_filepath, "w", encoding="utf-8") as f:
        f.write("=== Stratification Ranking (Lower Score = More Representative) ===\n")
        for rank, (strat, score) in enumerate(ranked_strats, start=1):
            f.write(f"{rank}. Stratified by {strat}: Score = {score:.4f}\n")

def main():

    subset_size = 3000
    num_bins = 25

    full_dataset_path = "./../data/datasets/wikilarge/dataset.jsonl"
    save_path = "./../data/datasets"
    subset_dir = f"{save_path}/wikilarge_{subset_size}_from_splits"
    metrics = ["char_count", "word_count", "sentence_count", "FKGL", "ARI", "FRE", "Dale-Chall"]
    
    data = load_jsonl(full_dataset_path)
    metric_values = extract_metrics(data, metrics)

    all_subset_metric_values = {}
    all_similarity_scores = {}

    for strat_metric in metrics:
        # TODO if using stratified sampling from corresponding splits
        subset_data = stratified_sampling_from_split(
            data, 
            strat_metric, 
            num_bins=num_bins, 
            subset_size=subset_size
        )

        # # TODO if using simple stratified sampling, regardless of the split
        # subset_data = stratified_sampling(
        #     data, 
        #     metric_values, 
        #     strat_metric, 
        #     num_bins=num_bins, 
        #     subset_size=subset_size
        # )

        subset_metric_values = extract_metrics(subset_data, metrics)

        all_subset_metric_values[strat_metric] = subset_metric_values
        all_similarity_scores[strat_metric] = compute_similarity_scores(metric_values, subset_metric_values, metrics)
    
    ranked_strats = rank_stratifications(all_similarity_scores)

    # Save ranking log
    save_ranking_log(ranked_strats, subset_dir)

    # Select the best stratification method
    best_strat_metric = ranked_strats[0][0]

    # TODO if using stratified sampling from corresponding splits
    best_subset_data = stratified_sampling_from_split(
        data, 
        best_strat_metric, 
        num_bins=num_bins, 
        subset_size=subset_size
    )

    # TODO if using simple stratified sampling, regardless of the split
    # best_subset_data = stratified_sampling(
    #     data, 
    #     metric_values, 
    #     best_strat_metric, 
    #     num_bins=num_bins, 
    #     subset_size=subset_size
    # )

    # Save only the best subset
    best_subset_filepath = f"{subset_dir}/dataset.jsonl"
    save_jsonl(best_subset_data, best_subset_filepath)

    # Generate and save the final plot
    plot_final_multi_metric(metric_values, all_subset_metric_values, metrics, metrics, subset_dir)

    print("\n=== Stratification Ranking (Lower Score = More Representative) ===")
    for rank, (strat, score) in enumerate(ranked_strats, start=1):
        print(f"{rank}. Stratified by {strat}: Score = {score:.4f}")

if __name__ == "__main__":
    main()
