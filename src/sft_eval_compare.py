import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_comparison_metrics(results, dataset, control_attr, save_dir, output_prefix):
    sari = []
    comet = []
    BLEU_to_source = []
    BLEU_to_ref = []
    BERTScore_to_source = []
    BERTScore_to_ref = []
    models = []
    for model in results.keys():
        models.append(model)
        model_data = results[model]
        if dataset in model_data and control_attr in model_data[dataset]:
            mean_metrics = model_data[dataset][control_attr]["mean_metrics"]
            sari.append(mean_metrics["SARI"]["mean"])
            comet.append(mean_metrics["COMET"]["mean"])
            BLEU_to_source.append(mean_metrics["BLEU_to_source"]["mean"])
            BLEU_to_ref.append(mean_metrics["BLEU_to_ref"]["mean"])
            BERTScore_to_source.append(mean_metrics["BERTScore_to_source"]["mean"])
            BERTScore_to_ref.append(mean_metrics["BERTScore_to_ref"]["mean"])

    print(models)
    print(sari)
    print(comet)

    # create a plot with subplots, one for each metric
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 6, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.bar(models, sari, color='blue')
    plt.ylabel('SARI')
    plt.xticks(rotation=90)

    plt.subplot(1, 6, 2)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.bar(models, comet, color='orange')
    plt.ylabel('COMET')
    plt.xticks(rotation=90)

    plt.subplot(1, 6, 3)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.bar(models, BLEU_to_source, color='green')
    plt.ylabel('BLEU to Source')
    plt.xticks(rotation=90)

    plt.subplot(1, 6, 4)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.bar(models, BLEU_to_ref, color='red')
    plt.ylabel('BLEU to Reference')
    plt.xticks(rotation=90)
    
    plt.subplot(1, 6, 5)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.bar(models, BERTScore_to_source, color='purple')
    plt.ylabel('BERTScore to Source')
    plt.xticks(rotation=90)

    plt.subplot(1, 6, 6)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.bar(models, BERTScore_to_ref, color='pink')
    plt.ylabel('BERTScore to Reference')
    plt.xticks(rotation=90)

    plt.suptitle(f"{control_attr} on {dataset}", fontsize=16)


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{output_prefix}_metrics.png"))
    plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_file", type=str, required=True, help="Path to JSON file with all results for all models.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name to compare")
    parser.add_argument("--control_attr", type=str, required=True, help="Metric key to compare")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the plots")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    output_prefix = f"{args.control_attr}_{args.dataset}"

    all_results = load_results(args.summary_file)

    plot_comparison_metrics(all_results, args.dataset, args.control_attr, args.save_dir, output_prefix)
    print(f"Plots saved in {args.save_dir}")


if __name__ == "__main__":
    main()
