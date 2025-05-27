import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

def load_results(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_color_map(color_map_path):
    with open(color_map_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_model_color(model_name, model_styles):
    return model_styles.get(model_name, {}).get("color", "gray")

def get_model_hatch(model_name, model_styles):
    hatch = model_styles.get(model_name, {}).get("hatches", "solid")
    return "" if hatch == "solid" else hatch

def load_model_info(json_path="data/models.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_model_family(model_name, model_info):

    size_order = {"XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5}

    for family, models in model_info.items():
        for name, details in models.items():
            if name in model_name:
                size = details.get("size", "other")
                rank = size_order.get(size, float('inf'))
                return (family, rank)
    return ("other", float('inf'))

# def get_model_family_map():
#     return {
#         "Llama-3.2-1B-Instruct": ("llama", 1),
#         "Meta-Llama-3-8B-Instruct": ("llama", 2),
#         "Llama-2-13b-chat-hf": ("llama", 3),
#         "Ministral-3b-instruct": ("mistral", 1),
#         "Mistral-7B-Instruct-v0.1": ("mistral", 2),
#         "Qwen2.5-1.5B-Instruct": ("qwen", 1),
#         "Qwen2.5-7B-Instruct": ("qwen", 2),
#         "Qwen2.5-14B-Instruct": ("qwen", 3),
#     }

# def get_model_family(model_name):
#     family_map = get_model_family_map()
#     for key, family in family_map.items():
#         if key in model_name:
#             return family
#     return "other"

def plot_comparison_metrics(results, dataset, control_attr, save_dir, output_prefix, color_map_path):
    model_info = load_model_info()
    color_map = load_color_map(color_map_path)
    model_styles = color_map.get("models", {})

    losses = {
        "MSE": [],
        "MAE": [],
        "std_error": [],
        "var_error": []
    }

    metrics = {
        "SARI": [],
        "COMET": [],
        "BLEU_to_source": [],
        "BLEU_to_ref": [],
        "BERTScore_to_source": [],
        "BERTScore_to_ref": []
    }

    errors = {metric: {"lower": [], "upper": []} for metric in metrics}
    models = []

    for model, model_data in results.items():
        if dataset in model_data and control_attr in model_data[dataset]:
            entry = model_data[dataset][control_attr]
            mean_metrics = entry.get("mean_metrics", {})
            losses_data = entry.get("losses", {})

            try:
                for metric in metrics:
                    mean = mean_metrics[metric]["mean"]
                    ci_low = mean_metrics[metric]["ci_lower"]
                    ci_up = mean_metrics[metric]["ci_upper"]

                    metrics[metric].append(mean)
                    errors[metric]["lower"].append(mean - ci_low)
                    errors[metric]["upper"].append(ci_up - mean)

                for loss in losses:
                    losses[loss].append(losses_data.get(loss, np.nan))

                models.append(model)
            except KeyError as e:
                print(f"Skipping model '{model}' with dataset '{dataset}' due to missing metric key: {e}")
                continue

    if not models:
        print(f"No valid data for {dataset}/{control_attr}")
        return
    
    sorted_indices = sorted(
        range(len(models)),
        key=lambda i: (
            get_model_family(models[i], model_info)[0],  # family
            get_model_family(models[i], model_info)[1],  # size ranked
            models[i]                                    # fallback sort by name
        )
    )
    
    models = [models[i] for i in sorted_indices]
    for metric in metrics:
        metrics[metric] = [metrics[metric][i] for i in sorted_indices]
        errors[metric]["lower"] = [errors[metric]["lower"][i] for i in sorted_indices]
        errors[metric]["upper"] = [errors[metric]["upper"][i] for i in sorted_indices]
    for loss in losses:
        losses[loss] = [losses[loss][i] for i in sorted_indices]

    total_metrics = list(metrics.keys())
    total_losses = list(losses.keys())
    total_items = total_metrics + total_losses
    total_plots = len(total_items)

    ncols = 2
    nrows = (total_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), sharex=False)
    fig.suptitle(f"{control_attr} on {dataset}", fontsize=18)
    axes = axes.flatten()

    # Plot metrics with error bars
    for i, metric in enumerate(total_metrics):
        means = metrics[metric]
        lower = errors[metric]["lower"]
        upper = errors[metric]["upper"]
        yerr = [lower, upper]

        ax = axes[i]
        for j, model in enumerate(models):
            color = get_model_color(model, model_styles)
            hatch = get_model_hatch(model, model_styles)
            ax.bar(j, means[j], yerr=[[lower[j]], [upper[j]]],
                   color=color, edgecolor='black', hatch=hatch,
                   capsize=10)
        ax.set_title(metric)
        # ax.set_xticks(np.arange(len(models)))
        # ax.set_xticklabels(models, rotation=90)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Plot loss values
    for j, loss in enumerate(total_losses, start=len(total_metrics)):
        values = losses[loss]
        ax = axes[j]
        for k, model in enumerate(models):
            color = get_model_color(model, model_styles)
            hatch = get_model_hatch(model, model_styles)
            ax.bar(k, values[k], color=color, edgecolor='black', hatch=hatch)
        ax.set_title(loss)
        # ax.set_xticks(np.arange(len(models)))
        # ax.set_xticklabels(models, rotation=90)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    for ax in axes[total_plots:]:
        ax.set_visible(False)

    # Add legend
    legend_handles = []
    seen = set()
    for model in models:
        color = get_model_color(model, model_styles)
        hatch = get_model_hatch(model, model_styles)
        key = (color, hatch)
        if key not in seen:
            seen.add(key)
            label = model
            rect = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', hatch=hatch, label=label, linewidth=1)
            legend_handles.append(rect)

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=4,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
        handlelength=2.5,
        handleheight=2,
        fontsize=10
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plot_path = os.path.join(save_dir, f"{output_prefix}_metrics_losses.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_file", type=str, required=True, help="Path to JSON file with all results for all models.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name to compare")
    parser.add_argument("--control_attr", type=str, required=True, help="Metric key to compare")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the plots")
    parser.add_argument("--color_map_path", type=str, default="data/colormap/color_map.json", help="Path to color and hatch map JSON")
    parser.add_argument("--user_prompt_id", type=str, required=True, help="User prompt ID")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    output_prefix = f"{args.control_attr}_{args.dataset}_{args.user_prompt_id}"

    all_results = load_results(args.summary_file)
    plot_comparison_metrics(all_results, args.dataset, args.control_attr, args.save_dir, output_prefix, args.color_map_path)

if __name__ == "__main__":
    main()
