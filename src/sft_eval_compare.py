import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
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
            if name.lower() in model_name.lower():
                size = details.get("size", "other")
                rank = size_order.get(size, float('inf'))
                return (family, rank)
    return ("other", float('inf'))

def plot_comparison_metrics(results, dataset, control_attr, save_dir, output_prefix, color_map_path):
    model_info = load_model_info()
    color_map = load_color_map(color_map_path)
    model_styles = color_map.get("models", {})

    losses = {
        "MSE": [],
        "MAE": [],
        # "std_error": [],
        # "var_error": []
    }

    metrics = {
        "BLEU_to_source": [],
        "BLEU_to_ref": [],
        "BERTScore_to_source": [],
        "BERTScore_to_ref": [],
        "SARI": [],
        "COMET": [],
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

    nrows = 2
    ncols = (total_plots + nrows - 1) // nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows), sharex=False)
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
        ax.set_title(metric, fontsize=16)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='y', labelsize=16)
        # TODO force consistent y‐ranges
        if metric in ("BLEU_to_source", "BLEU_to_ref", "SARI"):
            ax.set_ylim(0, 100)
        elif metric in ("BERTScore_to_source", "BERTScore_to_ref", "COMET"):
            ax.set_ylim(0, 1.0)



    for j, loss in enumerate(total_losses, start=len(total_metrics)):
        values = losses[loss]
        ax = axes[j]
        for k, model in enumerate(models):
            color = get_model_color(model, model_styles)
            hatch = get_model_hatch(model, model_styles)
            ax.bar(k, values[k], color=color, edgecolor='black', hatch=hatch)
        ax.set_title(loss, fontsize=16)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='y', labelsize=16)

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


    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{output_prefix}_metrics_losses.png")
    fig.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to {plot_path}")

    # legend as separate file
    fig_leg = plt.figure(figsize=(10,1.5))
    fig_leg.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        loc='center',
        ncol=5,
        frameon=True,
        handlelength=2.5,
        handleheight=2,
        fontsize=12
    )
    fig_leg.subplots_adjust(left=0, right=1, top=1, bottom=0)
    legend_path = os.path.join(save_dir, f"{output_prefix}_legend.png")
    fig_leg.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close(fig_leg)


def plot_pairwise_correlations(results, dataset, control_attr, save_dir, output_prefix, color_map_path):
    rows = []
    for model, model_data in results.items():
        if dataset not in model_data or control_attr not in model_data[dataset]:
            continue
        entry = model_data[dataset][control_attr]
        mm = entry["mean_metrics"]
        losses = entry["losses"]

        row = {
            "model": model,
            "SARI": mm["SARI"]["mean"],
            "COMET": round(mm["COMET"]["mean"],2),
            "BERT_to_src": round(mm["BERTScore_to_source"]["mean"],2),
            "BERT_to_ref": round(mm["BERTScore_to_ref"]["mean"],2),
            "BLEU_to_src": round(mm["BLEU_to_source"]["mean"],2),
            "BLEU_to_ref": round(mm["BLEU_to_ref"]["mean"],2),
            "MSE": losses["MSE"],
            "MAE": losses["MAE"],
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")

    # -------------- heatmap ---------------
    corr = df.corr(method="pearson")
    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(
            corr,
            # mask=mask,
            annot=True,
            fmt=".2f",
            # cmap="BrBG",
            cmap="PRGn",
            center=0,
            annot_kws={"size": 12},
            cbar_kws={"shrink": .8},
            square=True,
        )
    ax.tick_params(axis='x', labelsize=14, rotation=90)
    ax.tick_params(axis='y', labelsize=14, rotation=0)
    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, f"{output_prefix}_corr_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    # -------------- pairplot ---------------
    subset = ["SARI",
            "COMET",
            "BERT_to_src",
            "BERT_to_ref",
            "BLEU_to_src",
            "BLEU_to_ref",
            "MSE",
            # "MAE"
            ]
    g = sns.pairplot(
        df[subset],
        kind="reg",
        plot_kws={"line_kws":{"color":"orchid"}, "scatter_kws":{"s":30, "alpha":0.6}},
        # diag_kind="hist",
        diag_kind=None,
        diag_kws={"bins":10, "edgecolor":"k"},
    )

    for ax in g.axes.flatten():
        if ax is not None:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune=None))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune=None))
            # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            ax.tick_params(axis='x', rotation=45, labelsize=14)
            ax.tick_params(axis='y', rotation=0, labelsize=14)
            ax.xaxis.label.set_size(16)
            ax.yaxis.label.set_size(16)


    g.fig.set_size_inches(15, 15)
    # plt.suptitle("Pairwise Scatter + Regression", y=1.02)
    pairplot_path = os.path.join(save_dir, f"{output_prefix}_pairplot.png")
    plt.tight_layout()
    plt.savefig(pairplot_path, dpi=300)
    plt.close()

    print(f"Saved correlation heatmap to {heatmap_path}")
    print(f"Saved pairplot to {pairplot_path}")


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

    plot_comparison_metrics(
                all_results, 
                args.dataset, 
                args.control_attr, 
                args.save_dir, 
                output_prefix, 
                args.color_map_path
                )

    plot_pairwise_correlations(
                all_results,
                args.dataset,
                args.control_attr,
                args.save_dir,
                output_prefix,
                args.color_map_path
    )


if __name__ == "__main__":
    main()
