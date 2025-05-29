import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from helpers.utils import get_correlation_data
import pandas as pd
import scipy.stats as stats
from collections import defaultdict

def load_json(file_path: str):
    """Load JSON from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def is_source_metric(args):
    if args.metric_key in ["FRE", "FKGL", "ARI", "DALE-CHALL", "Dale-Chall"]:
        return True
    return False

def load_predictions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def average_predictions_across_runs(json_files):
    """Loads multiple JSON files and averages prediction metrics per sample."""
    all_runs = [load_json(file) for file in json_files]

    assert all(len(run) == len(all_runs[0]) for run in all_runs), "All files must have the same number of samples"

    averaged_predictions = []
    for i in range(len(all_runs[0])):
        averaged_item = {
            "source_metrics": all_runs[0][i]["source_metrics"],
            "reference_metrics": all_runs[0][i]["reference_metrics"],
            "prediction_metrics": defaultdict(list)
        }
        for run in all_runs:
            for key, val in run[i]["prediction_metrics"].items():
                averaged_item["prediction_metrics"][key].append(val)
        averaged_item["prediction_metrics"] = {
            key: np.mean(vals) for key, vals in averaged_item["prediction_metrics"].items()
        }
        averaged_predictions.append(averaged_item)

    return averaged_predictions

def extract_metric_values(predictions, metric_key, use_source=False):

    source_vals = []
    reference_vals = []
    prediction_vals = []

    for item in predictions:
        if use_source:
            # print(f"Source metrics: {item['source_metrics']}")
            source_val = item["source_metrics"].get(metric_key)
        reference_val = item["reference_metrics"].get(metric_key)
        prediction_val = item["prediction_metrics"].get(metric_key)

        if reference_val is not None and prediction_val is not None:
            if use_source:
                source_vals.append(source_val)
            reference_vals.append(reference_val)
            prediction_vals.append(prediction_val)

    return source_vals, reference_vals, prediction_vals

def compute_mean_losses(reference_vals, prediction_vals):
    mse = mean_squared_error(reference_vals, prediction_vals)
    mae = mean_absolute_error(reference_vals, prediction_vals)
    return mse, mae

def compute_per_sample_losses(reference_vals, prediction_vals):
    mse = []
    mae = []
    real_errors = []
    for ref, pred in zip(reference_vals, prediction_vals):
        mse.append(mean_squared_error([ref], [pred]))
        mae.append(mean_absolute_error([ref], [pred]))
        real_errors.append(pred - ref)
    return mse, mae, real_errors

def compute_mean_metrics(predictions):
    """Compute mean and 95% confidence intervals for all comparison metrics."""

    def bootstrap_ci(data, n_bootstraps=1000, ci=0.95):
        boot_means = [
            np.mean(np.random.choice(data, size=len(data), replace=True))
            for _ in range(n_bootstraps)
        ]
        lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
        upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
        return np.mean(data), lower, upper

    BLEU_to_source, BLEU_to_ref = [], []
    BERTScore_to_source, BERTScore_to_ref = [], []
    COMET, SARI = [], []

    for item in predictions:
        BLEU_to_source.append(item["prediction_metrics"].get("BLEU"))
        BLEU_to_ref.append(item["prediction_metrics"].get("BLEU_ref"))
        BERTScore_to_source.append(item["prediction_metrics"].get("BERTScore"))
        BERTScore_to_ref.append(item["prediction_metrics"].get("BERTScore_ref"))
        COMET.append(item["prediction_metrics"].get("COMET"))
        SARI.append(item["prediction_metrics"].get("SARI"))

    metrics = {
        "BLEU_to_source": BLEU_to_source,
        "BLEU_to_ref": BLEU_to_ref,
        "BERTScore_to_source": BERTScore_to_source,
        "BERTScore_to_ref": BERTScore_to_ref,
        "COMET": COMET,
        "SARI": SARI,
    }

    results = {}
    for name, values in metrics.items():
        if not values:
            continue
        mean, lower, upper = bootstrap_ci(values)
        delta_lower = mean - lower
        delta_upper = upper - mean
        results[name] = {
            "mean": mean,
            "ci_lower": lower,
            "ci_upper": upper,
            "ci_range_minus": delta_lower,
            "ci_range_plus": delta_upper,
            "formatted": f"{name}: {mean:.2f} (+{delta_upper:.2f} -{delta_lower:.2f})"
        }

    return results

def plot_metric_scatter(source_vals, reference_vals, prediction_vals, metric_key, output_dir, use_source=False):

    x = list(range(len(reference_vals)))

    plt.figure(figsize=(6, 5))

    # Cap outliers
    source_vals = cap_outliers(source_vals) if use_source else []
    reference_vals = cap_outliers(reference_vals)
    prediction_vals = cap_outliers(prediction_vals)

    if use_source:
        plt.scatter(x, source_vals, color="orchid", label="Source", alpha=0.7)
        source_trend = np.poly1d(np.polyfit(x, source_vals, 1))
        plt.plot(x, source_trend(x), color="orchid", linestyle="-", linewidth=2)

    plt.scatter(x, reference_vals, color="darkorange", label="Reference", alpha=0.7)
    plt.scatter(x, prediction_vals, color="seagreen", label="Prediction", alpha=0.7)
    reference_trend = np.poly1d(np.polyfit(x, reference_vals, 1))
    prediction_trend = np.poly1d(np.polyfit(x, prediction_vals, 1))
    plt.plot(x, reference_trend(x), color="darkorange", linestyle="-", linewidth=2)
    plt.plot(x, prediction_trend(x), color="seagreen", linestyle="-", linewidth=2)

    # plt.title(f"{metric_key}")
    plt.xlabel("idx")
    plt.ylabel(metric_key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric_key}_scatter_plot.png", bbox_inches='tight', dpi=400)
    print(f"Scatter plot saved as {metric_key}_scatter.png")

def polyfit_plot(ax, x, y, color):
    x_np = np.array(x, dtype=np.float32)
    y_np = np.array(y, dtype=np.float32)
    mask = ~np.isnan(x_np) & ~np.isnan(y_np)
    if np.sum(mask) >= 2:
        coeffs = np.polyfit(x_np[mask], y_np[mask], 1)
        trend = np.poly1d(coeffs)
        sorted_x = np.sort(x_np[mask])
        ax.plot(sorted_x, trend(sorted_x), color=color, linestyle="--", linewidth=2)

def plot_ctrl_attr_vs_metrics(predictions, metric_key, output_dir):

    control_attr_vals = []
    BLEU_to_source, BLEU_to_ref = [], []
    BERTScore_to_source, BERTScore_to_ref = [], []
    COMET, SARI = [], []

    for item in predictions:
        control_attr_vals.append(item["prediction_metrics"].get(metric_key))
        BLEU_to_source.append(item["prediction_metrics"].get("BLEU"))
        BLEU_to_ref.append(item["prediction_metrics"].get("BLEU_ref"))
        BERTScore_to_source.append(item["prediction_metrics"].get("BERTScore"))
        BERTScore_to_ref.append(item["prediction_metrics"].get("BERTScore_ref"))
        COMET.append(item["prediction_metrics"].get("COMET"))
        SARI.append(item["prediction_metrics"].get("SARI"))

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    metric_groups = [
        (BLEU_to_source, "BLEU to Source", "skyblue"),
        (BLEU_to_ref, "BLEU to Reference", "skyblue"),
        (COMET, "COMET", "skyblue"),
        (BERTScore_to_source, "BERTScore to Source", "skyblue"),
        (BERTScore_to_ref, "BERTScore to Reference", "skyblue"),
        (SARI, "SARI", "skyblue")
    ]

    # get correlation data
    for metric_vals, title, color in metric_groups:
        correlation_data = get_correlation_data(control_attr_vals, metric_vals)
        corr_coeff = correlation_data["correlation_coefficient"]
        significance_level = correlation_data["significance_level"]
        corr_strength = correlation_data["correlation_strength"]
        significance_level = correlation_data["significance_level"]

    for ax, (metric_vals, title, color) in zip(axs.flat, metric_groups):
        ax.scatter(control_attr_vals, metric_vals, color=color, alpha=0.7, label=title)
        polyfit_plot(ax, control_attr_vals, metric_vals, color="black")

        # Compute correlation
        correlation_data = get_correlation_data(control_attr_vals, metric_vals)
        corr_coeff = correlation_data["correlation_coefficient"]
        p_value = correlation_data["p_value"]
        significance_level = correlation_data["significance_level"]
        corr_strength = correlation_data["correlation_strength"]

        ax.set_title(significance_level)
        ax.set_xlabel(metric_key)
        ax.set_ylabel(title)
        ax.text(0.05, 0.85, f"p={p_value:.2f}\nr={corr_coeff:.2f} ({corr_strength})", transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric_key}_vs_metrics.png", bbox_inches='tight', dpi=400)
    print(f"Control attribute vs metrics plot saved as {metric_key}_ctrl_attr_vs_metrics.png")

def cap_outliers(y_vals, lower_pct=1, upper_pct=99):
    if len(y_vals) == 0:
        return y_vals
    y_np = np.array(y_vals, dtype=np.float32)
    lower = np.percentile(y_np, lower_pct)
    upper = np.percentile(y_np, upper_pct)
    return np.clip(y_np, lower, upper)

def plot_errors_vs_metrics(predictions, metric_key_mapped, metric_key, output_dir):

    def extract_errors_and_metric(predictions, metric_name, loss_key):
        metric_vals, abs_errors, sq_errors = [], [], []
        for item in predictions:
            loss_data = item.get(f"{loss_key}_losses")
            metric_val = item["prediction_metrics"].get(metric_name)
            if loss_data and metric_val is not None:
                metric_vals.append(metric_val)
                abs_errors.append(loss_data["absolute_error"])
                sq_errors.append(loss_data["squared_error"])
        return metric_vals, abs_errors, sq_errors

    metrics = ["BLEU", "BLEU_ref", "COMET", "BERTScore", "BERTScore_ref", "SARI"]
    labels = ["BLEU to Source", "BLEU to Reference", "COMET", "BERTScore to Source", "BERTScore to Reference", "SARI"]

    fig_abs, axs_abs = plt.subplots(2, 3, figsize=(10, 6))
    fig_sq, axs_sq = plt.subplots(2, 3, figsize=(10, 6))

    for (ax_abs, ax_sq, metric_name, label) in zip(axs_abs.flat, axs_sq.flat, metrics, labels):
        x_vals, abs_errors, sq_errors = extract_errors_and_metric(predictions, metric_name, metric_key)

        abs_errors_capped = cap_outliers(abs_errors)
        sq_errors_capped = cap_outliers(sq_errors)

        # compute correlations
        correlation_data_abs = get_correlation_data(x_vals, abs_errors_capped)
        correlation_data_sq = get_correlation_data(x_vals, sq_errors_capped)
        corr_coeff_abs = correlation_data_abs["correlation_coefficient"]
        p_value_abs = correlation_data_abs["p_value"]
        significance_level_abs = correlation_data_abs["significance_level"]
        corr_strength_abs = correlation_data_abs["correlation_strength"]
        corr_coeff_sq = correlation_data_sq["correlation_coefficient"]
        p_value_sq = correlation_data_sq["p_value"]
        significance_level_sq = correlation_data_sq["significance_level"]
        corr_strength_sq = correlation_data_sq["correlation_strength"]

        ax_abs.scatter(x_vals, abs_errors_capped, alpha=0.7, color="skyblue")
        polyfit_plot(ax_abs, x_vals, abs_errors_capped, color="black")
        ax_abs.set_xlabel(label)
        ax_abs.set_ylabel(f"Absolute Error ({metric_key_mapped})")
        ax_abs.set_title(f"{significance_level_abs}")
        ax_abs.text(0.05, 0.85, f"p={p_value_abs:.2f}\nr={corr_coeff_abs:.2f} ({corr_strength_abs})", transform=ax_abs.transAxes, fontsize=10)

        ax_sq.scatter(x_vals, sq_errors_capped, alpha=0.7, color="skyblue")
        polyfit_plot(ax_sq, x_vals, sq_errors_capped, color="black")
        ax_sq.set_title(f"{significance_level_sq}")
        ax_sq.set_xlabel(label)
        ax_sq.set_ylabel(f"Squared Error ({metric_key_mapped})")
        ax_sq.text(0.05, 0.85, f"p={p_value_sq:.2f}\nr={corr_coeff_sq:.2f} ({corr_strength_sq})", transform=ax_sq.transAxes, fontsize=10)

    fig_abs.tight_layout()
    fig_sq.tight_layout()

    fig_abs.savefig(f"{output_dir}/{metric_key_mapped}_abs_error_vs_metrics.png", bbox_inches='tight', dpi=400)
    fig_sq.savefig(f"{output_dir}/{metric_key_mapped}_sq_error_vs_metrics.png", bbox_inches='tight', dpi=400)
    print(f"Error-vs-metrics plots saved.")

def plot_error_std_vs_reference(reference_vals, real_errors, metric_key, output_dir, num_bins=10):
    import pandas as pd

    # Create DataFrame
    df = pd.DataFrame({
        "reference": reference_vals,
        "error": real_errors
    })
    df["bin"] = pd.qcut(df["reference"], q=num_bins, duplicates='drop')
    std_by_bin = df.groupby("bin")["error"].std()
    bin_labels = [f"{interval.left:.1f}–{interval.right:.1f}" for interval in std_by_bin.index]

    plt.figure(figsize=(5, 4))
    plt.bar(bin_labels, std_by_bin.values, color="skyblue", edgecolor="black")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("error std")
    plt.xlabel(f"{metric_key}")
    # plt.title(f"Error Variability by Reference {metric_key}")
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle="--", alpha=0.5)
    plt.savefig(f"{output_dir}/{metric_key}_error_std_by_ref_bin.png", bbox_inches='tight', dpi=400)
    print(f"STD of error by reference bin saved as {metric_key}_error_std_by_ref_bin.png")

def plot_error_std_binned(reference_vals, real_errors, metric_key, output_dir, num_bins=25):
    mean_error = np.mean(real_errors)

    plt.figure(figsize=(5, 4))
    sns.histplot(
        real_errors, 
        bins=num_bins, 
        kde=True, 
        line_kws={"color":"royalblue", "linewidth": 1},
        color="skyblue", 
        edgecolor="black"
        )
    plt.axvline(mean_error, color='black', linestyle='--', label=f'Mean Error: {mean_error:.2f}')
    plt.xlabel("Standard deviation of errors")
    plt.ylabel("Count by bin")
    # horizontal grid
    plt.grid(axis='y', linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric_key}_error_std_binned.png", bbox_inches='tight', dpi=400)

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_file", type=str, required=True, help="Path to JSON file with predictions")
    parser.add_argument("--input_files", type=str, nargs='+', required=True, help="Path to JSON file(s) with predictions")
    parser.add_argument("--metric_key", type=str, required=True, help="Metric to visualize")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save plots")
    parser.add_argument("--metric_mapping", type=str, required=True)
    # TODO add
    parser.add_argument("--model_name", type=str, required=True, help="Name of the evaluated model")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--user_prompt_id", type=str, required=True, help="User prompt ID")
    parser.add_argument("--summary_file", type=str, default="output/sft_results/all_results.json", help="Path to save overall summary results")

    return parser.parse_args()

def main():

    print("--- Running evaluation...")

    args = parse_args()
    print("Input files:\t", args.input_files)
    print("Metric:\t", args.metric_key)

    use_source = is_source_metric(args) # source vals used
    metric_mapping = load_json(args.metric_mapping)
    metric_key_mapped = metric_mapping[args.metric_key]

    # predictions = load_predictions(args.input_file) 
    predictions = average_predictions_across_runs(args.input_files)
    with open(f"{args.output_dir}/output_averaged.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)

    source_vals, reference_vals, prediction_vals = extract_metric_values(predictions, metric_key_mapped, use_source=use_source)

    mse, mae = compute_mean_losses(reference_vals, prediction_vals)
    per_sample_mse, per_sample_mae, per_sample_real_loss = compute_per_sample_losses(reference_vals, prediction_vals)
    std_error = np.std(per_sample_real_loss)
    var_error = np.var(per_sample_real_loss)

    print(f"--- {metric_key_mapped} losses:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Standard Deviation of Errors: {std_error}")
    print(f"Variance of Errors: {var_error}")
    print()

    loss_output = {
        "metric": args.metric_key,
        "MSE": mse,
        "MAE": mae,
        "std_error": std_error,
        "var_error": var_error
    }

    with open(f"{args.output_dir}/stats.json", "w") as f:
        json.dump(loss_output, f, indent=4)
    print(f"Saved loss values to stats.json")

    for i, item in enumerate(predictions):
        if i < len(reference_vals):
            item[f"{args.metric_key}_losses"] = {
                "reference": reference_vals[i],
                "prediction": prediction_vals[i],
                "squared_error": per_sample_mse[i],
                "absolute_error": per_sample_mae[i],
                "real_loss": per_sample_real_loss[i]
            }

    with open(f"{args.output_dir}/output_averaged.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)
    print(f"Updated averaged predictions file with per-sample losses: {args.output_dir}/output_averaged.json")

    plot_metric_scatter(source_vals, reference_vals, prediction_vals, metric_key_mapped, args.output_dir, use_source=use_source)
    plot_ctrl_attr_vs_metrics(predictions, metric_key_mapped, args.output_dir)
    plot_errors_vs_metrics(predictions, metric_key_mapped, args.metric_key, args.output_dir)
    plot_error_std_vs_reference(reference_vals, per_sample_real_loss, metric_key_mapped, args.output_dir)
    plot_error_std_binned(reference_vals, per_sample_real_loss, metric_key_mapped, args.output_dir)
    print(f"Plots saved to {args.output_dir}")


    print(f"\nAll plots saved to {args.output_dir}")

    mean_metrics = compute_mean_metrics(predictions)
    print(f"\n--- Mean metrics:")
    for name, values in mean_metrics.items():
        print(f"{name}: {values['formatted']}")
    print()
    try:
        with open(args.summary_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}

    # nest by model -> dataset -> metric
    model = args.model_name
    dataset = args.dataset
    metric = args.metric_key

    if model not in all_results:
        all_results[model] = {}
    if dataset not in all_results[model]:
        all_results[model][dataset] = {}

    all_results[model][dataset][metric] = {
        "losses": {
            "MSE": mse,
            "MAE": mae,
            "std_error": std_error,
            "var_error": var_error
        },
        "mean_metrics": mean_metrics
    }

    # Write back to the summary file
    with open(args.summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(f"Saved evaluation summary to {args.summary_file}")


    print("\n--- Evaluation complete.")



if __name__ == "__main__":
    main()