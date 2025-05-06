import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

def extract_metric_values(predictions, metric_key, use_source=False):

    source_vals = []
    reference_vals = []
    prediction_vals = []

    for item in predictions:
        # Extract from nested metrics
        if use_source:
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
    for ref, pred in zip(reference_vals, prediction_vals):
        mse.append(mean_squared_error([ref], [pred]))
        mae.append(mean_absolute_error([ref], [pred]))
    return mse, mae

def plot_metric_scatter(source_vals, reference_vals, prediction_vals, metric_key, output_dir, use_source=False):

    x = list(range(len(reference_vals)))

    plt.figure(figsize=(6, 5))

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

def plot_metric_lines(source_vals, reference_vals, prediction_vals, metric_key, output_dir, use_source=False):

    x = list(range(len(prediction_vals)))

    plt.figure(figsize=(6, 5))

    if use_source:
        plt.plot(x, source_vals, color="orchid", label="Source", alpha=0.7)

    plt.plot(x, reference_vals, color="darkorange", label="Reference", alpha=0.7)
    plt.plot(x, prediction_vals, color="seagreen", label="Prediction", alpha=0.7)

    # plt.title(f"{metric_key}")
    plt.xlabel("idx")
    plt.ylabel(metric_key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric_key}_line.png", bbox_inches='tight', dpi=400)
    print(f"Line plot saved as {metric_key}_line.png")

def safe_polyfit_plot(ax, x, y, color):
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

    fig, axs = plt.subplots(3, 2, figsize=(8, 10))
    metric_groups = [
        (BLEU_to_source, "BLEU to Source", "skyblue"),
        (BLEU_to_ref, "BLEU to Reference", "skyblue"),
        (BERTScore_to_source, "BERTScore to Source", "skyblue"),
        (BERTScore_to_ref, "BERTScore to Reference", "skyblue"),
        (COMET, "COMET", "skyblue"),
        (SARI, "SARI", "skyblue")
    ]

    for ax, (metric_vals, title, color) in zip(axs.flat, metric_groups):
        ax.scatter(control_attr_vals, metric_vals, color=color, alpha=0.7, label=title)
        safe_polyfit_plot(ax, control_attr_vals, metric_vals, color="black")
        # ax.set_title(title)
        ax.set_xlabel(metric_key)
        ax.set_ylabel(title.split()[0])
        # ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric_key}_vs_metrics.png", bbox_inches='tight', dpi=400)
    print(f"Control attribute vs metrics plot saved as {metric_key}_ctrl_attr_vs_metrics.png")

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
    
    def cap_outliers(y_vals, lower_pct=1, upper_pct=99):
        if len(y_vals) == 0:
            return y_vals
        y_np = np.array(y_vals, dtype=np.float32)
        lower = np.percentile(y_np, lower_pct)
        upper = np.percentile(y_np, upper_pct)
        return np.clip(y_np, lower, upper)

    metrics = ["BLEU", "BLEU_ref", "BERTScore", "BERTScore_ref", "COMET", "SARI"]
    labels = ["BLEU to Source", "BLEU to Reference", "BERTScore to Source", "BERTScore to Reference", "COMET", "SARI"]

    fig_abs, axs_abs = plt.subplots(3, 2, figsize=(8, 10))
    fig_sq, axs_sq = plt.subplots(3, 2, figsize=(8, 10))

    for (ax_abs, ax_sq, metric_name, label) in zip(axs_abs.flat, axs_sq.flat, metrics, labels):
        x_vals, abs_errors, sq_errors = extract_errors_and_metric(predictions, metric_name, metric_key)

        abs_errors_capped = cap_outliers(abs_errors)
        sq_errors_capped = cap_outliers(sq_errors)

        ax_abs.scatter(x_vals, abs_errors_capped, alpha=0.7, color="skyblue")
        safe_polyfit_plot(ax_abs, x_vals, abs_errors_capped, color="black")
        # ax_abs.set_title(f"Abs Error vs {label}")
        ax_abs.set_xlabel(label)
        ax_abs.set_ylabel(f"Absolute Error ({metric_key_mapped})")

        ax_sq.scatter(x_vals, sq_errors_capped, alpha=0.7, color="skyblue")
        safe_polyfit_plot(ax_sq, x_vals, sq_errors_capped, color="black")
        # ax_sq.set_title(f"Squared Error vs {label}")
        ax_sq.set_xlabel(label)
        ax_sq.set_ylabel(f"Squared Error ({metric_key_mapped})")

    fig_abs.tight_layout()
    fig_sq.tight_layout()

    fig_abs.savefig(f"{output_dir}/{metric_key_mapped}_abs_error_vs_metrics.png", bbox_inches='tight', dpi=400)
    fig_sq.savefig(f"{output_dir}/{metric_key_mapped}_sq_error_vs_metrics.png", bbox_inches='tight', dpi=400)
    print(f"Error-vs-metrics plots saved.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to JSON file with predictions")
    parser.add_argument("--metric_key", type=str, required=True, help="Metric to visualize")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save plots")
    parser.add_argument("--metric_mapping", type=str, required=True)

    return parser.parse_args()

def main():

    print("--- Running evaluation...")

    args = parse_args()
    print("Input file:\t", args.input_file)
    print("Metric:\t", args.metric_key)
    use_source = is_source_metric(args) # source vals used

    metric_mapping = load_json(args.metric_mapping)
    metric_key_mapped = metric_mapping[args.metric_key]

    predictions = load_predictions(args.input_file) 

    source_vals, reference_vals, prediction_vals = extract_metric_values(predictions, metric_key_mapped, use_source=use_source)

    mse, mae = compute_mean_losses(reference_vals, prediction_vals)
    per_sample_mse, per_sample_mae = compute_per_sample_losses(reference_vals, prediction_vals)
    print(f"--- {metric_key_mapped} losses:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print()

    loss_output = {
        "metric": args.metric_key,
        "MSE": mse,
        "MAE": mae
    }

    with open(f"{args.output_dir}/losses.json", "w") as f:
        json.dump(loss_output, f, indent=4)
    print(f"Saved loss values to losses.json")

    for i, item in enumerate(predictions):
        if i < len(reference_vals):
            item[f"{args.metric_key}_losses"] = {
                "reference": reference_vals[i],
                "prediction": prediction_vals[i],
                "squared_error": per_sample_mse[i],
                "absolute_error": per_sample_mae[i]
            }

    with open(args.input_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)
    print(f"Updated input file with per-sample losses: {args.input_file}")

    plot_metric_scatter(source_vals, reference_vals, prediction_vals, metric_key_mapped, args.output_dir, use_source=use_source)
    plot_metric_lines(source_vals, reference_vals, prediction_vals, metric_key_mapped, args.output_dir, use_source=use_source)
    plot_ctrl_attr_vs_metrics(predictions, metric_key_mapped, args.output_dir)
    plot_errors_vs_metrics(predictions, metric_key_mapped, args.metric_key, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}")
    print("\n--- Evaluation complete.")


if __name__ == "__main__":
    main()