import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

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

def plot_metric_scatter(source_vals, reference_vals, prediction_vals, metric_key, output_dir, use_source=False):

    x = list(range(len(reference_vals)))

    plt.figure(figsize=(12, 6))

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

def plot_metric_lines(source_vals, reference_vals, prediction_vals, metric_key, output_dir, use_source=False):

    x = list(range(len(prediction_vals)))

    plt.figure(figsize=(12, 6))

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to JSON file with predictions")
    parser.add_argument("--metric_key", type=str, required=True, help="Metric to visualize")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save plots")
    parser.add_argument("--metric_mapping", type=str, required=True)

    return parser


def main():

    print("--- Running evaluation...")

    args = parse_args()
    print("Input file:\t", args.input_file)
    print("Metric:\t", args.metric_key)
    use_source = is_source_metric(args) # source vals used

    metric_mapping = load_json(args.metric_mapping)
    metric_key_mapped = metric_mapping[args.metric_key]

    predictions = load_predictions(args.input_file) # load preds from inference json 

    source_vals, reference_vals, prediction_vals = extract_metric_values(predictions, metric_key_mapped, use_source=use_source)

    plot_metric_scatter(source_vals, reference_vals, prediction_vals, metric_key_mapped, args.output_dir, use_source=use_source)
    plot_metric_lines(source_vals, reference_vals, prediction_vals, metric_key_mapped, args.output_dir, use_source=use_source)

    print(f"Scatter plot saved as {metric_key_mapped}_scatter.png")
    print(f"Line plot saved as {metric_key_mapped}_line.png")
    print("--- Evaluation complete.")


if __name__ == "__main__":
    main()