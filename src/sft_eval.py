import json
import argparse
import matplotlib.pyplot as plt

def load_predictions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_metric_values(predictions, metric_key):
    source_vals = []
    reference_vals = []
    prediction_vals = []

    for item in predictions:
        # Extract from nested metrics
        source_val = item["source_metrics"].get(metric_key)
        reference_val = item["reference_metrics"].get(metric_key)
        prediction_val = item["prediction_metrics"].get(metric_key)

        if source_val is not None and reference_val is not None and prediction_val is not None:
            source_vals.append(source_val)
            reference_vals.append(reference_val)
            prediction_vals.append(prediction_val)

    return source_vals, reference_vals, prediction_vals

def plot_metric_scatter(source_vals, reference_vals, prediction_vals, metric_key, output_dir):
    x = list(range(len(source_vals)))

    plt.figure(figsize=(12, 6))
    plt.scatter(x, source_vals, color="orchid", label="Source", alpha=0.7)
    plt.scatter(x, reference_vals, color="darkorange", label="Reference", alpha=0.7)
    plt.scatter(x, prediction_vals, color="seagreen", label="Prediction", alpha=0.7)

    plt.title(f"{metric_key}")
    plt.xlabel("idx")
    plt.ylabel(metric_key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric_key}_scatter_plot.png", bbox_inches='tight', dpi=400)

def plot_metric_lines(source_vals, reference_vals, prediction_vals, metric_key, output_dir):
    x = list(range(len(source_vals)))

    plt.figure(figsize=(12, 6))

    plt.plot(x, source_vals, color="orchid", label="Source", alpha=0.7)
    plt.plot(x, reference_vals, color="darkorange", label="Reference", alpha=0.7)
    plt.plot(x, prediction_vals, color="seagreen", label="Prediction", alpha=0.7)

    plt.title(f"{metric_key}")
    plt.xlabel("idx")
    plt.ylabel(metric_key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric_key}_line.png", bbox_inches='tight', dpi=400)
    print(f"Line plot saved as {metric_key}_line.png")

def main():
    print("--- Running evaluation...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to JSON file with predictions")
    parser.add_argument("--metric_key", type=str, required=True, help="Metric to visualize")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save plots")

    args = parser.parse_args()

    predictions = load_predictions(args.input_file)
    source_vals, reference_vals, prediction_vals = extract_metric_values(predictions, args.metric_key)
    plot_metric_scatter(source_vals, reference_vals, prediction_vals, args.metric_key)
    plot_metric_lines(source_vals, reference_vals, prediction_vals, args.metric_key)
    print(f"Scatter plot saved as {args.metric_key}_scatter.png")
    print(f"Line plot saved as {args.metric_key}_line.png")
    print("--- Evaluation complete.")


if __name__ == "__main__":
    main()