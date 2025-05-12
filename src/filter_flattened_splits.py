import os
import json
from collections import defaultdict

def log_stats(logfile_path, message):
    with open(logfile_path, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")
    print(message)

def filter_metrics(flattened_input_path, filtered_output_path, log_file_path=None, metrics=None):
    total = 0
    kept = 0
    removed = 0
    filtered_data = []

    if metrics is None:
        metrics = ["FKGL"]

    # Track how many times each metric caused an example to be removed
    per_metric_removals = defaultdict(int)

    with open(flattened_input_path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            example = json.loads(line.strip())
            source = example.get("source_metrics", {})
            target = example.get("target_metrics", {})

            failed_metrics = [
                metric for metric in metrics
                if not (
                    source.get(metric) is not None and 
                    target.get(metric) is not None and 
                    target[metric] < source[metric]
                )
            ]

            if not failed_metrics:
                kept += 1
                filtered_data.append(example)
            else:
                removed += 1
                for metric in failed_metrics:
                    per_metric_removals[metric] += 1

    with open(filtered_output_path, 'w', encoding='utf-8') as f:
        for entry in filtered_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Logging
    log_stats(log_file_path, f"Input file: {flattened_input_path}")
    log_stats(log_file_path, f"Total examples read: {total}")
    log_stats(log_file_path, f"Examples kept (all metrics ↓: {metrics}): {kept}")
    log_stats(log_file_path, f"Examples removed: {removed}")
    for metric in metrics:
        log_stats(log_file_path, f"→ Removed due to {metric} not decreasing: {per_metric_removals[metric]}")
    log_stats(log_file_path, "-" * 50)

def main():
    datasets = ["newsela", "simpa", "medeasi", "wikilarge_ori_splitwise"]
    metrics_to_filter_by = ["FKGL", "ARI", "Dale-Chall"]

    for dataset in datasets:
        print(f"{5 * '*'} Filtering {dataset}... {5 * '*'}")
        input_dir = f"./../data/splits_flattened/{dataset}"
        output_dir = f"./../data/splits_flattened_filtered/{dataset}"
        os.makedirs(output_dir, exist_ok=True)

        log_file_path = os.path.join(output_dir, "log.txt")
        open(log_file_path, "w").close()  # reset log

        for split in ["train", "val", "test"]:
            input_path = f"{input_dir}/{split}.jsonl"
            output_path = f"{output_dir}/{split}.jsonl"
            filter_metrics(
                input_path,
                output_path,
                log_file_path=log_file_path,
                metrics=metrics_to_filter_by
            )

        log_stats(log_file_path, f"Filtering completed for dataset: {dataset}\n")

if __name__ == "__main__":
    main()
