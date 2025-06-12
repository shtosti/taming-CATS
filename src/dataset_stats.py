import os
import json
import sys
import argparse
import matplotlib.pyplot as plt
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset statistics script.")
    parser.add_argument("--splits_path", type=str, required=True, help="Path to the dataset splits dir.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the stats and plots.")
    return parser.parse_args()

def load_jsonl_file(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def compute_source_metrics(data):
    source_fkgl_scores = [entry["source_metrics"]["FKGL"] for entry in data]
    source_fre_scores = [entry["source_metrics"]["FRE"] for entry in data]
    source_ari_scores = [entry["source_metrics"]["ARI"] for entry in data]
    source_dale_chall_scores = [entry["source_metrics"]["Dale-Chall"] for entry in data]
    source_word_count = [entry["source_metrics"]["word_count"] for entry in data]
    source_char_count = [entry["source_metrics"]["char_count"] for entry in data]
    source_sent_count = [entry["source_metrics"]["sentence_count"] for entry in data]

    avg_metrics = {
        "FKGL": round(sum(source_fkgl_scores) / len(source_fkgl_scores),1),
        "FRE": round(sum(source_fre_scores) / len(source_fre_scores),1),
        "ARI": round(sum(source_ari_scores) / len(source_ari_scores),1),
        "Dale-Chall": round(sum(source_dale_chall_scores) / len(source_dale_chall_scores),1),
        "char_count": round(sum(source_char_count) / len(source_char_count),1),
        "word_count": round(sum(source_word_count) / len(source_word_count),1),
        "sentence_count": round(sum(source_sent_count) / len(source_sent_count),1)
    }

    return avg_metrics

def compute_target_metrics(data):
    target_fkgl_scores = [entry["target_metrics"]["FKGL"] for entry in data]
    target_fre_scores = [entry["target_metrics"]["FRE"] for entry in data]
    target_ari_scores = [entry["target_metrics"]["ARI"] for entry in data]
    target_dale_chall_scores = [entry["target_metrics"]["Dale-Chall"] for entry in data]
    target_word_count = [entry["target_metrics"]["word_count"] for entry in data]
    target_char_count = [entry["target_metrics"]["char_count"] for entry in data]
    target_sent_count = [entry["target_metrics"]["sentence_count"] for entry in data]

    avg_metrics = {
        "FKGL": round(sum(target_fkgl_scores) / len(target_fkgl_scores),1),
        "FRE": round(sum(target_fre_scores) / len(target_fre_scores),1),
        "ARI": round(sum(target_ari_scores) / len(target_ari_scores),1),
        "Dale-Chall": round(sum(target_dale_chall_scores) / len(target_dale_chall_scores),1),
        "char_count": round(sum(target_char_count) / len(target_char_count),1),
        "word_count": round(sum(target_word_count) / len(target_word_count),1),
        "sentence_count": round(sum(target_sent_count) / len(target_sent_count),1)
    }

    return avg_metrics

def compute_combined_stats(splits_path, save_dir):
    print("\n Computing combined dataset statistics...")
    all_data = []

    for split in ["train", "val", "test"]:
        filepath = os.path.join(splits_path, f"{split}.jsonl")
        if not os.path.exists(filepath):
            print(f"Skipping {split} — file not found: {filepath}")
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            split_data = [json.loads(line) for line in f]
            all_data.extend(split_data)

    if not all_data:
        print("No data found in any split.")
        return

    source_metrics = compute_source_metrics(all_data)
    target_metrics = compute_target_metrics(all_data)

    stats = {
        "num_samples": len(all_data),
        "source_metrics": source_metrics,
        "target_metrics": target_metrics
    }

    combined_save_path = os.path.join(save_dir, "combined")
    os.makedirs(combined_save_path, exist_ok=True)

    log_file = os.path.join(combined_save_path, "log.txt")
    with open(log_file, "w", encoding="utf-8") as log:
        original_stdout = sys.stdout
        sys.stdout = log

        print("Combined Dataset Statistics")
        print(f"Total samples: {len(all_data)}")

        print(f"\nSOURCE:")
        for key, value in source_metrics.items():
            print(f"{key}: {value:.1f}")
        
        print(f"\nTARGET:")
        for key, value in target_metrics.items():
            print(f"{key}: {value:.1f}")

        sys.stdout = original_stdout

    print(f"Combined dataset stats saved to {log_file}")



def main():
    args = parse_args()

    for split in ["train", "val", "test"]:
        filepath = os.path.join(args.splits_path, f"{split}.jsonl")
        if not os.path.exists(filepath):
            print(f"Skipping {split} — file not found: {filepath}")
            continue

        print(f"Processing {split} split...")

        with open(filepath, "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f]

        split_save_dir = os.path.join(args.save_dir, split)
        os.makedirs(split_save_dir, exist_ok=True)

        global DATASET_DIR
        DATASET_DIR = split_save_dir

        source_metrics = compute_source_metrics(dataset)
        target_metrics = compute_target_metrics(dataset)
        combined_metrics = compute_combined_stats(args.splits_path, args.save_dir)

        stats = {
            "num_samples": len(dataset),
            "source_metrics": source_metrics,
            "target_metrics": target_metrics,
            "combined_metrics": combined_metrics
        }

        log_file = os.path.join(split_save_dir, "log.txt")
        with open(log_file, "w", encoding="utf-8") as log:
            original_stdout = sys.stdout
            sys.stdout = log

            for key, value in stats.items():
                print(f"{key}: {value}")

            print(f"\nSOURCE:")
            for key, value in source_metrics.items():
                print(f"{key}: {value}")
            
            print(f"\nTARGET:")
            for key, value in target_metrics.items():
                print(f"{key}: {value}")

            sys.stdout = original_stdout

        print(f"Saved logs for {split} to {split_save_dir}")



if __name__ == "__main__":
    main()