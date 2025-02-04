import json
import pandas as pd
from collections import Counter
from Dataset import NewselaDataset
from Metrics import Metrics

# Load the dataset
DATA_DIR = "./../data"
DATASET_PATH = f"{DATA_DIR}/datasets/newsela/dataset.jsonl"

def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def compute_statistics(data):
    num_sources = len(data)
    num_simplifications = sum(len(entry["simplifications"]) for entry in data)
    
    # Grade level distribution
    grade_levels = [simp["grade_level"] for entry in data for simp in entry["simplifications"]]
    grade_level_counts = Counter(grade_levels)
    
    # Language distribution
    languages = [entry["metadata"]["language"] for entry in data]
    language_counts = Counter(languages)
    
    # Text length stats
    source_word_counts = [entry["source_metrics"]["word_count"] for entry in data]
    simplification_word_counts = [simp["target_metrics"]["word_count"] for entry in data for simp in entry["simplifications"]]
    
    # Readability scores
    fkgl_scores = [simp["target_metrics"]["FKGL"] for entry in data for simp in entry["simplifications"]]
    fre_scores = [simp["target_metrics"]["FRE"] for entry in data for simp in entry["simplifications"]]
    
    stats = {
        "Number of unique source texts": num_sources,
        "Total simplifications": num_simplifications,
        "Grade level distribution": grade_level_counts,
        "Language distribution": language_counts,
        "Average words per source text": sum(source_word_counts) / len(source_word_counts),
        "Average words per simplification": sum(simplification_word_counts) / len(simplification_word_counts),
        "Average FKGL score": sum(fkgl_scores) / len(fkgl_scores),
        "Average FRE score": sum(fre_scores) / len(fre_scores)
    }
    
    return stats

def main():
    dataset = load_jsonl(DATASET_PATH)
    stats = compute_statistics(dataset)
    
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
