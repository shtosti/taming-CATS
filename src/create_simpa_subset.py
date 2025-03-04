import json
import random
import os

def load_jsonl(filepath: str) -> list:
    """Load dataset from a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(filepath: str, data: list) -> None:
    """Save dataset as a JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Load datasets
simpa_lex_path = "./../data/datasets/simpa_lexical/dataset.jsonl"
simpa_syn_path = "./../data/datasets/simpa_syntactic/dataset.jsonl"

lex = load_jsonl(simpa_lex_path)
syn = load_jsonl(simpa_syn_path)

# Convert to dictionaries for quick lookup
lex_dict = {line["source_text"]: line for line in lex}
syn_dict = {line["source_text"]: line for line in syn}

# Identify overlapping and unique source texts
overlapping_sources = set(lex_dict.keys()) & set(syn_dict.keys())
unique_lex = {src: lex_dict[src] for src in set(lex_dict.keys()) - overlapping_sources}
unique_syn = {src: syn_dict[src] for src in set(syn_dict.keys()) - overlapping_sources}

# Split overlapping sentences into two halves (randomly assigning them to lex or syn)
overlapping_sources = list(overlapping_sources)
random.shuffle(overlapping_sources)  # Shuffle to ensure randomness
split_point = len(overlapping_sources) // 2
selected_lex_sources = set(overlapping_sources[:split_point])
selected_syn_sources = set(overlapping_sources[split_point:])

# Get the actual entries
selected_lex = {src: lex_dict[src] for src in selected_lex_sources}
selected_syn = {src: syn_dict[src] for src in selected_syn_sources}

# Combine all selected entries ensuring uniqueness
combined_dataset = list(unique_lex.values()) + list(unique_syn.values()) + list(selected_lex.values()) + list(selected_syn.values())

# Save the new dataset
output_dir = "./../data/datasets/simpa"
os.makedirs(output_dir, exist_ok=True)
combined_path = f"{output_dir}/dataset.jsonl"
save_jsonl(combined_path, combined_dataset)

print(f"Final dataset saved with {len(combined_dataset)} unique entries!")
