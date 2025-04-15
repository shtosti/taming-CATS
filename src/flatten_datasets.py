import json
import os

def flatten_jsonl(input_file, output_file):
    flattened_data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse each line in the .jsonl file
            example = json.loads(line.strip())
            
            # If 'simplifications' is a list, flatten each simplification
            if isinstance(example.get("simplifications", None), list):
                for simplification in example["simplifications"]:
                    flattened_data.append({
                        "global_id": example["global_id"],
                        "source_text": example.get("source_text", ""),
                        "source_metrics": example.get("source_metrics", {}),
                        "metadata": example["metadata"],
                        "simplification_text": simplification.get("simplification_text", ""),
                        "target_metrics": simplification.get("target_metrics", {}),
                        "simplification_version": simplification.get("simplification_version", ""),
                        "grade_level": simplification.get("grade_level", ""),
                        "simplification_dimensions": simplification.get("simplification_dimensions", {})
                    })

            else:
                # If 'simplifications' is missing or not a list, handle it here
                print(f"Warning: 'simplifications' missing or not a list in example: {example}")
    
    # Save the flattened data to a new file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in flattened_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def main():
    datasets = [
                "newsela", 
                "simpa", 
                "medeasi"
                ] # TODO add wikilarge once the splits have been generated
    for dataset in datasets:
        print(f"{5*"*"} Processing {dataset}... {5*"*"}")
        input_dir = f"./data/splits/{dataset}"
        output_dir = f"./data/splits_flattened/{dataset}"
        os.makedirs(output_dir, exist_ok=True)
        for split in ["train", "val", "test"]:
            input_jsonl = f'{input_dir}/{split}.jsonl'
            output_jsonl = f'{output_dir}/{split}.jsonl'

            flatten_jsonl(input_jsonl, output_jsonl)

if __name__=="__main__":
    main()
