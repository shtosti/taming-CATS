import json
import os
import re

def clean_text(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def log_stats(logfile_path, message):
    with open(logfile_path, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")
    print(message)

def flatten_jsonl(input_file, output_file, log_file_path=None):
    flattened_data = []
    total_examples = 0
    total_flattened = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_examples += 1
            example = json.loads(line.strip())
            if isinstance(example.get("simplifications", None), list):
                for simplification in example["simplifications"]:
                    total_flattened += 1
                    flattened_data.append({
                        "global_id": example["global_id"],
                        "source_text": clean_text(example.get("source_text", "")),
                        "source_metrics": example.get("source_metrics", {}),
                        "metadata": example["metadata"],
                        "simplification_text": clean_text(simplification.get("simplification_text", "")),
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

    log_stats(log_file_path, f"Input file: {input_file}")
    log_stats(log_file_path, f"Total examples read: {total_examples}")
    log_stats(log_file_path, f"Total simplifications flattened: {total_flattened}")
    log_stats(log_file_path, "-" * 50)

def main():
    datasets = [
                "newsela", 
                "simpa", 
                "medeasi",
                "wikilarge_ori_splitwise",
                "wikilarge_ori_global"
                ]
    for dataset in datasets:
        print(f"{5*"*"} Processing {dataset}... {5*"*"}")
        input_dir = f"./data/splits_new/{dataset}"
        output_dir = f"./data/splits_flattened/{dataset}"
        os.makedirs(output_dir, exist_ok=True)

        log_file_path = os.path.join(output_dir, "log.txt")
        open(log_file_path, "w").close()

        for split in ["train", "val", "test"]:
            input_jsonl = f'{input_dir}/{split}.jsonl'
            output_jsonl = f'{output_dir}/{split}.jsonl'

            flatten_jsonl(input_jsonl, output_jsonl, log_file_path=log_file_path)

    log_stats(log_file_path, f"Flattening completed for dataset: {dataset}")

if __name__=="__main__":
    main()
