import json
import uuid
import pandas as pd
from Dataset import NewselaDataset
from Metrics import Metrics
import os

# common
DATA_DIR = "./../data"

def load_newsela():

    dataset = NewselaDataset(limit=None)
    dataset.load_data()
    df = dataset.data_df
    grouped = df.groupby(dataset.grouping_tag) # to group simplificaitions by source text id
    return dataset, grouped

def convert_to_jsonl(dataset, grouped_df):

    jsonl_data = []

    for slug, group in grouped_df:
        group = group.sort_values(by=dataset.grade_level)
        source_text = group.iloc[0][dataset.text]
        source_metrics = Metrics(source_text).compute_metrics()

        # iterate through all grade levels ~ simplifications
        for _, row in group.iterrows():
            language = row[dataset.language]

        json_entry = {
            "global_id": "newsela_" + str(uuid.uuid4()),
            "source_text": source_text,
            "metadata": {
                "dataset": dataset.dataset_name,
                "split": "",
                "domain": dataset.domain,
                "language": row[dataset.language], 
                "annotation": dataset.annotation,
                "level": dataset.alignment_level
            },
            "source_metrics": {
                "CEFR": "",
                "FRE": source_metrics["FRE"],
                "ARI": source_metrics["ARI"],
                "FKGL": source_metrics["FKGL"],
                "Dale-Chall": source_metrics["Dale-Chall"],
                "char_count": source_metrics["char_count"],
                "alphanum_count": source_metrics["alphanum_count"],
                "word_count": source_metrics["word_count"],
                "sentence_count": source_metrics["sent_count_nltk"]
            },
            "simplifications": []
        }

        # iterate through all grade levels ~ simplifications
        for _, row in group.iterrows():
            simplification_text = row[dataset.text]
            simplification_metrics = Metrics(simplification_text).compute_metrics()

            simplification_entry = {
                "simplification_text": simplification_text,
                "simplification_version": int(row[dataset.simplification_version]),
                "CEFR_level": "",  # Placeholder, to be computed later
                "grade_level": int(row[dataset.grade_level]),
                "control_attributes": {
                    "LevSim": 0.0,
                    "NbChar": 0.0 # TODO expand as needed
                },
                "simplification_dimensions": {
                    "linguistic": {
                        "lexical": False,
                        "syntactic": False,
                        "not_specified": False
                    },
                    "information": {
                        "explicitation": False,
                        "omission": False,
                        "generalization": False
                    }
                },
                "target_metrics": {
                    "CEFR": "", # TODO
                    "FRE": simplification_metrics["FRE"],
                    "ARI": simplification_metrics["ARI"],
                    "FKGL": simplification_metrics["FKGL"],
                    "Dale-Chall": simplification_metrics["Dale-Chall"],
                    "char_count": simplification_metrics["char_count"],
                    "alphanum_count": simplification_metrics["alphanum_count"],
                    "word_count": simplification_metrics["word_count"],
                    "sentence_count": simplification_metrics["sent_count_nltk"]
                },
                "evaluation_metrics": { # TODO
                    "BLEU": 0.0,
                    "ROUGE": 0.0,
                    "SARI": 0.0,
                    "BERTScore": 0.0
                }
            }

            json_entry["simplifications"].append(simplification_entry)

        jsonl_data.append(json_entry)

    return jsonl_data

def save_jsonl(dataset, jsonl_data):

    output_dir = f"{DATA_DIR}/datasets/{dataset.dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/dataset.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry) + "\n")

    print(f"JSONL file saved: {output_file}")
    print(f"Total entries: {len(jsonl_data)}")

def main():

    dataset, grouped = load_newsela()
    jsonl_data = convert_to_jsonl(dataset, grouped)
    save_jsonl(dataset, jsonl_data)


if __name__=="__main__":
    main()
