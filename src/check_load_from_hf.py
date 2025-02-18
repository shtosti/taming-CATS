from datasets import load_dataset
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env", override=True)

data_files = {"train": "train.jsonl", "test": "test.jsonl", "validation": "val.jsonl"}
dataset = load_dataset("shtosti/SimPA_lex", token=os.getenv("HF_TOKEN"), data_files=data_files)



validation_dataset = dataset["validation"]

print(len(validation_dataset))
for line in validation_dataset:
    print(line)