# keep-it-simple

## Working on Cluster

### Account
Project tenant name (account): **iict-sp1.ebling.cl.uzh**

### conda venv
Activate virtual environment: `conda activate sft`
Install additional packages: `conda install package-name`
Update YAML file with new packages: `conda env update --name sft --file environment.yml --prune`
Update conda if required: `conda update -n base -c conda-forge conda`


## Classes
- Dataset
- Metrics
- ControlTokensAccess
- ControlTokensMuss

## scripts
- `generate_responses.py` - generate responses from the model to explore simplification via prompting
- `create_wikilarge_subset.py` - create a subset of the wikilarge dataset using stratification metrics
- `plot_wikilarge_divergencies` - plot the divergencies of the wikilarge dataset, comparing stats between the original large dataset and the sampled subset
- `explore_datasets.py` - explore dataset stats and plot them
- `create_simpa_subset.py` - create a combined dataset from Simpa lexical and Simpa syntactic. No duplicates.
- `create_splits.py` - create train/dev/test splits from the combined dataset. 9:1:1. Remove bottom 3 and top 3 percentiles of the char_count distribution.


## Data
Dataset splits have been uploaded to Hugging Face. They can be loaded from `datasets` with e.g. the following code:
```
from datasets import load_dataset
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env", override=True)


dataset = load_dataset(
    "shtosti/SimPA_lex", 
    token=os.getenv("HF_TOKEN"), 
    data_files={"train": "train.jsonl", "test": "test.jsonl", "validation": "val.jsonl"}
    )

validation_dataset = dataset["validation"]
```


## Pipeline

### Datasets
1. create jsonl from the datasets wwith `dataset_to_jsonl.py`
2. create a subset of the wikilarge dataset with `create_wikilarge_subset.py`
3. explore the datasets with `explore_datasets.py`

### Prompting
1. explore with `generate_responses.py`



## Stratification metrics (wikilarge experiment)
- **number of characters** is statistically the best stratification metric to create a subset of the dataset or create train/dev/test splits. Problem: this is only true when sampling from the whole dataset.
- unstable top ranking metric when sampling from corresponding splits of the source dataset (wikilarge). What shall we do? I will use **char_count** for creating the splits for all datasets, for now.
- Tried to create a larget subset with from-split sampling, but not possibel due to the small test-set size of the original wikilarge dataset.
- problem with the **original wikilarge splits**: not really representative for the whole dataset.


## Wikilarge alignment analysis
- strangely aligned sentence pairs, which make no sense. 
- [ ] **TODO**: check if the automatic alignment is correct
- [ ] **TODO**: what about the entities? add or keep substitutions?
- [ ] **TODO**: train from train, dev from dev, test from test or all from the whole dataset?