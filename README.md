# keep-it-simple

## Classes
- Dataset
- Metrics
- ControlTokensAccess
- ControlTokensMuss

## scripts
- `generate_responses.py` - generate responses from the model to explore simplification via prompting
- `create_wikilarge_subset.py` - create a subset of the wikilarge dataset using stratification metrics
- `explore_datasets.py` - explore dataset stats and plot them
- `dataset_stats.py` - (old) compute dataset stats. Replace through `explore_datasets.py`?

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