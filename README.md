# keep-it-simple

## Pipeline
### Classes
- Dataset
- Metrics
- ControlTokensAccess
- ControlTokensMuss

### scripts
- `generate_responses.py` - generate responses from the model to explore simplification via prompting
- `create_wikilarge_subset.py` - create a subset of the wikilarge dataset using stratification metrics
- `explore_datasets.py` - explore dataset stats and plot them
- `dataset_stats.py` - (old) compute dataset stats. Replace through `explore_datasets.py`?


## Stratification metrics
<br>**number of characters** is statistically the best stratification metric to create a subset of the dataset or create train/dev/test splits. 