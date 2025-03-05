import json
import pandas as pd
import numpy as np

results = {}

METRICS = ["char_count", "word_count", "FKGL", "ARI", "FRE"]

for metric in METRICS:

    # Load experiment results
    RESULTS_FILE = f"./../experiments/splits/stratified_by_{metric}/all_results.json"

    with open(RESULTS_FILE, "r") as f:
        experiment_results = json.load(f)

    df = pd.DataFrame(experiment_results)

    best_per_bin = df.loc[df.groupby("num_bins")["average_KL"].idxmin()]
    best_bin = best_per_bin.loc[best_per_bin["average_KL"].idxmin()]

    results[metric] = {
        "best_num_bins": int(best_bin["num_bins"]),
        "average_KL": float(best_bin["average_KL"]),
    }

    with open("./../experiments/splits/best_stratification_results.json", "w") as f:
        json.dump(results, f, indent=4)


best_metric = None
best_num_bins = None
best_kl = float("inf")

# Iterate over each stratification metric
for strat_metric, values in results.items():
    avg_kl = values["average_KL"]  # Extract KL divergence
    num_bins = values["best_num_bins"]  # Extract best num_bins

    if avg_kl < best_kl:  # Find the minimum KL
        best_kl = avg_kl
        best_metric = strat_metric
        best_num_bins = num_bins  # Assign correct num_bins value

print(f"Best stratification metric: {best_metric}")
print(f"Best number of bins: {best_num_bins}")
print(f"Lowest average KL divergence: {best_kl:.6f}")