import json
import os
import matplotlib.pyplot as plt
import numpy as np

NUM_BINS = 25
DATASET_NAME = "wikilarge_ori"
EXPERIMENT_DIR = f"./../experiments/sample_from_{DATASET_NAME}"
BIN_KEY = f"num_bins_{NUM_BINS}"
DIVERGENCE_FILE = f"{EXPERIMENT_DIR}/all_divergence_results.json"
COLOR_MAP_FILE = "./../data/colormap/color_map.json"
OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, BIN_KEY, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(DIVERGENCE_FILE, "r") as f:
    ALL_RESULTS = json.load(f)

with open(COLOR_MAP_FILE, "r") as f:
    COLOR_MAP = json.load(f)
METRIC_COLORS = COLOR_MAP["metrics"]

STATS = ["JSD", "EMD", "KS"]
STRAT_TYPES = ["splitwise", "global"]

SEED_KEYS = list(ALL_RESULTS.keys())

first_seed_data = ALL_RESULTS[SEED_KEYS[0]][BIN_KEY]
SUBSET_SIZES = sorted([int(k.split("_")[-1]) for k in first_seed_data.keys()])
SUBSET_KEYS = [f"subset_size_{s}" for s in SUBSET_SIZES]
STRAT_METRICS = [m[0] for m in first_seed_data[SUBSET_KEYS[0]]["splitwise"]["JSD"]]


score_tracker = {
    stat: {
        strat_type: {
            strat_metric: {size: [] for size in SUBSET_SIZES}
            for strat_metric in STRAT_METRICS
        } for strat_type in STRAT_TYPES
    } for stat in STATS
}

for seed_key in SEED_KEYS:
    if BIN_KEY not in ALL_RESULTS[seed_key]:
        continue
    seed_data = ALL_RESULTS[seed_key][BIN_KEY]

    for subset_key in SUBSET_KEYS:
        if subset_key not in seed_data:
            continue
        subset_size = int(subset_key.split("_")[-1])
        entry = seed_data[subset_key]

        for stat in STATS:
            for strat_type in STRAT_TYPES:
                for strat_metric, value in entry[strat_type][stat]:
                    score_tracker[stat][strat_type][strat_metric][subset_size].append(value)


def plot_separately():
    for stat in STATS:
        for strat_type in STRAT_TYPES:
            plt.figure(figsize=(8, 5))

            for strat_metric in STRAT_METRICS:
                means = []
                stds = []

                for size in SUBSET_SIZES:
                    values = score_tracker[stat][strat_type][strat_metric][size]
                    means.append(np.mean(values))
                    stds.append(np.std(values))

                means = np.array(means)
                stds = np.array(stds)

                plt.plot(SUBSET_SIZES, means,
                         label=strat_metric,
                         color=METRIC_COLORS.get(strat_metric,"#000000"))

                plt.fill_between(SUBSET_SIZES, means - stds, means + stds,
                                 alpha=0.2,
                                 color=METRIC_COLORS.get(strat_metric,"#000000"))

            plt.xlabel("Subset Size")
            plt.ylabel(f"{stat} Score")
            # plt.title(f"{stat} Divergence ({strat_type}) - Avg over Seeds")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/{strat_type}_{stat}.png", dpi=400)
            plt.close()

def plot_together():
    for stat in STATS:
        plt.figure(figsize=(8, 5))

        for strat_type in STRAT_TYPES:
            linestyle = "-" if strat_type == "splitwise" else "--"

            for strat_metric in STRAT_METRICS:
                means = []
                stds = []

                for size in SUBSET_SIZES:
                    values = score_tracker[stat][strat_type][strat_metric][size]
                    means.append(np.mean(values))
                    stds.append(np.std(values))

                means = np.array(means)
                stds = np.array(stds)

                plt.plot(SUBSET_SIZES, means,
                         linestyle=linestyle,
                         label=f"{strat_metric} ({strat_type})",
                         color=METRIC_COLORS.get(strat_metric, "#000000"))

                plt.fill_between(SUBSET_SIZES, means - stds, means + stds,
                                 alpha=0.2,
                                 color=METRIC_COLORS.get(strat_metric, "#000000"))

        plt.xlabel("Subset Size")
        plt.ylabel(f"{stat} Score")
        # plt.title(f"{stat} Divergence (All Strat Types) - Avg over Seeds")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/both_{stat}.png", dpi=300)
        plt.close()

# === EXECUTE ===
plot_separately()
plot_together()
print(f"Plots saved to {OUTPUT_DIR}")
