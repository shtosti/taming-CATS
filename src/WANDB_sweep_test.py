import os
import random
import torch
import wandb
from dotenv import load_dotenv

# --- Load secrets ---
load_dotenv(dotenv_path="./.env", override=True)
wandb_api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_api_key)

# --- Print GPU info ---
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# --- Sweep configuration ---
sweep_config = {
    "method": "random",  # or "grid", "bayes"
    "metric": {
        "name": "loss",
        "goal": "minimize"
    },
    "parameters": {
        "learning_rate": {
            "values": [0.001, 0.01, 0.02, 0.05]
        },
        "architecture": {
            "values": ["CNN", "RNN"]
        },
        "epochs": {
            "values": [5, 10]
        }
    }
}

# --- Define your training function ---
def train():
    run = wandb.init()
    config = run.config

    print(f"Running with config: {config}")

    offset = random.random() / 5
    for epoch in range(2, config.epochs):
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset

        run.log({"acc": acc, "loss": loss})

    run.finish()

# --- Main block ---
if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity="shtosti",
        project="thesis-SFT"
    )

    # Run the sweep agent for N runs (or leave out `count` to run forever)
    wandb.agent(sweep_id, function=train, count=10)
