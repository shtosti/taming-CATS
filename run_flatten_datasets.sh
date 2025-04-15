#!/bin/bash

# Navigate to your working directory
echo "current working directory:"
pwd

echo "Running on $(hostname)"
nvidia-smi

# Run your Python script
echo "Running the script ./src/flatten_datasets.py ..."
python ./src/flatten_datasets.py
echo "Script finished."