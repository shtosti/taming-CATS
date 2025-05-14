#!/bin/bash

# Navigate to your working directory
echo "current working directory:"
pwd

echo "Running on $(hostname)"
nvidia-smi

# Run your Python script
echo "Running the script ./src/flatten_splits.py ..."
python ./src/flatten_splits.py
echo "Script finished."