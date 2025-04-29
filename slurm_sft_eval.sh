#!/bin/bash

#SBATCH --job-name=sft_eval

#SBATCH --account=iict-sp1.ebling.cl.uzh
#SBATCH --output=logs/%x_%j_out.log
#SBATCH --error=logs/%x_%j_err.log 
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --time=00:20:00              # Max runtime (hh:mm:ss)

conda activate sft

# Run the script
bash ./run_sft_eval.sh