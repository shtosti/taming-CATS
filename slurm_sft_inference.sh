#!/bin/bash

#SBATCH --job-name=sft_inference

#SBATCH --account=iict-sp1.ebling.cl.uzh
#SBATCH --output=logs/%x_%j_out.log
#SBATCH --error=logs/%x_%j_err.log 
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000
#SBATCH --time=01:00:00              # Max runtime (hh:mm:ss)

conda activate sft
module load gpu
nvidia-smi

# Run the script
bash ./run_sft_inference.sh