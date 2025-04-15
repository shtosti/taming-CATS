#!/bin/bash

#SBATCH --job-name=wandb_finetune_parameterized

#SBATCH --account=iict-sp1.ebling.cl.uzh
#SBATCH --output=logs/%x_%j_out.log
#SBATCH --error=logs/%x_%j_err.log 
#SBATCH --gpus=1
#SBATCH --mem=32000
#SBATCH --time=02:00:00              # Max runtime (hh:mm:ss)

conda activate sft
module load gpu
nvidia-smi

# Run the script
bash ./run_WANDB_finetune_parameterized.sh