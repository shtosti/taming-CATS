ENV_NAME = sft
PYTHON = /home/hhubar/data/conda/envs/$(ENV_NAME)/bin/python
WANDB_PROJECT = thesis-SFT

q: # check server queue
	squeue --user=$(USER)

activate:
	conda activate $(ENV_NAME)
	# @echo "Activated conda environment: $(ENV_NAME)"

clean: # Clean cache
	rm -rf __pycache__ *.pyc */__pycache__ */*.pyc

finetune:
	sbatch ./slurm_sft_finetune.sh

eval:
	sbatch ./slurm_sft_eval.sh

# Just a nice summary
help:
	@echo "Makefile commands:"
	@echo "  activate     - Activate the conda environment"
	@echo "  make finetune        - Run the training script"
	@echo "  clean        - Clean python caches"