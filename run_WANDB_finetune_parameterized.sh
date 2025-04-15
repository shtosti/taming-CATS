#!/bin/bash

# --- model settings ---
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"

# --- dataset settings ---
DATASET_NAME="Med-EASi"
SLICE_TRAIN="-1" # -1 means no slicing
SLICE_VAL="-1" # -1 means no slicing

# --- prompting settings ---
PROMPTING_TYPE="vanilla"
USER_PROMPT_ID="token"

# --- metric settings ---
METRIC_NAME="FKGL"

# --- hyperparameters ---
EPOCHS=2
BATCH_SIZE=4 # TODO increase to 8 or 16
LR=1e-5
WEIGHT_DECAY=0.01
LOGGING_STEPS=20
LOG_EVERY=20

# --- WANDB settings ---
WANDB_PROJECT_NAME="thesis-SFT"
WANDB_ENTITY="shtosti"

python src/WANDB_finetune_parameterized.py \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --slice_train "$SLICE_TRAIN" \
    --slice_val "$SLICE_VAL" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --logging_steps "$LOGGING_STEPS" \
    --epochs "$EPOCHS"\
    --wandb_project_name "$WANDB_PROJECT_NAME" \
    --wandb_entity "$WANDB_ENTITY"\
    --prompting_type "$PROMPTING_TYPE" \
    --user_prompt_id "$USER_PROMPT_ID" \
    --metric_name "$METRIC_NAME" \
    --control_tokens "data/prompts/control_tokens.json" \
    --system_prompts "data/prompts/system_prompts.json" \
    --user_prompts "data/prompts/user_prompts.json"\
    --log_every "$LOG_EVERY"