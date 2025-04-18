#!/bin/bash

# --- model family # TODO
# MODEL_FAMILY="llama"
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"

MODEL_FAMILY="auto"
# MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
# MODEL_NAME="ministral/Ministral-3b-instruct"
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.1"

# --- PEFT flag --- # TODO
USE_PEFT=true 

# --- dataset settings ---
DATASET_NAME="Med-EASi"
SLICE_TRAIN="-1" # -1 means no slicing
SLICE_VAL="-1" # -1 means no slicing

# --- prompting settings ---
# "vanilla", "reasoning", "transformations"
PROMPTING_TYPE="vanilla" 
# "token", "token_explanation", "token_explanation_examples"
USER_PROMPT_ID="token_explanation_examples"

# --- metric settings ---
METRIC_NAME="FKGL"

# --- hyperparameters ---
EPOCHS=2
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=2
LR=1e-5
WEIGHT_DECAY=0.01
LOGGING_STEPS=20
LOG_EVERY=40

# --- WANDB settings ---
WANDB_PROJECT_NAME="thesis-SFT"
WANDB_ENTITY="shtosti"

if [ "$USE_PEFT" = true ]; then
    echo "Using PEFT..."
    PEFT_FLAG="--peft"  # Enable PEFT by passing the --peft flag
else
    PEFT_FLAG=""  # If not using PEFT, leave the flag empty
fi

CUDA_LAUNCH_BLOCKING=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_USE_CUDA_DSA=1 \
python src/WANDB_finetune_parameterized.py \
    --model_family "$MODEL_FAMILY" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --slice_train "$SLICE_TRAIN" \
    --slice_val "$SLICE_VAL" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
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
    --log_every "$LOG_EVERY"\
    $PEFT_FLAG