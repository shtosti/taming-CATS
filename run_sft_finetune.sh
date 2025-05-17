#!/bin/bash

# *** TODO ***

USE_PEFT=false 
MODEL_CLASS="llama" # --- model class: "llama" for llama and mistral, "auto" for qwen
MODEL_FAMILY="base" # --- model family: "llama", "mistral", "qwen", "base" (for prompt template)

# --- model name
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_NAME="meta-llama/Llama-2-13b-chat-hf"
# MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
# MODEL_NAME="ministral/Ministral-3b-instruct"
# MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.1"


# --- dataset settings ---
DATASET_NAME="Med-EASi_hq"
SLICE_TRAIN="-1" # -1 means no slicing
SLICE_VAL="-1" # -1 means no slicing

# --- prompting settings ---
PROMPTING_TYPE="vanilla" # "vanilla", "reasoning", "transformations"
USER_PROMPT_ID="token_explanation" # "token", "token_explanation", "token_explanation_examples"
METRIC_NAME="ARI"

# --- hyperparameters ---
EPOCHS=3
PATIENCE=3
MAX_LENGTH=512
BATCH_SIZE=8
EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
LR=1e-5
WEIGHT_DECAY=0.01
LOGGING_STEPS=20
WARMUP_STEPS=30
MAX_GRAD_NORM=0.5
LOG_EVERY=20

# --- WANDB settings ---
WANDB_PROJECT_NAME="ATS_with_control_tokens"
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
python src/sft_finetune.py \
    --model_class "$MODEL_CLASS" \
    --model_family "$MODEL_FAMILY" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --slice_train "$SLICE_TRAIN" \
    --slice_val "$SLICE_VAL" \
    --batch_size "$BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_steps "$WARMUP_STEPS" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --logging_steps "$LOGGING_STEPS" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --max_length "$MAX_LENGTH"\
    --wandb_project_name "$WANDB_PROJECT_NAME" \
    --wandb_entity "$WANDB_ENTITY"\
    --prompting_type "$PROMPTING_TYPE" \
    --user_prompt_id "$USER_PROMPT_ID" \
    --metric_name "$METRIC_NAME" \
    --control_tokens "data/prompts/control_tokens.json" \
    --system_prompts "data/prompts/system_prompts.json" \
    --user_prompts "data/prompts/user_prompts.json"\
    --metric_mapping "data/metric_mapping.json"\
    --log_every "$LOG_EVERY"\
    $PEFT_FLAG