#!/bin/bash


# ========== inference =========
USE_PEFT=true
MODEL_PATH="models/Qwen2.5-7B-Instruct-WikiLarge_ori_splitwise_hq-FKGL-token_explanation-20250519-022012"

# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_NAME="meta-llama/Llama-2-13b-chat-hf"
# MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
# MODEL_NAME="ministral/Ministral-3b-instruct"
# MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.1"

METRIC_NAME="FKGL"
DATASET_NAME="WikiLarge_ori_splitwise_hq"


OUTPUT_DIR="output/$MODEL_PATH"
mkdir -p "$OUTPUT_DIR"

SEEDS=(37 15 96 2 28)
i=1
for SEED in "${SEEDS[@]}"; do
  echo "Running inference $i with seed $SEED..."

  OUTPUT_FILE="$OUTPUT_DIR/output_$i.json"

  ARGS=(
    --seed 42
    --model_path "$MODEL_PATH"
    --model_name "$MODEL_NAME"
    --dataset_name "$DATASET_NAME"
    --model_class "auto"
    --model_family "base"
    --max_length 512
    --batch_size 4
    --slice_test -1
    --output_file "$OUTPUT_FILE"
    --control_tokens "data/prompts/control_tokens.json"
    --system_prompts "data/prompts/system_prompts.json"
    --user_prompts "data/prompts/user_prompts.json"
    --metric_mapping "data/metric_mapping.json"
    --metric_name "$METRIC_NAME"
    --user_prompt_id "token_explanation"
  )

  if [ "$USE_PEFT" = true ]; then
    echo "Using PEFT..."
    ARGS+=( --peft_path "$MODEL_PATH" )
  fi

  python src/sft_inference.py "${ARGS[@]}"

  ((i++))
  echo "Inference $i completed."

done
