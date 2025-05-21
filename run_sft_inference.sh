#!/bin/bash


# ========== inference =========
USE_PEFT=false
MODEL_PATH="models/Llama-3.2-1B-Instruct-Newsela-FKGL-token_explanation-20250521-000403"
METRIC_NAME="FKGL"
DATASET_NAME="Newsela"
MODEL_CLASS="auto" # "llama" or "auto"


OUTPUT_DIR="output/$MODEL_PATH"
OUTPUT_FILE="$OUTPUT_DIR/output.json"
mkdir -p "$OUTPUT_DIR"  # Ensure the directory exists

ARGS=(
  # --use_vllm
  --model_path "$MODEL_PATH"
  --dataset_name "$DATASET_NAME"
  --model_class "$MODEL_CLASS"
  --model_family "base"
  --max_length 8192
  --batch_size 4
  --slice_test 5
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


