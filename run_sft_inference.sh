#!/bin/bash


# ========== inference =========
MODEL_PATH="models/Meta-Llama-3-8B-Instruct-Med-EASi-FKGL-base-token_explanation-20250514-2147-u4v4gkmz"
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
USER_PROMPT_ID="token_explanation"
METRIC_NAME="FKGL"
DATASET_NAME="Med-EASi"
# "llama", "mistral", "qwen", "base" (for prompt template)
MODEL_FAMILY="base"
# "llama" or "auto"
MODEL_CLASS="llama"
USE_PEFT=true


OUTPUT_DIR="output/$MODEL_PATH"
OUTPUT_FILE="$OUTPUT_DIR/output.json"
mkdir -p "$OUTPUT_DIR"  # Ensure the directory exists

ARGS=(
  --model_path "$MODEL_PATH"
  --model_name "$MODEL_NAME"
  --dataset_name "$DATASET_NAME"
  --model_class "$MODEL_CLASS"
  --model_family "$MODEL_FAMILY"
  --max_length 512
  --batch_size 4
  --slice_test 5
  --output_file "$OUTPUT_FILE"
  --control_tokens "data/prompts/control_tokens.json"
  --system_prompts "data/prompts/system_prompts.json"
  --user_prompts "data/prompts/user_prompts.json"
  --metric_mapping "data/metric_mapping.json"
  --metric_name "$METRIC_NAME"
  --user_prompt_id "$USER_PROMPT_ID"
)

if [ "$USE_PEFT" = true ]; then
  echo "Using PEFT..."
  ARGS+=( --peft_path "$MODEL_PATH" )
fi

python src/sft_inference.py "${ARGS[@]}"


