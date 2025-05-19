#!/bin/bash


# ========== inference =========
USE_PEFT=true
MODEL_PATH="models/Llama-2-13b-chat-hf-Med-EASi_hq-FKGL-token_explanation-20250519-192205"
MODEL_NAME="meta-llama/Llama-2-13b-chat-hf"
METRIC_NAME="FKGL"
DATASET_NAME="Med-EASi_hq"
MODEL_CLASS="llama" # "llama" or "auto"


OUTPUT_DIR="output/$MODEL_PATH"
OUTPUT_FILE="$OUTPUT_DIR/output.json"
mkdir -p "$OUTPUT_DIR"  # Ensure the directory exists

ARGS=(
  --model_path "$MODEL_PATH"
  --model_name "$MODEL_NAME"
  --dataset_name "$DATASET_NAME"
  --model_class "$MODEL_CLASS"
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


