#!/bin/bash


# ========== inference =========
MODEL_PATH="models/Llama-3.2-1B-Instruct-Med-EASi-ARI-base-token_explanation-20250512-1705-zh8jna8l"
USER_PROMPT_ID="token_explanation"
METRIC_NAME="ARI"
DATASET_NAME="Med-EASi"
# "llama", "mistral", "qwen", "base" (for prompt template)
MODEL_FAMILY="base"
# "llama" or "auto"
MODEL_CLASS="llama"



OUTPUT_DIR="output/$MODEL_PATH"
OUTPUT_FILE="$OUTPUT_DIR/output.json"
mkdir -p "$OUTPUT_DIR"  # Ensure the directory exists

python src/sft_inference.py \
  --model_path "$MODEL_PATH"\
  --peft_path "$MODEL_PATH" \
  --dataset_name "$DATASET_NAME" \
  --model_class "$MODEL_CLASS" \
  --model_family "$MODEL_FAMILY" \
  --max_length 512 \
  --batch_size 4 \
  --slice_test -1 \
  --output_file "$OUTPUT_FILE" \
  --control_tokens "data/prompts/control_tokens.json" \
  --system_prompts "data/prompts/system_prompts.json" \
  --user_prompts "data/prompts/user_prompts.json" \
  --metric_mapping "data/metric_mapping.json" \
  --metric_name "$METRIC_NAME" \
  --user_prompt_id "$USER_PROMPT_ID"

