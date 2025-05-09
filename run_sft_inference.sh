#!/bin/bash


# ========== inference =========
MODEL_PATH="models/Qwen2.5-1.5B-Instruct-Med-EASi-FKGL-token_explanation_examples-20250430-2013-0s2ly6g9"
USER_PROMPT_ID="token_explanation_examples"
METRIC_NAME="FKGL"
DATASET_NAME="Med-EASi"
# "llama", "mistral", "qwen", "base" (for prompt template)
MODEL_FAMILY="base"
MODEL_CLASS="auto"



OUTPUT_DIR="output/$MODEL_PATH"
OUTPUT_FILE="$OUTPUT_DIR/output.json"
mkdir -p "$OUTPUT_DIR"  # Ensure the directory exists

python src/sft_inference.py \
  --model_path "$MODEL_PATH"\
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
# ================================



# ========== evaluation ==========
INPUT_FILE="$OUTPUT_FILE"

python src/sft_eval.py \
  --input_file "$INPUT_FILE" \
  --metric_key "$METRIC_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --metric_mapping "data/metric_mapping.json"
# ================================

