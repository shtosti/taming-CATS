#!/bin/bash

echo "Script started: $(date)"

# TODO
INPUT_DIR="output/sft_inference/CHAR_COMPRESSION-Med-EASi-token_explanation-Llama-3.1-8B-Instruct-20250527"
MODEL_NAME="Llama-3.1-8B-Instruct"
DATASET="Med-EASi"
METRIC_NAME="CHAR_COMPRESSION"
USER_PROMPT_ID="token_explanation"



INPUT_FILES=(
  "$INPUT_DIR/output_1.json"
  "$INPUT_DIR/output_2.json"
  "$INPUT_DIR/output_3.json"
  "$INPUT_DIR/output_4.json"
  "$INPUT_DIR/output_5.json"
)

# INPUT_FILES=(
#   "$INPUT_DIR/output.json"
# )

python src/sft_eval.py \
  --input_files "${INPUT_FILES[@]}" \
  --metric_key "$METRIC_NAME" \
  --output_dir "$INPUT_DIR" \
  --metric_mapping "data/metric_mapping.json"\
  --model_name "$MODEL_NAME"\
  --dataset "$DATASET"\
  --user_prompt_id="$USER_PROMPT_ID"\
  --summary_file="output/sft_results/all_results.json"

echo "Script completed: $(date)"