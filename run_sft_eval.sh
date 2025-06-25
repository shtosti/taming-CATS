#!/bin/bash

echo "Script started: $(date)"

# TODO
DATA_DIR="DALE-CHALL-SimPA-token_explanation-Qwen3-8B-20250624"
MODEL_NAME="Qwen3-8B"
DATASET="SimPA"
METRIC_NAME="DALE-CHALL"



BASE_DIR="output/sft_inference"
INPUT_DIR="$BASE_DIR/$DATA_DIR"

INPUT_FILES=(
  "$INPUT_DIR/output_1.json"
  "$INPUT_DIR/output_2.json"
  "$INPUT_DIR/output_3.json"
  "$INPUT_DIR/output_4.json"
  "$INPUT_DIR/output_5.json"
)


python src/sft_eval.py \
  --input_files "${INPUT_FILES[@]}" \
  --metric_key "$METRIC_NAME" \
  --output_dir "$INPUT_DIR" \
  --metric_mapping "data/metric_mapping.json"\
  --model_name "$MODEL_NAME"\
  --dataset "$DATASET"\
  --user_prompt_id="token_explanation"\
  --summary_file="output/sft_results/all_results.json"

echo "Script completed: $(date)"