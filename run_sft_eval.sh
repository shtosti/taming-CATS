#!/bin/bash

echo "Script started: $(date)"

# TODO
DATA_DIR="Llama-3.1-8B-Instruct-SimPA-WORD_COMPRESSION"
MODEL_NAME="Llama-3.1-8B-Instruct"
DATASET="Med-EASi"
METRIC_NAME="WORD_COMPRESSION"

# BASE_DIR="output/sft_inference"
BASE_DIR="output/baseline_inference"
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
  --summary_file="output/nonsft_results_baseline/all_results.json"

echo "Script completed: $(date)"