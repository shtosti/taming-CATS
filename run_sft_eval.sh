#!/bin/bash

# TODO
INPUT_DIR="output/models/Llama-3.2-1B-Instruct-Newsela_s-FKGL-token_explanation-20250521-113051"
MODEL_NAME="Llama-3.2-1B-Instruct"
DATASET="Newsela_s"
METRIC_NAME="FKGL"
USER_PROMPT_ID="token_explanation"


python src/sft_eval.py \
  --input_file "$INPUT_DIR/output.json" \
  --metric_key "$METRIC_NAME" \
  --output_dir "$INPUT_DIR" \
  --metric_mapping "data/metric_mapping.json"\
  --model_name "$MODEL_NAME"\
  --dataset "$DATASET"\
  --user_prompt_id="$USER_PROMPT_ID"\
  --summary_file="output/models/all_results.json"