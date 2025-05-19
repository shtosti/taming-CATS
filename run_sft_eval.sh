#!/bin/bash

# TODO
INPUT_DIR="output/models/Meta-Llama-3-8B-Instruct-Med-EASi-CHAR_COMPRESSION-token_explanation-20250518-223705"
MODEL_NAME="Meta-Llama-3-8B-Instruct"
DATASET="Med-EASi"
METRIC_NAME="CHAR_COMPRESSION"
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