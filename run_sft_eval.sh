#!/bin/bash

INPUT_DIR="output/models/Llama-3.2-1B-Instruct-Med-EASi_hq-FKGL-base-token_explanation-20250516-1632-av5ug43l"
INPUT_FILE="$INPUT_DIR/output.json"
METRIC_NAME="FKGL"



OUTPUT_DIR=$INPUT_DIR

python src/sft_eval.py \
  --input_file "$INPUT_FILE" \
  --metric_key "$METRIC_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --metric_mapping "data/metric_mapping.json"