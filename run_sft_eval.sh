#!/bin/bash

INPUT_DIR="output/models/Llama-3.2-1B-Instruct-Med-EASi-DALE-CHALL-base-token_explanation-20250505-1317-oqxznih4"
INPUT_FILE="$INPUT_DIR/output.json"
METRIC_NAME="DALE-CHALL"



OUTPUT_DIR=$INPUT_DIR

python src/sft_eval.py \
  --input_file "$INPUT_FILE" \
  --metric_key "$METRIC_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --metric_mapping "data/metric_mapping.json"