#!/bin/bash

INPUT_DIR="output/sft/Llama-3.2-1B-Instruct-Med-EASi-20250423-1516-l9qlzepa"
INPUT_FILE="$INPUT_DIR/output.json"
METRIC_NAME="ARI"



OUTPUT_DIR=$INPUT_DIR

python src/sft_eval.py \
  --input_file "$INPUT_FILE" \
  --metric_key "$METRIC_NAME" \
  --output_dir "$OUTPUT_DIR" \
