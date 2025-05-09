#!/bin/bash

INPUT_DIR="output/models/Llama-3.2-1B-Instruct-Med-EASi-WORD_COMPRESSION-base-token_explanation_examples-20250505-1439-ls0sezbu"
INPUT_FILE="$INPUT_DIR/output.json"
METRIC_NAME="WORD_COMPRESSION"



OUTPUT_DIR=$INPUT_DIR

python src/sft_eval.py \
  --input_file "$INPUT_FILE" \
  --metric_key "$METRIC_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --metric_mapping "data/metric_mapping.json"