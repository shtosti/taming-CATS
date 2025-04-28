#!/bin/bash

USER_PROMPT_ID="token_explanation"
METRIC_NAME="ARI"
DATASET_NAME="Med-EASi"
MODEL_PATH="sft/Llama-3.2-1B-Instruct-Med-EASi-20250423-1516-l9qlzepa"



python src/sft_inference.py \
  --model_path "$MODEL_PATH"\
  --dataset_name "$DATASET_NAME" \
  --model_class llama \
  --max_length 512 \
  --batch_size 4 \
  --output_file predictions.txt \
  --control_tokens "data/prompts/control_tokens.json" \
  --system_prompts "data/prompts/system_prompts.json" \
  --user_prompts "data/prompts/user_prompts.json" \
  --metric_mapping "data/metric_mapping.json" \
  --metric_name "$METRIC_NAME" \
  --user_prompt_id "$USER_PROMPT_ID"
