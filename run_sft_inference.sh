#!/bin/bash


# ========== inference =========
USE_PEFT=false
MODEL_PATH="models/Llama-3.2-1B-Instruct-Med-EASi-CHAR_COMPRESSION-token_explanation-20250518-221639"
METRIC_NAME="CHAR_COMPRESSION"
DATASET_NAME="Med-EASi"


OUTPUT_DIR="output/$MODEL_PATH"
mkdir -p "$OUTPUT_DIR"

SEEDS=(37 15 96 2 28)
i=1
for SEED in "${SEEDS[@]}"; do
  echo "Running inference $i with seed $SEED..."

  OUTPUT_FILE="$OUTPUT_DIR/output_$i.json"

  ARGS=(
    # --use_vllm
    --seed "$SEED"
    --model_path "$MODEL_PATH"
    --dataset_name "$DATASET_NAME"
    --model_class "auto"
    --model_family "base"
    --max_length 512
    --batch_size 4
    --slice_test -1
    --output_file "$OUTPUT_FILE"
    --control_tokens "data/prompts/control_tokens.json"
    --system_prompts "data/prompts/system_prompts.json"
    --user_prompts "data/prompts/user_prompts.json"
    --metric_mapping "data/metric_mapping.json"
    --metric_name "$METRIC_NAME"
    --user_prompt_id "token_explanation"
  )

  if [ "$USE_PEFT" = true ]; then
    echo "Using PEFT..."
    ARGS+=( --peft_path "$MODEL_PATH" )
  fi

  python src/sft_inference.py "${ARGS[@]}"

  ((i++))
  echo "Inference $i completed."

done