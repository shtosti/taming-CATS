#!/bin/bash

echo "Script started: $(date)"

# =========================================================================
# BASELINE CONFIGURATION - Non-finetuned model from Hugging Face
# =========================================================================
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"  # Change to any HF model
# MODEL_NAME="Qwen/Qwen3-1.7B"
# MODEL_NAME="mistralai/Ministral-3b-instruct"

METRIC_NAME="CHAR_COMPRESSION"
DATASET="Newsela_s"
USER_PROMPT_ID="token_explanation"
# =========================================================================

# For baseline, model_path = model_name (no local finetuned model)
MODEL_PATH="$MODEL_NAME"
OUTPUT_DIR="output/baseline_inference/$(basename $MODEL_NAME)-$DATASET-$METRIC_NAME"
mkdir -p "$OUTPUT_DIR"

SEEDS=(
  42
  37 
  96
)

i=1
for SEED in "${SEEDS[@]}"; do
  echo "Running baseline inference $i with seed $SEED..."

  OUTPUT_FILE="$OUTPUT_DIR/output_$i.json"

  ARGS=(
    # --use_vllm  # Uncomment for faster inference if vllm is installed
    --seed "$SEED"
    --model_path "$MODEL_PATH"
    --model_name "$MODEL_NAME"
    --dataset_name "$DATASET"
    --model_class "auto"
    --model_family "base"  # Options: "llama", "mistral", "qwen", "base"
    --max_length 4096
    --batch_size 16
    --slice_test -1  # -1 for full test set, or specify a number for subset
    --output_file "$OUTPUT_FILE"
    --control_tokens "data/prompts/control_tokens.json"
    --system_prompts "data/prompts/system_prompts.json"
    --user_prompts "data/prompts/user_prompts.json"
    --metric_mapping "data/metric_mapping.json"
    --metric_name "$METRIC_NAME"
    --user_prompt_id "$USER_PROMPT_ID"
  )

  # Note: No --peft_path for baseline (non-finetuned models)

  python src/sft_inference.py "${ARGS[@]}"

  ((i++))
  echo "Baseline inference $i completed."

done

# =========================================================================
echo "Running evaluation script..."

INPUT_DIR=$OUTPUT_DIR
INPUT_FILES=(
  "$INPUT_DIR/output_1.json"
  "$INPUT_DIR/output_2.json"
  "$INPUT_DIR/output_3.json"
)

python src/sft_eval.py \
  --input_files "${INPUT_FILES[@]}" \
  --metric_key "$METRIC_NAME" \
  --output_dir "$INPUT_DIR" \
  --metric_mapping "data/metric_mapping.json"\
  --model_name "$(basename $MODEL_NAME)"\
  --dataset "$DATASET"\
  --user_prompt_id="$USER_PROMPT_ID"\
  --summary_file="output/sft_results/baseline_results.json"

echo "Script completed: $(date)"
