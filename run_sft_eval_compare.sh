
python src/sft_eval_compare.py \
  --summary_file="output/models/all_results.json" \
  --dataset="Med-EASi" \
  --control_attr="CHAR_COMPRESSION" \
  --save_dir="output/sft_results" \
  --color_map_path="data/colormap/color_map.json"\
  --user_prompt_id="token_explanation"
