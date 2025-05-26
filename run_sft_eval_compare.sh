
python src/sft_eval_compare.py \
  --summary_file="output/models/all_results.json" \
  --dataset="WikiLarge_ori_splitwise_hq" \
  --control_attr="FKGL" \
  --save_dir="output/sft_results" \
  --color_map_path="data/colormap/color_map.json"\
  --user_prompt_id="token_explanation"
