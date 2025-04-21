import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from helpers.hugging_face import load_dataset_from_hf
from helpers.prompting import create_inference_prompt
from classes.Metrics import Metrics

# === CONFIG ===
model_name = "your-model-name"
dataset_name = "your-dataset-name"
split = "validation"
metric_name = "FKGL"
output_file = "predictions_fkgl.jsonl"

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
device = model.device

# === Load dataset ===
dataset = load_dataset_from_hf(dataset_name, split=split)

# === Evaluation ===
results = []
fkgl_scores = []

for example in tqdm(dataset):
    reference = example["simplification_text"].strip()
    source = example["source_text"].strip()
    prompt = create_inference_prompt(source, metric_name, example["target_metrics"][metric_name])

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    prediction = decoded.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()

    # Compute FKGL using your Metrics class
    metric_obj = Metrics(input_text=prediction)
    fkgl_score = metric_obj.compute_fkgl()

    results.append({
        "global_id": example.get("global_id"),
        "source_text": source,
        "reference": reference,
        "prediction": prediction,
        "fkgl": fkgl_score
    })

    fkgl_scores.append(fkgl_score)

# === Print summary ===
avg_fkgl = sum(fkgl_scores) / len(fkgl_scores)
print(f"\nAverage FKGL of predictions: {avg_fkgl:.2f}")

# === Save results ===
with open(output_file, "w") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")

print(f"Saved results to {output_file}")
