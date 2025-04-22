import torch
import json
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from helpers.hugging_face import load_dataset_from_hf
from helpers.prompting import create_inference_prompt
from classes.Metrics import Metrics
import torch.nn as nn
mse_loss_fn = nn.MSELoss()

# === CONFIG ===
model_name = "your-model-name"
dataset_name = "your-dataset-name"
split = "test"
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
mse_losses = []

def parse_args():
    parser = argparse.ArgumentParser()
    # hyperparams
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved checkpoints and tokenizer.")
    parser.add_argument("--model_family", type=str, required=True, choices=["llama", "auto"], help="Model class to use.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)

    return parser.parse_args()

for example in tqdm(dataset):
    reference = example["simplification_text"].strip()
    source = example["source_text"].strip()
    target_fkgl = float(example["target_metrics"]["FKGL"])
    prompt = create_inference_prompt(source, metric_name, target_fkgl)

    # prompt = create_inference_prompt(source, metric_name, example["target_metrics"][metric_name])
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    prediction = decoded.strip()
    # prediction = decoded.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip() # TODO remove or change

    # Compute FKGL using your Metrics class
    metric_obj = Metrics(input_text=prediction)
    predicted_fkgl = metric_obj.compute_fkgl()

    # Compute MSE between predicted and target FKGL
    fkgl_tensor = torch.tensor([predicted_fkgl], dtype=torch.float32)
    target_tensor = torch.tensor([target_fkgl], dtype=torch.float32)
    loss = mse_loss_fn(fkgl_tensor, target_tensor).item()

    fkgl_scores.append(predicted_fkgl)
    mse_losses.append(loss)

    results.append({
        "global_id": example.get("global_id"),
        "source_text": source,
        "reference": reference,
        "prediction": prediction,
        "fkgl": fkgl_score,
        "predicted_fkgl": predicted_fkgl,
        "target_fkgl": target_fkgl,
        "mse_loss": loss
    })


# === Print summary ===
avg_fkgl = sum(fkgl_scores) / len(fkgl_scores)
avg_mse = sum(mse_losses) / len(mse_losses)
print(f"\nAverage FKGL of predictions: {avg_fkgl:.2f}")
print(f"Average MSE Loss on FKGL: {avg_mse:.4f}")

# === Save results ===
with open(output_file, "w") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")

print(f"Saved results to {output_file}")
