import torch
import json
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from helpers.hugging_face import load_dataset_from_hf
from helpers.prompting import create_inference_prompt, select_random_system_prompt
from classes.Metrics import Metrics
import torch.nn as nn
mse_loss_fn = nn.MSELoss()



def load_json(file_path: str):
    """Load JSON from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
    


# === CONFIG ===
model_name = "your-model-name"
model_family = "str"
model_class = "str"
dataset_name = "your-dataset-name"
split = "test"
metric_name = "FKGL"
output_file = "predictions_fkgl.jsonl"

def parse_args():
    parser = argparse.ArgumentParser()
    # hyperparams
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved checkpoints and tokenizer.")
    parser.add_argument("--model_family", type=str, required=True, choices=["llama", "auto"], help="Model class to use.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    # === Load model and tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    device = model.device
    pass

# === Load dataset ===
dataset = load_dataset_from_hf(dataset_name, split=split)


system_prompts = load_json(args.system_prompts)
system_id, system_prompt = select_random_system_prompt(system_prompts)

# === Evaluation ===
results = []
fkgl_scores = []
mse_losses = []


for example in tqdm(dataset):
    reference = example["simplification_text"].strip()
    source = example["source_text"].strip()
    target_fkgl = float(example["target_metrics"]["FKGL"])
    prompt = create_inference_prompt(
        text=source, 
        metric_name=metric_name, 
        metric_value=target_fkgl, 
        system_prompt, 
        model_family="base"
    )
    
    (
        
        
        
 )

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
        "fkgl": target_fkgl,
        "predicted_fkgl": predicted_fkgl,
        "target_fkgl": target_fkgl,
        "mse_loss": loss
    })


# === Print summary ===
avg_mse = sum(mse_losses) / len(mse_losses)
print(f"Average MSE Loss on FKGL: {avg_mse:.4f}")

# === Save results ===
with open(output_file, "w") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")

print(f"Saved results to {output_file}")
