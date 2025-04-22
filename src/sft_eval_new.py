import os
import torch
import argparse
import json
import wandb
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from classes.Metrics import Metrics
from helpers.hugging_face import load_dataset_from_hf
from helpers.prompting import create_inference_prompt, format_completion_with_special_tokens, select_random_system_prompt
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", type=str, required=True, choices=["llama", "auto"], help="Model class to use.")
    parser.add_argument("--model_family", type=str, required=True, default="llama", choices=["llama", "mistral", "qwen", "base"], help="Model type to choose from.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the fine-tuned model is stored.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name to evaluate the model on.")
    parser.add_argument("--slice_val", type=int, default=-1, help="Slice for the validation dataset.")
    parser.add_argument("--control_tokens", type=str, required=True)
    parser.add_argument("--system_prompts", type=str, required=True)
    parser.add_argument("--user_prompts", type=str, required=True)
    return parser.parse_args()

def compute_fkgl(text):
    metric_obj = Metrics(input_text=text)
    fkgl = metric_obj.compute_fkgl()
    return fkgl

def generate_simplification(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_length=512, num_return_sequences=1)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def evaluate_model(model, tokenizer, dataset, device):
    all_actual_fkgl = []
    all_predicted_fkgl = []

    for example in dataset:
        prompt = example['prompt']
        reference_simplification = example['completion']
        
        # Generate simplification
        generated_simplification = generate_simplification(model, tokenizer, prompt, device)
        
        # Compute FKGL scores
        reference_fkgl = compute_fkgl(reference_simplification)
        prediction_fkgl = compute_fkgl(generated_simplification)
        
        all_actual_fkgl.append(reference_fkgl)
        all_predicted_fkgl.append(prediction_fkgl)
        
        print(f"Prompt: {prompt}")
        print(f"Expected simplification: {reference_simplification}")
        print(f"Generated simplification: {generated_simplification}")
        print(f"Actual FKGL: {reference_fkgl}, Predicted FKGL: {prediction_fkgl}")
        print("-" * 50)

    # Convert lists to torch tensors for MSE loss calculation
    actual_fkgl_tensor = torch.tensor(all_actual_fkgl, dtype=torch.float32)
    predicted_fkgl_tensor = torch.tensor(all_predicted_fkgl, dtype=torch.float32)

    # Initialize MSE Loss function
    mse_loss_fn = nn.MSELoss()

    # Calculate MSE loss
    mse_loss = mse_loss_fn(predicted_fkgl_tensor, actual_fkgl_tensor)
    print(f"\nMSE Loss: {mse_loss.item()}")

    return mse_loss.item()

def load_model_and_tokenizer(model_dir, model_class):
    if model_class == "llama":
        model = LlamaForCausalLM.from_pretrained(model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def load_test_dataset(dataset_name, tokenizer, args):
    from helpers.prompting import select_random_system_prompt

    # Load prompts and tokens
    control_tokens = json.load(open(args.control_tokens))
    system_prompts = json.load(open(args.system_prompts))
    
    # Randomly select a system prompt
    system_id, system_prompt = select_random_system_prompt(system_prompts)

    test_dataset = load_dataset_from_hf(dataset_name, split="test", slice=args.slice_val)

    def process_instance(row):
        metric_value = row["target_metrics"]["fkgl"]  # default to FKGL — adapt if needed
        reference_simplification = row["simplification_text"]

        prompt = create_inference_prompt(
            text=row["source_text"],
            metric_name="fkgl",
            metric_value=metric_value,
            system_prompt=system_prompt,
            model_family=args.model_family
        )

        return {
            "prompt": prompt,
            "completion": format_completion_with_special_tokens(reference_simplification, model_family=args.model_family),
            "metric_value": metric_value,
            "system_prompt_id": system_id
        }

    test_dataset = test_dataset.map(process_instance)
    return test_dataset

def main():
    args = parse_args()

    # Initialize wandb (optional)
    wandb.init(project="thesis-evaluation", entity="shtosti")
    wandb.config.update(vars(args))

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
                                                model_dir=args.model_dir, 
                                                model_class=args.model_class
                                                )

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load test dataset
    test_dataset = load_test_dataset(args.dataset_name, tokenizer, slice_val=args.slice_val)

    # Evaluate model
    mse_loss = evaluate_model(model, tokenizer, test_dataset, device)

    # Log the MSE loss to wandb
    wandb.log({"MSE Loss": mse_loss})

    wandb.finish()

if __name__ == "__main__":
    main()
