import os
import argparse
import torch
from tqdm import tqdm
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
from helpers.prompting import select_random_system_prompt, select_random_user_prompt, select_control_token_explanation, select_random_control_token_examples
from helpers.prompting import create_user_prompt, format_prompt_with_special_tokens, format_completion_with_special_tokens
from helpers.hugging_face import load_dataset_from_hf, get_model_short_name
from classes.Metrics import Metrics


def load_json(file_path: str):
    """Load JSON from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def load_and_prepare_model(model_path, model_class, max_length):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.model_max_length = max_length
    tokenizer.truncation_side = "right"
    
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({
            'pad_token': '[PAD]'
        })
    tokenizer.padding_side = "left"
    
    if tokenizer.eos_token is None or tokenizer.eos_token != "<|eot_id|>":
        tokenizer.add_special_tokens({
            'eos_token': '<|eot_id|>'
        })

    # Load the model
    if model_class == "llama":
        model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    model.gradient_checkpointing_enable()  # for batching imitation
    model.config.use_cache = False  # use less memory
    model.resize_token_embeddings(len(tokenizer))  # resize after adding new tokens
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def load_and_prepare_test_set(dataset_name, tokenizer, max_length, control_tokens, system_prompts, user_prompts, metric_mapping, metric_name, user_prompt_id, model_family, slice_test=None):
    test_dataset = load_dataset_from_hf(dataset_name, split="test", slice=slice_test)

    # Select random system and user prompts
    system_id, system_prompt = select_random_system_prompt(system_prompts)
    
    def process_instance(row):

        # Extract relevant values from the row
        metric_key_in_dataset = metric_mapping[metric_name]
        source_metric_value = row["source_metrics"][metric_key_in_dataset]
        target_metric_value = row["target_metrics"][metric_key_in_dataset]
        reference_simplification = row["simplification_text"]
        
        # Dynamic explanation and examples if needed
        explanation = select_control_token_explanation(control_tokens, metric_name, target_metric_value) \
            if "explanation" in user_prompt_id else None
        
        examples = select_random_control_token_examples(control_tokens, metric_name) \
            if "examples" in user_prompt_id else None
        
        # Create the user prompt dynamically
        _, user_prompt = create_user_prompt(
            user_prompts=user_prompts,
            metric_name=metric_name,
            source_metric_value=source_metric_value,
            target_metric_value=target_metric_value,
            user_prompt_id=user_prompt_id,
            text=row["source_text"],
            explanation=explanation,
            examples=examples
        )
        
        # Format the prompt with special tokens
        formatted_prompt = format_prompt_with_special_tokens(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metric_name=metric_name,
            target_metric_value=target_metric_value,
            model_family=model_family
        )
        
        # Format the completion
        formatted_completion = format_completion_with_special_tokens(completion=reference_simplification, model_family=model_family)
        # Encode the prompt and completion using the tokenizer
        input_ids = tokenizer.encode(formatted_prompt, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        completion_ids = tokenizer.encode(formatted_completion, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        return {
            "prompt": formatted_prompt,
            "completion": formatted_completion,
            "input_ids": input_ids.squeeze(0),  # Remove batch dimension
            "completion_ids": completion_ids.squeeze(0),  # Remove batch dimension
        }

    test_dataset = test_dataset.map(process_instance, batched=False) # batching enabled
    
    return test_dataset

def run_inference(args, metric_mapping, model, tokenizer, test_dataset, batch_size=4, device="cuda", max_length=512, max_new_tokens=511):
    model.eval()
    predictions = []
    
    # Create a DataLoader to handle batching
    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Running inference on test set"):
        batch = test_dataset[i:i + batch_size]
        batch = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]

        # Ensure we're working with a list of dictionaries
        input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch]).to(device)
        
        with torch.no_grad():
            # Ensure attention mask is provided if it's not None
            attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch]).to(device) if "attention_mask" in batch[0] else None
            
            # Generate with the max_new_tokens to limit the number of tokens generated beyond the input length
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length + max_new_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # temperature=0.7,
                # top_k=50,
                # top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Slice off the generated portion (remove the input tokens)
            generated_only_ids = outputs[:, input_ids.shape[-1]:]  # Skip the input portion
            
            decoded_preds = tokenizer.batch_decode(
                generated_only_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            print("\n--- Processing batch:")
            for item, pred in zip(batch, decoded_preds):
                prediction_metrics = Metrics(input_text=pred.strip(), reference_text=item["simplification_text"], source_text=item["source_text"])
                computed_prediction_metrics = prediction_metrics.compute_metrics()
                source_metrics = Metrics(input_text=item["source_text"])
                computed_source_metrics = source_metrics.compute_metrics()
                reference_metrics = Metrics(input_text=item["simplification_text"], source_text=item["source_text"])
                computed_reference_metrics = reference_metrics.compute_metrics()
                predictions.append({
                    "global_id": item["global_id"],
                    "control_token": f"{args.metric_name}={item['target_metrics'][metric_mapping[args.metric_name]]}",
                    "metric_name": args.metric_name,
                    "source_metric_value": item["source_metrics"][metric_mapping[args.metric_name]],
                    "reference_metric_value": item["target_metrics"][metric_mapping[args.metric_name]],
                    "source_text": item["source_text"],
                    "reference_simplification": item["simplification_text"],
                    "prediction": pred.strip(),
                    "prompt": item["prompt"],
                    "source_metrics": computed_source_metrics,
                    "prediction_metrics": computed_prediction_metrics,
                    "reference_metrics": computed_reference_metrics,
                    })
                print(f"{pred.strip()[:50]}...")

    return predictions

def save_predictions_as_json(predictions, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        # json.dump([{"prediction": p} for p in predictions], f, indent=2, ensure_ascii=False)
        json.dump([p for p in predictions], f, indent=2, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="The path to the model dir.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset on Hugging Face.")
    parser.add_argument("--model_class", type=str, required=True, choices=["llama", "auto"], help="Model class to use.")
    parser.add_argument("--model_family", type=str, default="llama", choices=["llama", "mistral", "qwen", "base"], help="Model family to use.")
    parser.add_argument("--max_length", type=int, default=512, help="Max length for tokenization.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--slice_test", type=int, default=-1, help="Slice the test set for dev. -1 means no slicing.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the predictions.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run inference on.")
    
    # Arguments for dynamic prompting
    parser.add_argument("--control_tokens", type=str, required=True, help="Path to the control tokens JSON file.")
    parser.add_argument("--system_prompts", type=str, required=True, help="Path to the system prompts JSON file.")
    parser.add_argument("--user_prompts", type=str, required=True, help="Path to the user prompts JSON file.")
    parser.add_argument("--metric_mapping", type=str, required=True, help="Path to the metric mapping JSON file.")
    parser.add_argument("--metric_name", type=str, required=True, help="Metric name to use.")
    parser.add_argument("--user_prompt_id", type=str, required=True, choices=["no_token", "token", "token_explanation", "token_explanation_examples"], help="The user prompt ID to use.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the model and tokenizer
    model, tokenizer = load_and_prepare_model(args.model_path, args.model_class, args.max_length)
    model.to(args.device)

    # Load and prepare the dynamic prompting information (control tokens, system prompts, etc.)
    control_tokens = load_json(args.control_tokens)
    system_prompts = load_json(args.system_prompts)
    user_prompts = load_json(args.user_prompts)
    metric_mapping = load_json(args.metric_mapping)

    # Load and prepare the test dataset with dynamic prompts
    test_dataset = load_and_prepare_test_set(
        args.dataset_name,
        tokenizer,
        args.max_length,
        control_tokens,
        system_prompts,
        user_prompts,
        metric_mapping,
        args.metric_name,
        args.user_prompt_id,
        args.model_family,
        args.slice_test
    )

    predictions = run_inference(
        args, 
        metric_mapping, 
        model, 
        tokenizer, 
        test_dataset, 
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length, 
        max_new_tokens=args.max_length - 1
        )

    # Save the predictions to a file
    save_predictions_as_json(predictions, args.output_file)
    print(f"Predictions saved to {args.output_file}")

 
if __name__ == "__main__":
    main()
