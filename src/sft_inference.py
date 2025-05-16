import os
import argparse
import torch
from tqdm import tqdm
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
# from peft import PeftModel, PeftConfig
# from peft import LoraConfig, PeftModelForCausalLM, get_peft_config

from helpers.prompting import select_random_system_prompt, select_random_user_prompt, select_control_token_explanation, select_random_control_token_examples
from helpers.prompting import create_user_prompt, format_prompt_with_special_tokens, format_completion_with_special_tokens, format_prompt_with_tokenizer, format_completion_with_tokenizer
from helpers.hugging_face import load_dataset_from_hf, get_model_short_name
from classes.Metrics import Metrics


def load_json(file_path: str):
    """Load JSON from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def setup_tokenizer(model_source, model_family, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    tokenizer.model_max_length = max_length
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.eos_token is None:
        if model_family == "qwen":
            tokenizer.add_special_tokens({'eos_token': '<|im_end|>'})
        elif model_family == "base":
            tokenizer.add_special_tokens({'eos_token': '<|eot_id|>'})

    return tokenizer

def load_and_prepare_model(model_name, model_family, model_path, model_class, max_length, peft_path=None):

    tokenizer = setup_tokenizer(model_path, model_family, max_length)

    if peft_path is not None:
        base_model_class = LlamaForCausalLM if model_class == "llama" else AutoModelForCausalLM
        base_model = base_model_class.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        base_model.resize_token_embeddings(len(tokenizer))
        # from peft import PeftModel
        # model = PeftModel.from_pretrained(base_model, peft_path)
        from peft import PeftModelForCausalLM
        model = PeftModelForCausalLM.from_pretrained(base_model, peft_path)
        model.resize_token_embeddings(len(tokenizer))

        if model_class == "llama":
            base_model = LlamaForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                # torch_dtype=torch.float16
            )

        base_model.resize_token_embeddings(len(tokenizer))

    else:
        model_class = LlamaForCausalLM if model_class == "llama" else AutoModelForCausalLM
        model = model_class.from_pretrained(
            model_path,
            device_map="auto",
            # torch_dtype=torch.float16
            )
        model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def is_source_metric(args):
    if args.metric_name in ["FRE", "FKGL", "ARI", "DALE-CHALL", "Dale-Chall"]:
        return True
    return False
    
def is_compression_metric(args):
    if args.metric_name in ["CHAR_COMPRESSION", "WORD_COMPRESSION", "SENTENCE_COMPRESSION"]:
        return True
    return False

def load_and_prepare_test_set(dataset_name, tokenizer, max_length, control_tokens, system_prompts, user_prompts, metric_mapping, args, user_prompt_id, model_family, slice_test=None, source_based_metric=False, compression_metric=False):
    test_dataset = load_dataset_from_hf(dataset_name, split="test", slice=slice_test)
    
    system_id, system_prompt = select_random_system_prompt(system_prompts)
    
    def process_instance(row):

        metric_key_mapped = metric_mapping[args.metric_name]

        if source_based_metric:
            source_metric_value = row["source_metrics"][metric_key_mapped]
            target_metric_value = row["target_metrics"][metric_key_mapped]
        else:
            source_metric_value = None
            target_metric_value = None

        if compression_metric:
            if args.metric_name == "WORD_COMPRESSION":
                target_metric_value = round(row["target_metrics"]["word_count"] / row["source_metrics"]["word_count"], 1)
            if args.metric_name == "CHAR_COMPRESSION":
                target_metric_value = round(row["target_metrics"]["char_count"] / row["source_metrics"]["char_count"], 1)
            if args.metric_name == "SENTENCE_COMPRESSION":
                target_metric_value = round(row["target_metrics"]["sent_count"] / row["source_metrics"]["sent_count"], 1)

        reference_simplification = row["simplification_text"]
        
        # Dynamic explanation and examples if needed
        explanation = select_control_token_explanation(control_tokens, args.metric_name, target_metric_value) \
            if "explanation" in user_prompt_id else None
        
        examples = select_random_control_token_examples(control_tokens, args.metric_name) \
            if "examples" in user_prompt_id else None
        
        # Create the user prompt dynamically
        _, user_prompt = create_user_prompt(
            user_prompts=user_prompts,
            metric_name=metric_key_mapped,
            source_metric_value=source_metric_value,
            target_metric_value=target_metric_value,
            user_prompt_id=user_prompt_id,
            text=row["source_text"],
            explanation=explanation,
            examples=examples
        )
        
        # Format the prompt with special tokens
        # formatted_prompt = format_prompt_with_special_tokens(
        formatted_prompt = format_prompt_with_tokenizer(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metric_name=metric_key_mapped,
            target_metric_value=target_metric_value,
            # model_family=model_family
        )
        
        # Format the completion
        # formatted_completion = format_completion_with_special_tokens(
        formatted_completion = format_completion_with_tokenizer(
            tokenizer=tokenizer,
            completion=reference_simplification, 
            # model_family=model_family
            )

        encoded = tokenizer(
            formatted_prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",  # or "longest" for dynamic padding
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"]
        # attention_mask = encoded["attention_mask"]

        encoded_completion = tokenizer(
            formatted_completion,
            truncation=True,
            max_length=max_length,
            padding="max_length",  # or "longest"
            return_tensors="pt"
        )
        completion_ids = encoded_completion["input_ids"]

        return {
            "prompt": formatted_prompt,
            "completion": formatted_completion,
            "input_ids": input_ids.squeeze(0),  # Remove batch dimension
            "completion_ids": completion_ids.squeeze(0),  # Remove batch dimension,
            # "attention_mask": attention_mask.squeeze(0),  # Remove batch dimension
        }

    test_dataset = test_dataset.map(process_instance, batched=False) # batching enabled
    
    return test_dataset

def run_inference(args, metric_mapping, model, tokenizer, test_dataset, batch_size=4, device="cuda", max_length=512, max_new_tokens=511, source_based_metric=False, compression_metric=False):
    model.eval()
    predictions = []
    
    # Create a DataLoader to handle batching
    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Running inference on test set"):
        batch = test_dataset[i:i + batch_size]
        batch = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]

        # Ensure we're working with a list of dictionaries
        input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch]).to(device) # TODO  bring back
        
        with torch.no_grad():
            torch.cuda.empty_cache()
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
                # skip_special_tokens=False,
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

                prediction_char_compression = round(computed_prediction_metrics["char_count"] / computed_source_metrics["char_count"], 1)
                prediction_word_compression = round(computed_prediction_metrics["word_count"] / computed_source_metrics["word_count"], 1)
                prediction_sent_compression = round(computed_prediction_metrics["sent_count"] / computed_source_metrics["sent_count"], 1)
                computed_prediction_metrics["char_compression_rate"] = prediction_char_compression
                computed_prediction_metrics["word_compression_rate"] = prediction_word_compression
                computed_prediction_metrics["sentence_compression_rate"] = prediction_sent_compression

                reference_char_compression = round(computed_reference_metrics["char_count"] / computed_source_metrics["char_count"], 1)
                reference_word_compression = round(computed_reference_metrics["word_count"] / computed_source_metrics["word_count"], 1)
                reference_sent_compression = round(computed_reference_metrics["sent_count"] / computed_source_metrics["sent_count"], 1)
                computed_reference_metrics["char_compression_rate"] = reference_char_compression
                computed_reference_metrics["word_compression_rate"] = reference_word_compression
                computed_reference_metrics["sentence_compression_rate"] = reference_sent_compression

                prediction_dict = {
                    "global_id": item["global_id"],
                    "control_token": f"{args.metric_name}={computed_reference_metrics[metric_mapping[args.metric_name]]}",
                    "metric_name": args.metric_name,
                    "reference_metric_value": computed_reference_metrics[metric_mapping[args.metric_name]],
                    "source_text": item["source_text"],
                    "reference_simplification": item["simplification_text"],
                    "prediction": pred.strip(),
                    "prompt": item["prompt"],
                    "source_metrics": computed_source_metrics,
                    "prediction_metrics": computed_prediction_metrics,
                    "reference_metrics": computed_reference_metrics,
                    }
                if source_based_metric:
                    prediction_dict["source_metric_value"] = item["source_metrics"][metric_mapping[args.metric_name]]
                
                predictions.append(prediction_dict)

                print(f"{pred.strip()[:100]}...")

    return predictions

def save_predictions_as_json(predictions, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([p for p in predictions], f, indent=2, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="The path to the model dir.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset on Hugging Face.")
    parser.add_argument("--model_class", type=str, required=True, choices=["llama", "auto"], help="Model class to use.")
    parser.add_argument("--model_family", type=str, default="llama", choices=["llama", "mistral", "qwen", "base"], help="Model family to use.")
    parser.add_argument("--max_length", type=int, default=512, help="Max length for tokenization.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--slice_test", type=int, default=-1, help="Slice the test set for dev. -1 means no slicing.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the predictions.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run inference on.")
    parser.add_argument("--peft_path", type=str, default=None, help="Path to PEFT adapter dir.")

    
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
    print(f"Loading from {args.model_path}...")
    print(f"Inference args:\n{args}\n")

    # Load the model and tokenizer
    model, tokenizer = load_and_prepare_model(
        args.model_name,
        args.model_family, 
        args.model_path, 
        args.model_class, 
        args.max_length,
        peft_path=args.peft_path
        )
    model.to(args.device)

    # Load and prepare the dynamic prompting information (control tokens, system prompts, etc.)
    control_tokens = load_json(args.control_tokens)
    system_prompts = load_json(args.system_prompts)
    user_prompts = load_json(args.user_prompts)
    metric_mapping = load_json(args.metric_mapping)

    # check if source value exists
    source_based_metric = is_source_metric(args)
    # check if compression rate needs to be calculated
    compression_metric = is_compression_metric(args)

    # Load and prepare the test dataset with dynamic prompts
    test_dataset = load_and_prepare_test_set(
        args.dataset_name,
        tokenizer,
        args.max_length,
        control_tokens,
        system_prompts,
        user_prompts,
        metric_mapping,
        args,
        args.user_prompt_id,
        args.model_family,
        args.slice_test,
        source_based_metric=source_based_metric,
        compression_metric=compression_metric
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
        max_new_tokens=args.max_length - 1,
        source_based_metric=source_based_metric,
        compression_metric=compression_metric
        )

    # Save the predictions to a file
    save_predictions_as_json(predictions, args.output_file)
    print(f"Predictions saved to {args.output_file}")

 
if __name__ == "__main__":
    main()
