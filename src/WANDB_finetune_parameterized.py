import argparse
import os
import sys
import json
import random
from datetime import datetime
from dotenv import load_dotenv
import wandb
import torch
import transformers
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from helpers.hugging_face import load_dataset_from_hf, get_model_short_name
from helpers.prompting import select_random_system_prompt, select_random_user_prompt, select_control_token_explanation, select_random_control_token_examples
from helpers.prompting import create_user_prompt, format_prompt_with_special_tokens, format_completion_with_special_tokens
from classes.PredictionLoggerCallback import PredictionLoggerCallback

print("Transformers version:", transformers.__version__)
print("Python path:", sys.executable)
random.seed(42)


def load_json(file_path: str):
    """Load JSON from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
    
def print_gpu_info():
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))


def print_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


def load_and_prepare_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # add custom padding token if necessary
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # define and add custom special tokens
    special_tokens = {
        'additional_special_tokens': ['<|user|>', 
                                      '<|system|>', 
                                      '<|assistant|>', 
                                      '<|begin_of_text|>', 
                                      '<|end_of_text|>',
                                      '<|start_header_id|>',
                                      '<|end_header_id|>',
                                      '<|eot_id|>'
                                      ]
    }
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [tok for tok in special_tokens['additional_special_tokens'] if tok not in existing_tokens]
    if new_tokens:
        print("New tokens added to the tokenizer:", new_tokens)
        tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
        # tokenizer.add_special_tokens(special_tokens)

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # resize after adding new tokens
    model.resize_token_embeddings(len(tokenizer))
    # for debugging
    print_trainable_params(model)

    return model, tokenizer

def tokenize_dataset(dataset, tokenizer):
    def tokenize(example):
        tokenized = tokenizer(
            example["prompt"],
            text_target=example["completion"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        # tokenized["labels"] = tokenized["input_ids"].copy() # if train on prompt+completion
        return tokenized
    return dataset.map(tokenize, batched=True)

def show_examples(dataset, n=3):
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    print(f"\nShowing {n} examples:")
    for i, idx in enumerate(indices):
        example = dataset[idx]
        prompt = example.get("prompt", "N/A")
        completion = example.get("completion", "N/A")

        print(f"\nExample {i + 1} (index {idx})")
        print("--- Prompt:\n", prompt)
        print("--- Completion:\n", completion)
        print("=" * 50)

def load_and_prepare_dataset(dataset_name, tokenizer, args):

    control_tokens = load_json(args.control_tokens)
    system_prompts = load_json(args.system_prompts)
    user_prompts = load_json(args.user_prompts)

    system_id, system_prompt = select_random_system_prompt(system_prompts)

    # def process_example(example):
    #     metric_value = example["simplifications"][0]["target_metrics"][args.metric_name]
    #     # TODO modify to use them all, not just the first simplification from the simplifications array

    #     # Get control token explanation and examples if needed
    #     explanation = select_control_token_explanation(control_tokens, args.metric_name, metric_value) \
    #         if "explanation" in args.user_prompt_id else None
        
    #     examples = select_random_control_token_examples(control_tokens, args.metric_name) \
    #         if "examples" in args.user_prompt_id else None

    #     reference_simplification = example["simplifications"][0]["simplification_text"]
    #     # TODO modify to use them all, not just the first simplification from the simplifications array

    #     # Build user prompt dynamically
    #     _, user_prompt = create_user_prompt(
    #         user_prompts,
    #         metric_name=args.metric_name,
    #         metric_value=metric_value,
    #         user_prompt_id=args.user_prompt_id,
    #         text=example["source_text"],
    #         explanation=explanation,
    #         examples=examples
    #     )

    #     metadata = {
    #         "prompt": format_prompt_with_special_tokens(system_prompt, user_prompt),
    #         "completion": format_completion_with_special_tokens(reference_simplification),
    #         "system_prompt_id": system_id,
    #         "user_prompt_id": args.user_prompt_id,
    #         "metric_value": metric_value
    #     }

    #     # return format_instruction(example, system_prompt, user_prompt)
    #     return metadata
    
    def flatten_rows(batch):
        outputs = []
        
        # Check if 'simplifications' exists and is a list
        if isinstance(batch.get("simplifications", None), list):
            for simplification in batch["simplifications"]:
                if isinstance(simplification, dict):  # Ensure it's a dictionary
                    outputs.append({
                        "source_text": batch["source_text"],  
                        "source_metrics": batch["source_metrics"],  
                        "simplification_text": simplification.get("simplification_text", ""),  # Use .get() for safety
                        "target_metrics": simplification.get("target_metrics", {}),  # Use .get() for safety
                    })
                else:
                    print(f"Warning: Simplification is not a dictionary, skipping: {simplification}")
        else:
            print(f"Warning: 'simplifications' is not a list or missing in batch: {batch}")
        
        return outputs


    def process_example(example):
        metric_value = example["target_metrics"][args.metric_name]
        reference_simplification = example["simplification_text"]

        explanation = select_control_token_explanation(control_tokens, args.metric_name, metric_value) \
            if "explanation" in args.user_prompt_id else None
        
        examples = select_random_control_token_examples(control_tokens, args.metric_name) \
            if "examples" in args.user_prompt_id else None

        _, user_prompt = create_user_prompt(
            user_prompts,
            metric_name=args.metric_name,
            metric_value=metric_value,
            user_prompt_id=args.user_prompt_id,
            text=example["source_text"],
            explanation=explanation,
            examples=examples
        )

        return {
            "prompt": format_prompt_with_special_tokens(system_prompt, user_prompt),
            "completion": format_completion_with_special_tokens(reference_simplification),
            "system_prompt_id": system_id,
            "user_prompt_id": args.user_prompt_id,
            "metric_value": metric_value
        }

    train_dataset = load_dataset_from_hf(dataset_name, split="train", slice=args.slice_train)
    val_dataset = load_dataset_from_hf(dataset_name, split="validation", slice=args.slice_val)

    print(f"Original train dataset length: {len(train_dataset)}")
    print(f"Original val dataset length: {len(val_dataset)}")

    train_dataset = train_dataset.map(flatten_rows, remove_columns=train_dataset.column_names, batched=True)
    val_dataset = val_dataset.map(flatten_rows,remove_columns=val_dataset.column_names, batched=True)

    print(f"Flattened train dataset length: {len(train_dataset)}")
    print(f"Flattened val dataset length: {len(val_dataset)}")

    train_dataset = train_dataset.map(process_example)
    val_dataset = val_dataset.map(process_example)

    print("\n\n *** Before tokenization ***")
    show_examples(train_dataset, n=1)

    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    val_dataset = tokenize_dataset(val_dataset, tokenizer)

    print("\n\n *** After tokenization ***")
    show_examples(train_dataset, n=1)

    print(10*"*", "DEBUG", 10*"*")
    print("Train dataset columns:", train_dataset.column_names)
    print("Val dataset columns:", val_dataset.column_names)

    return train_dataset, val_dataset

def train_model(model, tokenizer, train_dataset, val_dataset, args, output_dir):

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        push_to_hub=False,
        report_to=["wandb"]
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        tokenizer=tokenizer,
        callbacks=[PredictionLoggerCallback(tokenizer, val_dataset, log_every=args.log_every)],
    )

    trainer.train()
    trainer.save_model(output_dir)
    wandb.save(output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--slice_train", type=int, default=-1)
    parser.add_argument("--slice_val", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--wandb_project_name", type=str, default="thesis-SFT")
    parser.add_argument("--wandb_entity", type=str, default="shtosti")
    parser.add_argument("--log_every", type=int, default=20)

    # for dynamic prompting
    parser.add_argument("--prompting_type", type=str, default="vanilla", choices=["vanilla", "reasoning", "transformations"])
    parser.add_argument("--user_prompt_id", type=str, default="token", choices=["no_token", "token", "token_explanation", "token_explanation_examples"])
    parser.add_argument("--metric_name", type=str, required=True)

    # Paths to external JSON files for control tokens, system prompts, and user prompts
    parser.add_argument("--control_tokens", type=str, required=True)
    parser.add_argument("--system_prompts", type=str, required=True)
    parser.add_argument("--user_prompts", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\nRun arguments:\n{args}\n")

    load_dotenv(dotenv_path="./.env", override=True)
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity
    )
    wandb.config.update(vars(args))
    wandb_run_id = wandb.run.id

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    short_model = get_model_short_name(args.model_name)
    output_dir = f"./sft/{short_model}-{args.dataset_name}-{timestamp}-{wandb_run_id}"
    print(f"\n*** Finetuning {short_model} with {args.dataset_name} ***\n")

    print_gpu_info()

    print("Current working directory:", os.getcwd())
    print("Saving to:", output_dir)

    model, tokenizer = load_and_prepare_model(args.model_name)
    train_dataset, val_dataset = load_and_prepare_dataset(args.dataset_name, tokenizer, args)

    print(10*"*", "DEBUG", 10*"*")
    example = train_dataset[0]
    print(type(example["input_ids"]))  # should be list of ints
    print(type(example["labels"]))     # should be list of ints


    train_model(model, tokenizer, train_dataset, val_dataset, args, output_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
