import os
import argparse
import sys
import json
import random
from datetime import datetime
from dotenv import load_dotenv
import wandb
from peft import get_peft_model, LoraConfig
import bitsandbytes as bnb
import torch
torch.cuda.empty_cache()
print("torch.cuda.is_bf16_supported:", torch.cuda.is_bf16_supported())
import transformers
from transformers import LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

from helpers.hugging_face import load_dataset_from_hf, get_model_short_name
from helpers.prompting import select_random_system_prompt, select_random_user_prompt, select_control_token_explanation, select_random_control_token_examples
from helpers.prompting import create_user_prompt, format_prompt_with_special_tokens, format_completion_with_special_tokens

from classes.PredictionLoggerCallback import PredictionLoggerCallback

print("Transformers version:", transformers.__version__)
print("Python path:", sys.executable)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def load_and_prepare_model(model_family, model_name, model_class, peft_enabled, max_length):
    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = max_length
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({
            'pad_token': '[PAD]'
        })
    tokenizer.padding_side = "left"

    if model_family == "qwen":
        if tokenizer.eos_token is None or tokenizer.eos_token != "<|im_end|>":
            tokenizer.add_special_tokens({
                'eos_token': '<|im_end|>'
            })
    elif model_family == "base":
        if tokenizer.eos_token is None or tokenizer.eos_token != "<|eot_id|>":
            tokenizer.add_special_tokens({
                'eos_token': '<|eot_id|>'
            })

    # --- model ---
    if model_class == "llama":
        model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")
    elif model_class == "auto":
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    else:
        raise ValueError(f"Unsupported model_class: {model_class}")

    model.gradient_checkpointing_enable() # batching imitation
    model.config.use_cache = False # use less memory
    model.resize_token_embeddings(len(tokenizer)) # resize after adding new tokens
    model.config.pad_token_id = tokenizer.pad_token_id

    if peft_enabled:
        print("Using PEFT for fine-tuning...")
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none"
        )
        model = get_peft_model(model, peft_config)

    # --- debug ---
    print("--- DEBUG ---")
    print("trainable params:")
    print_trainable_params(model)
    print(f"--- tokenizer pad_token_id: {tokenizer.pad_token_id}")
    print(f"--- tokenizer eos_token_id: {tokenizer.eos_token_id}")
    print(f"--- special tokens:")
    for token in tokenizer.additional_special_tokens:
        print(f"{token}: {tokenizer.convert_tokens_to_ids(token)}")

    return model, tokenizer

def tokenize_dataset(dataset, tokenizer, max_length):
    def tokenize(example):
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=True).input_ids
        completion_ids = tokenizer(example["completion"], add_special_tokens=True).input_ids

        input_ids = prompt_ids + completion_ids
        attention_mask = [1] * len(input_ids)

        # Create labels: mask out the prompt part
        labels = [-100] * len(prompt_ids) + completion_ids
        assert any(label != -100 for label in labels), "All labels are -100!"

        # Truncate to max_length after combining
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        attention_mask = attention_mask[:max_length]

        # Pad if necessary
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [tokenizer.pad_token_id] * padding_length
            labels += [-100] * padding_length
            attention_mask += [0] * padding_length

        # Ensure that labels are aligned with input_ids
        assert len(input_ids) == len(labels), f"Length mismatch: {len(input_ids)} != {len(labels)}"
        assert len(input_ids) == len(attention_mask), f"Length mismatch: {len(input_ids)} != {len(attention_mask)}"

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

    return dataset.map(tokenize, batched=False)

def show_examples(dataset, n=3, show_tokens=False):
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    print(f"\nShowing {n} example(s):")
    for i, idx in enumerate(indices):
        example = dataset[idx]
        prompt = example.get("prompt", "N/A")
        completion = example.get("completion", "N/A")

        print(f"\nExample {i + 1} (index {idx})")
        print("--- Prompt:\n", prompt)
        print("--- Completion:\n", completion)

        if show_tokens:
            print("--- input_ids:\n", example.get("input_ids"))
            print("--- labels:\n", example.get("labels"))

        print("=" * 50)

def load_and_prepare_dataset(dataset_name, tokenizer, args):

    control_tokens = load_json(args.control_tokens)
    system_prompts = load_json(args.system_prompts)
    user_prompts = load_json(args.user_prompts)
    metric_mapping = load_json(args.metric_mapping)

    system_id, system_prompt = select_random_system_prompt(system_prompts)
    
    def process_instance(row):
        metric_key_in_dataset = metric_mapping[args.metric_name]
        source_metric_value = row["source_metrics"][metric_key_in_dataset]
        target_metric_value = row["target_metrics"][metric_key_in_dataset]
        reference_simplification = row["simplification_text"]

        explanation = select_control_token_explanation(control_tokens, args.metric_name, target_metric_value) \
            if "explanation" in args.user_prompt_id else None
        
        examples = select_random_control_token_examples(control_tokens, args.metric_name) \
            if "examples" in args.user_prompt_id else None

        _, user_prompt = create_user_prompt(
            user_prompts=user_prompts,
            metric_name=args.metric_name,
            source_metric_value=source_metric_value,
            target_metric_value=target_metric_value,
            user_prompt_id=args.user_prompt_id,
            text=row["source_text"],
            explanation=explanation,
            examples=examples
        )

        return {
            "prompt": format_prompt_with_special_tokens(
                                                        system_prompt=system_prompt, 
                                                        user_prompt=user_prompt, 
                                                        metric_name=args.metric_name, 
                                                        target_metric_value=target_metric_value,
                                                        model_family=args.model_family
                                                        ),
            "completion": format_completion_with_special_tokens(
                                                        completion=reference_simplification, 
                                                        model_family=args.model_family
                                                        ),
            "system_prompt_id": system_id,
            "user_prompt_id": args.user_prompt_id,
            "metric_name": args.metric_name,
            "source_metric_value": source_metric_value,
            "target_metric_value": target_metric_value
        }


    train_dataset = load_dataset_from_hf(dataset_name, split="train", slice=args.slice_train)
    val_dataset = load_dataset_from_hf(dataset_name, split="validation", slice=args.slice_val)
    test_dataset = load_dataset_from_hf(dataset_name, split="test", slice=args.slice_test)

    train_dataset = train_dataset.map(process_instance)
    val_dataset = val_dataset.map(process_instance)
    test_dataset = test_dataset.map(process_instance)

    print("\n\n *** Before tokenization ***")
    print(">>> train:")
    show_examples(train_dataset, n=1)

    train_dataset = tokenize_dataset(train_dataset, tokenizer, args.max_length)
    val_dataset = tokenize_dataset(val_dataset, tokenizer, args.max_length)
    test_dataset = tokenize_dataset(test_dataset, tokenizer, args.max_length)

    print("\n\n *** After tokenization ***")
    print(">>> train:")
    show_examples(train_dataset, n=1, show_tokens=True)

    print(10*"*", "DEBUG", 10*"*")
    print("Train dataset columns:", train_dataset.column_names)
    print("Validation dataset columns:", val_dataset.column_names)
    print("Test dataset columns:", test_dataset.column_names)

    return train_dataset, val_dataset, test_dataset

def train_model(model, tokenizer, train_dataset, val_dataset, args, output_dir, peft_enabled):

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        max_grad_norm=0.5, # clipping to stabilize
        lr_scheduler_type="cosine",
        warmup_steps=30,
        # fp16=True,
        bf16=True,
        fp16=False,
        logging_steps=args.logging_steps,
        push_to_hub=False,
        report_to=["wandb"],
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=args.logging_steps,
        eval_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        tokenizer=tokenizer,
        callbacks=[PredictionLoggerCallback(
                        tokenizer=tokenizer, 
                        val_dataset=val_dataset, 
                        log_every=args.log_every,
                        num_samples=4,
                        max_length=args.max_length,
                        gen_kwargs=None
                        ),
                    EarlyStoppingCallback(
                        early_stopping_patience=args.patience
                        )
                        ]
                        )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    # val_dataset.save_to_disk(f"{output_dir}/val_dataset")
    wandb.save(output_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    # hyperparams
    parser.add_argument("--model_class", type=str, required=True, choices=["llama", "auto"], help="Model class to use.")
    parser.add_argument("--model_family", type=str, required=True, default="base", choices=["llama", "mistral", "qwen", "base"], help="Model type to choose from.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--slice_train", type=int, default=-1)
    parser.add_argument("--slice_val", type=int, default=-1)
    parser.add_argument("--slice_test", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512, help="Max length of output")
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--wandb_project_name", type=str, default="thesis-SFT")
    parser.add_argument("--wandb_entity", type=str, default="shtosti")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Seed for determenism")
    parser.add_argument("--patience", type=int, default=3, help="patience period for early stopping")
    # for dynamic prompting
    parser.add_argument("--prompting_type", type=str, default="vanilla", choices=["vanilla", "reasoning", "transformations"])
    parser.add_argument("--user_prompt_id", type=str, default="token", choices=["no_token", "token", "token_explanation", "token_explanation_examples"])
    parser.add_argument("--metric_name", type=str, required=True)
    # Paths to external JSON files for control tokens, system prompts, and user prompts
    parser.add_argument("--control_tokens", type=str, required=True)
    parser.add_argument("--system_prompts", type=str, required=True)
    parser.add_argument("--user_prompts", type=str, required=True)
    parser.add_argument("--metric_mapping", type=str, required=True)
    
    # peft flag
    parser.add_argument("--peft", action="store_true", help="Enable PEFT for large models.")

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    if args.peft:
        print("Using PEFT...")
    else:
        print("Not using PEFT...")

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
    output_dir = f"./models/{short_model}-{args.dataset_name}-{args.metric_name}-{args.model_family}-{args.user_prompt_id}-{timestamp}-{wandb_run_id}"
    os.makedirs(output_dir, exist_ok=True)
    print("Saving to:", output_dir)

    with open("last_run_path.txt", "w") as f:
        f.write(output_dir)

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n*** Finetuning {short_model} with {args.dataset_name} ***\n")

    print_gpu_info()

    model, tokenizer = load_and_prepare_model(args.model_family, args.model_name, args.model_class, args.peft, args.max_length)
    train_dataset, val_dataset, test_dataset = load_and_prepare_dataset(args.dataset_name, tokenizer, args)

    print("First 10 input_ids:", train_dataset[0]["input_ids"][:10])
    print("First 10 labels:", train_dataset[0]["labels"][:10])


    example = train_dataset[0]
    print("type of input_ids:", type(example["input_ids"]))
    print("type of labels:", type(example["labels"]))
    print("\n--- DEBUG: Tokenized fields ---")
    assert len(example["input_ids"]) == len(example["labels"]), "Input and label lengths do not match!"
    print("--- Tokenizer vocab size:", tokenizer.vocab_size)
    print("--- input_ids:", example["input_ids"])
    print("--- labels:", example["labels"])
    print("--- decoded input_ids:\n", tokenizer.decode(example["input_ids"], skip_special_tokens=False))
    decoded_labels = tokenizer.decode([token_id for token_id in example["labels"] if token_id != -100], skip_special_tokens=True)
    print("--- decoded labels:\n", decoded_labels)


    train_model(
                model, 
                tokenizer, 
                train_dataset, 
                val_dataset, 
                args, 
                output_dir,
                args.peft
                )

    wandb.finish()


if __name__ == "__main__":
    main()
