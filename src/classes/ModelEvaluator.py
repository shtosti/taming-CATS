import torch
import wandb
import warnings
from tqdm import tqdm


class ModelEvaluator:
    def __init__(self, model, tokenizer, test_dataset, max_length=512, batch_size=2, gen_kwargs=None):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.max_length = max_length
        self.batch_size = batch_size
        self.gen_kwargs = gen_kwargs or {
            "max_new_tokens": 300,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def get_prompts_and_references(self):
        prompts, references = [], []
        
        # Iterate through the entire test dataset
        for sample in self.test_dataset:
            # Handle dict or tuple-style samples
            if isinstance(sample, dict):
                prompt = sample.get("prompt", "")
                reference = sample.get("completion", "")
            elif isinstance(sample, tuple):
                prompt = sample[0] if isinstance(sample[0], str) else str(sample[0])
                reference = sample[1] if len(sample) > 1 else ""
            else:
                continue

            if prompt:
                prompts.append(prompt)
                references.append(reference)

        return prompts, references

    def get_predictions(self, prompts):
        all_predictions = []
        
        # Process in batches
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Evaluating", ncols=100):
            batch_prompts = prompts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.model.device)

            # Generate the predictions
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **self.gen_kwargs
                )

            # Slice generated tokens to remove the prompt portion
            prompt_length = inputs["input_ids"].shape[1]
            generated_only_ids = output_ids[:, prompt_length:]

            decoded_preds = self.tokenizer.batch_decode(
                generated_only_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            all_predictions.extend(decoded_preds)

        return all_predictions

    def evaluate(self):
        # Get the full list of prompts and references from the test set
        prompts, references = self.get_prompts_and_references()

        if not prompts:
            return [], [], []

        # Get model predictions for the entire dataset
        predictions = self.get_predictions(prompts)

        # Store results and print them
        results = []
        for prompt, reference, prediction in zip(prompts, references, predictions):
            results.append({
                "prompt": prompt,
                "reference": reference,
                "prediction": prediction
            })

        # Log results to wandb in chunks or as a table
        table_data = [[r["prompt"], r["reference"], r["prediction"]] for r in results]
        table = wandb.Table(data=table_data, columns=["prompt", "reference", "prediction"])

        # table = wandb.Table(data=results, columns=["prompt", "reference", "prediction"])
        wandb.log({"evaluation": table}, step=0)
        
        return prompts, references, predictions
