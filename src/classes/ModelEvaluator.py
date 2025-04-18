from transformers import Trainer
from nltk.translate.bleu_score import corpus_bleu
import torch
import wandb

# Evaluation class (to be used after training)
class ModelEvaluator:
    def __init__(self, model, tokenizer, test_dataset, metric="bleu", gen_kwargs=None):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.metric = metric

        # Generation arguments
        self.gen_kwargs = gen_kwargs or {
            "max_new_tokens": 300,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def evaluate(self):
        self.model.eval()  # Switch model to evaluation mode
        prompts, references = [], []
        for idx in range(len(self.test_dataset)):
            sample = self.test_dataset[idx]
            prompt = sample.get("prompt", "")
            reference = sample.get("completion", "")
            if prompt:
                prompts.append(prompt)
                references.append(reference)

        if not prompts:
            return

        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        # Generate predictions
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                **self.gen_kwargs
            )

        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Evaluate metrics
        if self.metric == "bleu":
            references_list = [[ref.split()] for ref in references]
            predictions_list = [pred.split() for pred in decoded_preds]
            bleu_score = corpus_bleu(references_list, predictions_list)
            print(f"Final BLEU score: {bleu_score:.4f}")
            return bleu_score
        else:
            print("Unsupported metric, implement other metrics if needed!")
            return None


# After your training is complete, you can run evaluation:

def run_evaluation(model, tokenizer, test_dataset):
    evaluator = ModelEvaluator(
        model=model, 
        tokenizer=tokenizer, 
        test_dataset=test_dataset,  # Your test dataset here
        metric="bleu"  # You can change this to other metrics like ROUGE, etc.
    )
    
    bleu_score = evaluator.evaluate()  # Run evaluation
    # Optionally, log the results to wandb:
    wandb.log({"test_bleu_score": bleu_score})


def main():
    # Training steps...
    # After training completes, let's load the test dataset and evaluate

    # Assuming test_dataset is loaded similarly as train/val datasets
    test_dataset = load_and_prepare_dataset(args.dataset_name, tokenizer, args)  # Add the logic to load test data
    
    # Load the trained model
    model = LlamaForCausalLM.from_pretrained(output_dir)
    
    # Run the evaluation after training
    run_evaluation(model, tokenizer, test_dataset["test"])

