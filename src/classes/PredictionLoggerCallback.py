from transformers import TrainerCallback
import wandb
import torch

class PredictionLoggerCallback(TrainerCallback):
    def __init__(self, tokenizer, val_dataset, log_every=20):
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.log_every = log_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every == 0 and state.global_step > 0:
            model = kwargs['model']
            model.eval()
            predictions = []

            with torch.no_grad():
                for i in range(1):  # log 1 example
                    prompt_text = self.val_dataset[i]['prompt']
                    reference_text = self.val_dataset[i]['completion']

                    # Tokenize and move to device
                    input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
                    
                    # Generate deterministic output
                    output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=False)
                    prediction_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

                    # Print to console
                    print(f"\nPrompt:\n{prompt_text}")
                    print(f"Reference:\n{reference_text}")
                    print(f"Prediction:\n{prediction_text}")
                    print("=" * 80)

                    # Add to list for wandb logging
                    predictions.append(wandb.Table(data=[[
                        state.global_step,
                        prompt_text,
                        reference_text,
                        prediction_text
                    ]], columns=["step", "prompt", "reference", "prediction"]))

            # Log to wandb
            for table in predictions:
                wandb.log({"predictions": table}, step=state.global_step)
