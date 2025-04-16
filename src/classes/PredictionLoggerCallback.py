from transformers import TrainerCallback
import wandb
import torch
import random

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

                sample_indices = random.sample(range(len(self.val_dataset)), 2)

                for i in sample_indices:
                    prompt_text = self.val_dataset[i]['prompt']
                    reference_text = self.val_dataset[i]['completion']

                    # Corrected tokenization
                    inputs = self.tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(model.device)

                    try:
                        with torch.cuda.amp.autocast(enabled=False):  # ensure FP32
                            outputs = model(input_ids=inputs["input_ids"])
                            logits = outputs.logits
                            probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)

                            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                                print(f"--- [DEBUG] Invalid probs at step {state.global_step} for sample {i}")
                                print(f"--- [DEBUG] Logits stats — min: {logits.min().item():.4f}, max: {logits.max().item():.4f}, mean: {logits.mean().item():.4f}")
                                continue  # Skip this one
                    

                        # Generate deterministic output
                        output_ids = model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=300, 
                            do_sample=True,
                            # temperature=0.7,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                        
                        prediction_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)

                        # Print to console
                        print(f"\n>>> Prompt:\n{prompt_text}")
                        print(f">>> Reference:\n{reference_text}")
                        print(f">>> Prediction:\n{prediction_text}")
                        print("=" * 80)

                        # Add to list for wandb logging
                        predictions.append(wandb.Table(data=[[
                            state.global_step,
                            prompt_text,
                            reference_text,
                            prediction_text
                        ]], columns=["step", "prompt", "reference", "prediction"]))


                    except Exception as e:
                        print(f"[WARNING] Prediction generation failed at step {state.global_step} for sample {i}: {e}")
                        continue

            # Log to wandb
            for table in predictions:
                wandb.log({"predictions": table}, step=state.global_step)


# from transformers import TrainerCallback
# import wandb
# import torch
# import random

# class PredictionLoggerCallback(TrainerCallback):
#     def __init__(self, tokenizer, val_dataset, log_every=20, num_samples=2, gen_kwargs=None):
#         self.tokenizer = tokenizer
#         self.val_dataset = val_dataset
#         self.log_every = log_every
#         self.num_samples = num_samples
#         self.gen_kwargs = gen_kwargs or {
#             "max_new_tokens": 300,
#             "do_sample": True,
#             "pad_token_id": tokenizer.pad_token_id,
#             "eos_token_id": tokenizer.eos_token_id,
#         }

#     def on_step_end(self, args, state, control, **kwargs):
#         if state.global_step % self.log_every != 0 or state.global_step == 0:
#             return

#         model = kwargs['model']
#         model.eval()

#         sample_indices = random.sample(range(len(self.val_dataset)), self.num_samples)
#         samples = [self.val_dataset[i] for i in sample_indices]
#         prompts = [s['prompt'] for s in samples]
#         references = [s['completion'] for s in samples]

#         try:
#             inputs = self.tokenizer(
#                 prompts,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True
#             ).to(model.device)
#         except Exception as e:
#             print(f"[Tokenization Error] Skipping prediction logging at step {state.global_step}: {e}")
#             return

#         try:
#             with torch.no_grad():
#                 with torch.cuda.amp.autocast(enabled=False):  # Ensure FP32
#                     output_ids = model.generate(
#                         input_ids=inputs["input_ids"],
#                         attention_mask=inputs["attention_mask"],
#                         **self.gen_kwargs
#                     )

#             decoded_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)

#             rows = []
#             for i in range(self.num_samples):
#                 print(f"\n>>> Prompt:\n{prompts[i]}")
#                 print(f">>> Reference:\n{references[i]}")
#                 print(f">>> Prediction:\n{decoded_outputs[i]}")
#                 print("=" * 80)

#                 rows.append([
#                     state.global_step,
#                     prompts[i],
#                     references[i],
#                     decoded_outputs[i]
#                 ])

#             # Log to wandb
#             table = wandb.Table(data=rows, columns=["step", "prompt", "reference", "prediction"])
#             wandb.log({
#                 "predictions": table,
#                 "generation_config": self.gen_kwargs
#             }, step=state.global_step)

#         except Exception as e:
#             print(f"[Generation Error] Failed at step {state.global_step}: {e}")
