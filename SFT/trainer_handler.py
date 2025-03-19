from transformers import TrainingArguments, Trainer
# from torchmetrics import BLEUScore
import torch

class TrainerHandler:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, output_dir="./llama_finetuned"):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.trainer = None

    def prepare_trainer(self, learning_rate=2e-5, num_epochs=3, batch_size=2):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,
            logging_dir="./logs",
            logging_steps=50,
            save_strategy="epoch",
            fp16=True
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer
        )

    def train(self):
        self.trainer.train()

    def evaluate(self):
        metrics = self.trainer.evaluate()
        return metrics
