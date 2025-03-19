import optuna
from transformers import TrainingArguments, Trainer

class HyperparameterOptimizer:
    def __init__(self, model_handler, dataset_handler, output_dir="./llama_optuna"):
        self.model_handler = model_handler
        self.dataset_handler = dataset_handler
        self.output_dir = output_dir

    def objective(self, trial):
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
        num_epochs = trial.suggest_int("num_epochs", 2, 5)

        model, tokenizer = self.model_handler.get_model_and_tokenizer()
        train_dataset, eval_dataset = self.dataset_handler.get_datasets()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            logging_dir="./logs",
            fp16=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer
        )

        trainer.train()
        metrics = trainer.evaluate()
        return metrics["eval_loss"]

    def optimize(self, n_trials=10):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params
