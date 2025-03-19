from dataset_handler.py import DatasetHandler
from model_handler.py import ModelHandler
from trainer_handler.py import TrainerHandler
from optimizer.py import HyperparameterOptimizer

def main():
    # Data and Model Configuration
    dataset_path = "your_dataset.csv"
    model_name = "meta-llama/LLaMA-8b"

    # Initialize Handlers
    dataset_handler = DatasetHandler(dataset_path)
    dataset_handler.load_data()

    model_handler = ModelHandler(model_name)
    model_handler.load_model()

    # Hyperparameter Optimization
    optimizer = HyperparameterOptimizer(model_handler, dataset_handler)
    best_params = optimizer.optimize(n_trials=5)
    print("Best Hyperparameters:", best_params)

    # Fine-tuning
    train_data, eval_data = dataset_handler.get_datasets()
    model, tokenizer = model_handler.get_model_and_tokenizer()

    trainer_handler = TrainerHandler(model, tokenizer, train_data, eval_data)
    trainer_handler.prepare_trainer(
        learning_rate=best_params['learning_rate'],
        num_epochs=best_params['num_epochs']
    )

    trainer_handler.train()
    eval_metrics = trainer_handler.evaluate()
    print("Evaluation Metrics:", eval_metrics)

if __name__ == "__main__":
    main()
