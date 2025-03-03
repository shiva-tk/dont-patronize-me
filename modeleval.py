from abc import ABC, abstractmethod
from datasets import Dataset, DatasetDict, load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torch
from sklearn.metrics import f1_score
from abc import ABC, abstractmethod
from datasets import Dataset
from collections import Counter

class AugmentationStrategy(ABC):

    @abstractmethod
    def augment_data(self, data: Dataset) -> Dataset:
        pass

# Given a model setup, it evaluates it using cross-validation
def evaluate_model(train_df="data/train.csv", augmentation_strategy=None, epochs=3, learning_rate=3e-5, batch_size=8, lr_scheduler="linear"):
    # Load dataset
    df = pd.read_csv(train_df)
    dataset = Dataset.from_pandas(df)

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    # Check for GPU (CUDA)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device.upper()}")

    # Prepare for cross-validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset, dataset['label'])):
        print(f"Training fold {fold+1}...")

        train_dataset = dataset.select(train_idx)
        val_dataset = dataset.select(val_idx)

        # Apply augmentation if provided
        if augmentation_strategy:
            train_dataset = augmentation_strategy.augment_data(dataset)

        print(Counter(train_dataset["label"]))

        train_dataset = dataset.map(tokenize_function, batched=True)
        val_dataset = dataset.map(tokenize_function, batched=True)

        print(train_dataset)
        print(val_dataset)


        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)


        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=f"./results_fold_{fold}",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"./logs_fold_{fold}",
            lr_scheduler_type=lr_scheduler,
            logging_steps=10,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            report_to="none"
        )

        def compute_metrics(p):
            preds = p.predictions.argmax(-1)
            f1 = f1_score(p.label_ids, preds, pos_label=1)
            return {"eval_f1": f1}

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        # Train and evaluate the model
        trainer.train()

        # Evaluate the model
        eval_result = trainer.evaluate()
        results.append(eval_result)

        print(f"Fold {fold+1} evaluation results: {eval_result}")

    # Calculate average results from cross-validation
    avg_results = {key: sum(result[key] for result in results) / len(results) for key in results[0]}
    print("Average evaluation results from cross-validation:", avg_results)

    return avg_results
