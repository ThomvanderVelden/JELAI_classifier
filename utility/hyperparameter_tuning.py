"""
Hyperparameter Tuning for Dutch Questions Multi-Classifier
---------------------------------------------------------
This script uses Optuna to find the optimal hyperparameters for the 
RobBERT-based multi-class classifier.

Visualizations of the optimization process are created and saved for use in academic papers.

Usage:
  python hyperparameter_tuning.py
"""

import json
import os
import time
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    EarlyStoppingCallback,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    TrainingArguments,
)

# Import components from the multi_classifier script
from classifier import (
    LABEL_MAP,
    CustomTrainer,
    QuestionDataset,
    device,
    encode_texts,
    load_data,
)

# Configuration
DATA_PATH = "data/questions_before_midterm.csv"
STUDY_NAME = "robbert_hyperparameter_optimization"
N_TRIALS = 25  # Change this value to use more or less compute
MODEL_NAME = "pdelobelle/robbert-v2-dutch-base"
RESULTS_DIR = "hyperparameter_results"
STORAGE_PATH = os.path.join(RESULTS_DIR, f"{STUDY_NAME}.db")
BEST_MODEL_DIR = os.path.join(RESULTS_DIR, "best_model")
MAX_EPOCHS = 10  # Limit epochs for each trial to conserve resources

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)


def prepare_data():
    """
    Load and prepare the dataset for hyperparameter tuning.
    """

    # Load data
    df = load_data(DATA_PATH)

    # Split data into train and validation sets
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Compute class weights to handle imbalance
    classes = np.arange(10)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"],
    )
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    return train_df, val_df, class_weights_tensor


def objective(trial, train_df, val_df, class_weights_tensor):
    """
    Objective function for Optuna optimization. Takes a trial object and returns
    validation accuracy as the optimization metric.
    """
    trial_start_time = time.time()

    # Define hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.2, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2)
    gradient_accumulation_steps = trial.suggest_categorical(
        "gradient_accumulation_steps", [2, 4, 8]
    )

    # Create unique output directory for this trial
    output_dir = os.path.join(RESULTS_DIR, f"trial_{trial.number}")
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=10)
    model.to(device)

    # Tokenize data
    train_encodings = encode_texts(train_df["question"].tolist(), tokenizer)
    val_encodings = encode_texts(val_df["question"].tolist(), tokenizer)

    # Convert labels to tensors
    train_labels = torch.tensor(train_df["label"].tolist())
    val_labels = torch.tensor(val_df["label"].tolist())

    # Create datasets
    train_dataset = QuestionDataset(train_encodings, train_labels)
    val_dataset = QuestionDataset(val_encodings, val_labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=MAX_EPOCHS,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="epoch",
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        report_to="none",  # Disable logging
        save_total_limit=1,  # Only keep the best model to save disk space
    )

    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        class_weights_tensor=class_weights_tensor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train the model
    trainer.train()

    # Evaluate
    eval_result = trainer.evaluate()

    # Get predictions for more detailed metrics
    y_preds = []
    y_true = val_df["label"].tolist()

    # Run predictions in batches to avoid memory issues
    for i in range(0, len(val_dataset), batch_size):
        batch = {k: v[i : i + batch_size].to(device) for k, v in val_encodings.items()}
        with torch.no_grad():
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            y_preds.extend(preds)

    # Compute additional metrics
    accuracy = accuracy_score(y_true, y_preds)

    # Log metrics
    trial.set_user_attr("eval_loss", eval_result["eval_loss"])
    trial.set_user_attr("accuracy", accuracy)

    # Calculate and save confusion matrix for this trial
    cm = confusion_matrix(y_true, y_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[LABEL_MAP[i] for i in range(10)],
        yticklabels=[LABEL_MAP[i] for i in range(10)],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Trial {trial.number} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # Clean up to save memory
    del model, trainer, train_dataset, val_dataset
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    elapsed_time = time.time() - trial_start_time
    print(
        f"Trial {trial.number} finished in {elapsed_time:.2f} seconds with accuracy: {accuracy:.4f}"
    )

    # Return accuracy as the optimization metric (higher is better)
    return accuracy


def visualize_optimization_history(study):
    """Create visualizations of the optimization process."""
    # Create visualizations directory
    viz_dir = os.path.join(RESULTS_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # 1. Optimization history plot
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "optimization_history.png"))
    plt.close()

    # 2. Parameter importance plot
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "parameter_importance.png"))
    plt.close()

    # 3. Parallel coordinate plot
    plt.figure(figsize=(15, 8))
    optuna.visualization.matplotlib.plot_parallel_coordinate(
        study,
        params=[
            "learning_rate",
            "batch_size",
            "weight_decay",
            "warmup_ratio",
            "gradient_accumulation_steps",
        ],
    )
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "parallel_coordinate.png"))
    plt.close()

    # 4. Slice plot
    plt.figure(figsize=(15, 10))
    optuna.visualization.matplotlib.plot_slice(study)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "slice_plot.png"))
    plt.close()

    # 5. Learning rate vs accuracy scatter plot with batch size as color
    plt.figure(figsize=(10, 6))
    trials_df = study.trials_dataframe()
    sns.scatterplot(
        x="params_learning_rate",
        y="user_attrs_accuracy",
        hue="params_batch_size",
        palette="viridis",
        size="params_weight_decay",
        sizes=(50, 200),
        data=trials_df,
    )
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Validation Accuracy")
    plt.title("Learning Rate vs Accuracy by Batch Size")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "lr_vs_accuracy.png"))
    plt.close()

    # Save the trial results as CSV
    trials_df.to_csv(os.path.join(viz_dir, "hyperparameter_trials.csv"))

    print(f"Visualizations saved to {viz_dir}")


def save_best_hyperparameters(study):
    """Save the best hyperparameters to a JSON file."""
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial

    # Add some metadata
    results = {
        "best_hyperparameters": best_params,
        "best_accuracy": best_value,
        "best_trial_number": best_trial.number,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": MODEL_NAME,
        "n_trials": len(study.trials),
    }

    # Save as JSON
    params_file = os.path.join(RESULTS_DIR, "best_hyperparameters.json")
    with open(params_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Best hyperparameters saved to {params_file}")
    print(f"Best accuracy: {best_value:.4f}")
    print(f"Best hyperparameters: {best_params}")


def main():
    start_time = time.time()

    print("Starting hyperparameter optimization...")
    print(f"Device: {device}, Storage: {STORAGE_PATH}")
    print(f"Number of trials: {N_TRIALS}")

    # Prepare data
    train_df, val_df, class_weights_tensor = prepare_data()

    # Set up optuna storage
    storage = f"sqlite:///{STORAGE_PATH}"

    # Create a partial function for the objective
    objective_func = partial(
        objective,
        train_df=train_df,
        val_df=val_df,
        class_weights_tensor=class_weights_tensor,
    )

    # Create or load study
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",  # Maximize accuracy
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Run optimization
    try:
        study.optimize(objective_func, n_trials=N_TRIALS, timeout=None)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    # Save best parameters to file
    save_best_hyperparameters(study)

    # Create visualizations
    visualize_optimization_history(study)

    # Report total time taken
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")


if __name__ == "__main__":
    main()
