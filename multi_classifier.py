"""
Multi-Class Classification of Dutch Questions Using RobBERT and Transformers Trainer
-----------------------------------------------------------------------------------
1) Reads CSV data, containing 'question' (text) and 'label' (integer class, 0..9).
2) Splits data into train and validation sets.
3) Computes class weights for all 10 classes (if desired) to handle imbalance.
4) Uses a custom Trainer to incorporate the class weights into the loss function.
5) (Optionally) trains, evaluates, and then saves the model checkpoint.
6) Loads the model checkpoint to classify new questions without retraining.
-----------------------------------------------------------------------------------
NOTE: Make sure you have the necessary libraries installed:
  pip install 'transformers[torch]' accelerate -U scikit-learn pandas
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)

# ----------------------------
# Configuration
# ----------------------------
DO_TRAIN = (
    False  # <--- Toggle. If True: train model. If False: load from existing checkpoint.
)
CHECKPOINT_DIR = "my_saved_model"
DATA_PATH = (
    "/Users/thomvandervelden/dev/JELAI_classifier/data/questions_before_midterm.csv"
)
INFERENCE_FILE = (
    "/Users/thomvandervelden/dev/JELAI_classifier/data/questions_after_midterm.csv"
)

# Output file to write predictions
OUTPUT_FILE = "/Users/thomvandervelden/dev/JELAI_classifier/predictions.txt"

# Mapping from numeric label to class name
LABEL_MAP = {
    0: "Chatbot interaction",
    1: "Concept Comprehension",
    2: "Copying Notebook Questions",
    3: "Code Comprehension",
    4: "Fix this code / error",
    5: "Error Comprehension",
    6: "Task Related Delegation",
    7: "Pasting Code Without Explanation",
    8: "Random",
    9: "Question Comprehension",
}


# --------------------------------------------------------
# Decide on which device to train (CPU, MPS on Apple Silicon, or CUDA for NVIDIA GPUs)
# --------------------------------------------------------
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------------
# 1. LOAD AND INSPECT THE DATA
# --------------------------------------------------------
def load_data(file_path):
    """
    Loads CSV data, expects two columns:
      - 'question' (string)
      - 'label' (integer from 0..9)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    df = pd.read_csv(
        "data/questions_before_midterm.csv",
        sep=",",
        quotechar="'",
        escapechar="\\",
        engine="python",
    )
    print("Available columns:", df.columns.tolist())
    return df


df = load_data(DATA_PATH)

print("\nLabel distribution:")
print(df["label"].value_counts())


# --------------------------------------------------------
# 2. SPLIT DATA INTO TRAIN/VAL
# --------------------------------------------------------
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

# --------------------------------------------------------
# 3. COMPUTE CLASS WEIGHTS
# --------------------------------------------------------

classes = np.arange(10)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"],
)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)


# --------------------------------------------------------
# 4. LOAD / BUILD MODEL AND TOKENIZER
# --------------------------------------------------------
model_name = "pdelobelle/robbert-v2-dutch-base"

if DO_TRAIN:
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=10)
    model.to(device)
else:
    # Loading from previously saved checkpoint
    if not os.path.exists(CHECKPOINT_DIR):
        raise FileNotFoundError(
            f"You set DO_TRAIN=False, but no checkpoint directory found at '{CHECKPOINT_DIR}'"
        )
    tokenizer = RobertaTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = RobertaForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
    model.to(device)


# --------------------------------------------------------
# 4. ENCODE THE TEXT
# --------------------------------------------------------
def encode_texts(texts, tokenizer, max_length=128):
    """
    Tokenizes and encodes the raw text strings using a specified max_length.
    Pads/truncates to ensure consistent tensor shapes.
    Returns a dictionary of tensors (input_ids, attention_mask, etc.)
    """
    return tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )


train_encodings = encode_texts(train_df["question"].tolist(), tokenizer)
val_encodings = encode_texts(val_df["question"].tolist(), tokenizer)

# Convert labels to tensors
train_labels = torch.tensor(train_df["label"].tolist())
val_labels = torch.tensor(val_df["label"].tolist())


# --------------------------------------------------------
# 6. CREATE DATASET CLASS
# --------------------------------------------------------
class QuestionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = QuestionDataset(train_encodings, train_labels)
val_dataset = QuestionDataset(val_encodings, val_labels)


# --------------------------------------------------------
# 7. DEFINE TRAINING ARGUMENTS
# --------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=3.100652726283406e-05,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=25,
    weight_decay=0.01923838995198125,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="epoch",
    warmup_ratio=0.111937633641739,
    gradient_accumulation_steps=2,
)


# --------------------------------------------------------
# 8. CUSTOM TRAINER FOR WEIGHTED LOSS
# --------------------------------------------------------
class CustomTrainer(Trainer):
    """
    A custom Trainer that applies our class weights in the CrossEntropyLoss.
    """

    def __init__(self, class_weights_tensor=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights_tensor = class_weights_tensor

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Weighted cross-entropy for multi-class (10 labels)
        # Use provided weights or default to the global class_weights_tensor
        weights = (
            self.class_weights_tensor
            if self.class_weights_tensor is not None
            else class_weights_tensor
        )
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, 10), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# --------------------------------------------------------
# 9. INSTANTIATE TRAINER & (OPTIONALLY) TRAIN
# --------------------------------------------------------
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

if DO_TRAIN:
    # Train model
    trainer.train()
    # Evaluate
    trainer.evaluate()
    # Save the trained model & tokenizer
    trainer.save_model(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    print(f"\nModel + tokenizer saved to '{CHECKPOINT_DIR}'")
else:
    # Skip training/evaluation
    print("\nSkipped training; model loaded from checkpoint.")
    # Optionally do a quick evaluation:
    trainer.evaluate()


# --------------------------------------------------------
# 10. INFERENCE FUNCTION
# --------------------------------------------------------
def classify_question(question):
    """
    Given a single question (string),
    1) Tokenize and encode the input
    2) Run inference with the trained/loaded model
    3) Predict the label via argmax
    4) Print debug info (logits, probabilities, predicted label)
    5) Return the numeric label (0..9) or a string if you have a label map
    """
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        # Pick the class with the highest probability
        predicted_class = probabilities.argmax(dim=1).item()

    print("\nPrediction details:")
    print(f" Raw logits: {logits[0].tolist()}")
    print(f" Probabilities: {probabilities[0].tolist()}")
    print(f" Predicted class: {predicted_class} ({LABEL_MAP[predicted_class]})")

    return predicted_class


# --------------------------------------------------------
# 11. TEST WITH EXAMPLE QUESTIONS (INFERENCE)
# --------------------------------------------------------
test_questions = [
    "Hoi",
    "Test",
    "hoe kan ik de lengte van de naam bepalen",
    "Wat is fout aan deze code kgsbrajkmnocdjbhshuicwuiwe",
    "Hoe kan ik een nieuwe variabele aanmaken?",
    "doe dit met een for of while loop",
]

print("\nTesting multiple questions after training/loading model:")
for q in test_questions:
    print(f"\nQuestion: {q}")
    result = classify_question(q)


# --------------------------------------------------------
# 12. INFERENCE FROM A FILE, WRITE OUTPUT
# --------------------------------------------------------
# def classify_file(input_file_path: str, output_file_path: str):
#     """
#     Reads a file line-by-line (each line = one question).
#     Classifies each line using the model and writes results to 'output_file_path'.
#     """
#     if not os.path.exists(input_file_path):
#         print(f"[WARN] File '{input_file_path}' not found. Skipping.")
#         return

#     print(f"\n--- Classifying questions from file: {input_file_path} ---")
#     print(f"Results will be written to: {output_file_path}")

#     with open(input_file_path, "r", encoding="utf-8") as infile:
#         lines = infile.read().splitlines()

#     with open(output_file_path, "w", encoding="utf-8") as outfile:
#         for i, line in enumerate(lines, start=1):
#             line = line.strip()
#             if not line:
#                 continue

#             pred_num = classify_question(line)
#             pred_class_name = LABEL_MAP[pred_num]

#             # Print to console (optional)
#             print(f"{i}. {line} => {pred_class_name}")

#             # Write to output file
#             outfile.write(f"Line {i}: {line}\n")
#             outfile.write(f" Predicted Class => {pred_class_name}\n\n")


# # Finally, run inference on the file and write results
# classify_file(INFERENCE_FILE, OUTPUT_FILE)
