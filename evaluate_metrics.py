import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

from multi_classifier import LABEL_MAP, classify_question

# Mapping from numeric label to type
CATEGORY_TYPE = {
    0: "other",
    1: "instrumental",
    2: "executive",
    3: "instrumental",
    4: "executive",
    5: "instrumental",
    6: "instrumental",
    7: "executive",
    8: "other",
    9: "instrumental",
}

# Mapping from category text to type
category_to_type = {
    "Chatbot interaction": "other",
    "Concept Comprehension": "instrumental",
    "Copying Notebook Questions": "executive",
    "Code Comprehension": "instrumental",
    "Fix this code / error": "executive",
    "Error Comprehension": "instrumental",
    "Task Related Delegation": "instrumental",
    "Pasting Code Without Explanation": "executive",
    "Random": "other",
    "Question Comprehension": "instrumental",
}


def merge_labels(label):
    """
    This function merges Code Comprehension and Concept Comprehension into one label.
    For all other labels, the original label is returned.
    """
    if label in ["Code Comprehension", "Concept Comprehension"]:
        return "Code/Concept Comprehension"
    return label


def evaluate_classifier(ground_truth_file, report_file):
    """
    Evaluates the classifier against a ground truth Excel file.

    The file is expected to contain at least two columns:
      - 'question': the text question
      - 'label': the expected category (string)

    For each question the classifier provides:
      - Category evaluation: accuracy, precision, recall, and F1-score.
      - Type evaluation: accuracy, precision, recall, and F1-score (using the type mappings).
      - Merged evaluation: where "Code Comprehension" and "Concept Comprehension" are merged.

    The detailed results and classification reports are written to the report_file.
    """
    df = pd.read_excel(ground_truth_file)
    total = len(df)

    true_categories = []
    pred_categories = []

    true_types = []
    pred_types = []

    merged_true = []
    merged_pred = []

    for index, row in df.iterrows():
        question = row["question"]
        true_label = row["label"]  # Ground-truth category as text
        pred_num = classify_question(question)
        pred_label = LABEL_MAP[pred_num]

        true_categories.append(true_label)
        pred_categories.append(pred_label)

        # Map to types using the dictionaries defined
        true_type = category_to_type[true_label]
        pred_type = CATEGORY_TYPE[pred_num]
        true_types.append(true_type)
        pred_types.append(pred_type)

        # Handle merged labels evaluation for Code/Concept classes
        merged_true.append(merge_labels(true_label))
        merged_pred.append(merge_labels(pred_label))

    # Compute overall accuracy
    category_accuracy = accuracy_score(true_categories, pred_categories) * 100
    type_accuracy = accuracy_score(true_types, pred_types) * 100
    merged_accuracy = accuracy_score(merged_true, merged_pred) * 100

    # Compute precision, recall and f1 for categories, types, and merged labels
    cat_precision, cat_recall, cat_f1, _ = precision_recall_fscore_support(
        true_categories, pred_categories, average="weighted", zero_division=0
    )
    type_precision, type_recall, type_f1, _ = precision_recall_fscore_support(
        true_types, pred_types, average="weighted", zero_division=0
    )
    merged_precision, merged_recall, merged_f1, _ = precision_recall_fscore_support(
        merged_true, merged_pred, average="weighted", zero_division=0
    )

    # Create detailed reports using classification_report
    cat_report = classification_report(
        true_categories, pred_categories, zero_division=0
    )
    type_report = classification_report(true_types, pred_types, zero_division=0)
    merged_report = classification_report(merged_true, merged_pred, zero_division=0)

    # Write results to the report file
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== Classification Evaluation Report ===\n\n")

        f.write("1. Category Evaluation\n")
        f.write("--------------------------\n")
        f.write(f"Accuracy: {category_accuracy:.2f}%\n")
        f.write(f"Precision: {cat_precision:.4f}\n")
        f.write(f"Recall: {cat_recall:.4f}\n")
        f.write(f"F1-Score: {cat_f1:.4f}\n\n")
        f.write("Detailed Report:\n")
        f.write(cat_report + "\n\n")

        f.write("2. Merged Evaluation (Code/Concept Comprehension Unified)\n")
        f.write("-----------------------------------------------------------\n")
        f.write(f"Accuracy: {merged_accuracy:.2f}%\n")
        f.write(f"Precision: {merged_precision:.4f}\n")
        f.write(f"Recall: {merged_recall:.4f}\n")
        f.write(f"F1-Score: {merged_f1:.4f}\n\n")
        f.write("Detailed Report:\n")
        f.write(merged_report + "\n\n")

        f.write("3. Type Evaluation\n")
        f.write("-------------------\n")
        f.write(f"Accuracy: {type_accuracy:.2f}%\n")
        f.write(f"Precision: {type_precision:.4f}\n")
        f.write(f"Recall: {type_recall:.4f}\n")
        f.write(f"F1-Score: {type_f1:.4f}\n\n")
        f.write("Detailed Report:\n")
        f.write(type_report + "\n")

    print("Evaluation complete.")
    print(f"Category Accuracy: {category_accuracy:.2f}%")
    print(f"Merged Accuracy (Code/Concept unified): {merged_accuracy:.2f}%")
    print(f"Type Accuracy: {type_accuracy:.2f}%")
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    # Modify these paths as needed.
    ground_truth_file = "/Users/thomvandervelden/dev/JELAI_classifier/ground_truth.xlsx"
    report_file = "/Users/thomvandervelden/dev/JELAI_classifier/evaluation_report.txt"
    evaluate_classifier(ground_truth_file, report_file)
