import pandas as pd

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


def evaluate_classifier(ground_truth_file, report_file):
    # Adjust this if your Excel file has a different structure or file extension.
    df = pd.read_excel(ground_truth_file)

    total = len(df)
    correct_category = 0
    correct_type = 0
    correct_merged = 0  # New counter for merged accuracy
    mismatches_category = []
    mismatches_type = []
    mismatches_merged = []  # New list for merged mismatches

    for index, row in df.iterrows():
        question = row["question"]
        true_label = row["label"]  # assuming this column contains the category as text
        pred_num = classify_question(question)
        pred_label = LABEL_MAP[pred_num]

        # Check category correctness
        if pred_label == true_label:
            correct_category += 1
        else:
            mismatches_category.append(
                {
                    "question": question,
                    "true_label": true_label,
                    "pred_label": pred_label,
                }
            )

        # Check merged correctness for Code/Concept Comprehension
        if pred_label == true_label or (
            pred_label in ["Code Comprehension", "Concept Comprehension"]
            and true_label in ["Code Comprehension", "Concept Comprehension"]
        ):
            correct_merged += 1
        else:
            mismatches_merged.append(
                {
                    "question": question,
                    "true_label": true_label,
                    "pred_label": pred_label,
                }
            )

        # Check type correctness
        pred_type = CATEGORY_TYPE[pred_num]
        true_type = category_to_type[true_label]
        if pred_type == true_type:
            correct_type += 1
        else:
            mismatches_type.append(
                {
                    "question": question,
                    "true_type": true_type,
                    "pred_type": pred_type,
                }
            )

    category_accuracy = correct_category / total * 100
    merged_accuracy = correct_merged / total * 100  # Compute merged accuracy
    type_accuracy = correct_type / total * 100

    with open(report_file, "w", encoding="utf-8") as f:
        # Write category accuracy and mismatches
        f.write(f"Category Accuracy: {category_accuracy:.2f}%\n\n")
        f.write("Misclassified Categories:\n")
        for mm in mismatches_category:
            f.write(f"Question: {mm['question']}\n")
            f.write(
                f"True Label: {mm['true_label']} | Predicted Label: {mm['pred_label']}\n\n"
            )

        # Write merged accuracy and mismatches
        f.write(
            f"Merged Accuracy (Code/Concept Comprehension unified): {merged_accuracy:.2f}%\n\n"
        )
        f.write("Misclassified Merged Labels:\n")
        for mm in mismatches_merged:
            f.write(f"Question: {mm['question']}\n")
            f.write(
                f"True Label: {mm['true_label']} | Predicted Label: {mm['pred_label']}\n\n"
            )

        # Write type accuracy and mismatches
        f.write(f"Type Accuracy: {type_accuracy:.2f}%\n\n")
        f.write("Misclassified Types:\n")
        for mm in mismatches_type:
            f.write(f"Question: {mm['question']}\n")
            f.write(
                f"True Type: {mm['true_type']} | Predicted Type: {mm['pred_type']}\n\n"
            )

    print(f"Evaluation complete. Category Accuracy: {category_accuracy:.2f}%")
    print(f"Merged Accuracy (Code/Concept unified): {merged_accuracy:.2f}%")
    print(f"Type Accuracy: {type_accuracy:.2f}%")
    print(f"Report saved to: {report_file}")


# Example usage:
if __name__ == "__main__":
    ground_truth_file = (
        "/Users/thomvandervelden/dev/JELAI_classifier/data/ground_truth.xlsx"
    )
    report_file = "/Users/thomvandervelden/dev/JELAI_classifier/evaluation_report.txt"
    evaluate_classifier(ground_truth_file, report_file)
