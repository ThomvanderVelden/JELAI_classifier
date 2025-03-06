"""
Comprehensive Evaluation for Dutch Questions Classifier
------------------------------------------------------
This script produces a full evaluation report with metrics and visualizations:
- Accuracy, precision, recall, F1 for all categories and types
- Confusion matrices
- Error analysis with misclassified examples
- Interactive HTML report with charts

Usage:
  python evaluate.py --input ground_truth.xlsx --output evaluation_report
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from classifier import LABEL_MAP, classify_question

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


def prepare_output_dirs(output_dir):
    """Create directory structure for output files"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    return os.path.join(output_dir, "images"), os.path.join(output_dir, "data")


def create_confusion_matrix_plot(
    y_true, y_pred, labels, title, figsize=(10, 8), cmap="Blues", file_path=None
):
    """Create and save a confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path, dpi=150)
        plt.close()
        return file_path

    return plt.gcf()


def create_metrics_bar_chart(metrics_dict, title, figsize=(12, 6), file_path=None):
    """Create and save a bar chart with precision, recall, and F1 for each label"""
    labels = list(metrics_dict.keys())
    precision_vals = [metrics_dict[label]["precision"] for label in labels]
    recall_vals = [metrics_dict[label]["recall"] for label in labels]
    f1_vals = [metrics_dict[label]["f1-score"] for label in labels]

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    plt.figure(figsize=figsize)
    plt.bar(x - width, precision_vals, width, label="Precision")
    plt.bar(x, recall_vals, width, label="Recall")
    plt.bar(x + width, f1_vals, width, label="F1-score")

    plt.ylabel("Score")
    plt.title(title)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path, dpi=150)
        plt.close()
        return file_path

    return plt.gcf()


def create_distribution_plot(
    true_labels, pred_labels, title, figsize=(12, 6), file_path=None
):
    """Create a bar chart comparing the distribution of true and predicted labels"""
    # Count occurrences of each label
    true_counts = pd.Series(true_labels).value_counts().sort_index()
    pred_counts = pd.Series(pred_labels).value_counts().sort_index()

    # Combine into a dataframe for plotting
    df = pd.DataFrame({"True": true_counts, "Predicted": pred_counts}).fillna(0)

    plt.figure(figsize=figsize)
    df.plot(kind="bar", figsize=figsize)
    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel("Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path, dpi=150)
        plt.close()
        return file_path

    return plt.gcf()


def extract_metrics_from_report(report_dict):
    """Extract metrics from sklearn classification report dict"""
    metrics = {}
    for label, values in report_dict.items():
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            metrics[label] = {
                "precision": values["precision"],
                "recall": values["recall"],
                "f1-score": values["f1-score"],
                "support": values["support"],
            }
    return metrics


def process_misclassifications(questions, true_labels, pred_labels, labels_list=None):
    """Process and return misclassified examples"""
    misclassified = []
    for q, true, pred in zip(questions, true_labels, pred_labels):
        if true != pred:
            misclassified.append(
                {"question": q, "true_label": true, "predicted_label": pred}
            )

    # If labels list is provided, sort misclassifications by label
    if labels_list:
        sorted_misclassified = []
        for label in labels_list:
            for item in misclassified:
                if item["true_label"] == label:
                    sorted_misclassified.append(item)
        return sorted_misclassified

    return misclassified


def evaluate_classifier(ground_truth_file, output_dir):
    """
    Comprehensive evaluation of the classifier with visualizations and detailed metrics.

    Args:
        ground_truth_file: Path to Excel file with ground truth data
        output_dir: Directory to save the evaluation report and visualizations
    """
    print(f"Loading ground truth data from {ground_truth_file}...")
    df = pd.read_excel(ground_truth_file)
    total = len(df)

    # Create output directories
    img_dir, data_dir = prepare_output_dirs(output_dir)

    # Initialize lists for storing results
    questions = df["question"].tolist()
    true_categories = df["label"].tolist()  # Assuming "label" contains text labels
    pred_categories = []
    pred_nums = []

    print(f"Classifying {total} questions...")
    for question in questions:
        pred_num = classify_question(question)
        pred_label = LABEL_MAP[pred_num]
        pred_categories.append(pred_label)
        pred_nums.append(pred_num)

    # Derive the type labels
    true_types = [category_to_type[cat] for cat in true_categories]
    pred_types = [CATEGORY_TYPE[num] for num in pred_nums]

    # Derive the merged labels
    merged_true = [merge_labels(cat) for cat in true_categories]
    merged_pred = [merge_labels(cat) for cat in pred_categories]

    # Save processed data for future reference
    predictions_df = pd.DataFrame(
        {
            "question": questions,
            "true_category": true_categories,
            "pred_category": pred_categories,
            "true_type": true_types,
            "pred_type": pred_types,
            "merged_true": merged_true,
            "merged_pred": merged_pred,
        }
    )
    predictions_df.to_csv(os.path.join(data_dir, "predictions.csv"), index=False)

    # 1. Basic Accuracy Metrics
    category_accuracy = accuracy_score(true_categories, pred_categories)
    merged_accuracy = accuracy_score(merged_true, merged_pred)
    type_accuracy = accuracy_score(true_types, pred_types)

    print(f"Computing detailed metrics and generating visualizations...")

    # 2. Detailed Classification Reports
    cat_report = classification_report(
        true_categories, pred_categories, output_dict=True
    )
    merged_report = classification_report(merged_true, merged_pred, output_dict=True)
    type_report = classification_report(true_types, pred_types, output_dict=True)

    # Save reports as CSV
    pd.DataFrame(cat_report).transpose().to_csv(
        os.path.join(data_dir, "category_report.csv")
    )
    pd.DataFrame(merged_report).transpose().to_csv(
        os.path.join(data_dir, "merged_report.csv")
    )
    pd.DataFrame(type_report).transpose().to_csv(
        os.path.join(data_dir, "type_report.csv")
    )

    # 3. Extract metrics for visualizations
    cat_metrics = extract_metrics_from_report(cat_report)
    merged_metrics = extract_metrics_from_report(merged_report)
    type_metrics = extract_metrics_from_report(type_report)

    # 4. Misclassified Examples
    cat_misclassified = process_misclassifications(
        questions, true_categories, pred_categories, list(set(true_categories))
    )
    merged_misclassified = process_misclassifications(
        questions, merged_true, merged_pred, list(set(merged_true))
    )
    type_misclassified = process_misclassifications(
        questions, true_types, pred_types, ["instrumental", "executive", "other"]
    )

    # 5. Create Visualizations
    print("Creating visualizations...")

    # Unique labels for each evaluation mode
    cat_labels = sorted(list(set(true_categories + pred_categories)))
    merged_labels = sorted(list(set(merged_true + merged_pred)))
    type_labels = ["instrumental", "executive", "other"]

    # Confusion matrices
    cat_cm_path = create_confusion_matrix_plot(
        true_categories,
        pred_categories,
        cat_labels,
        "Category Confusion Matrix",
        figsize=(12, 10),
        file_path=os.path.join(img_dir, "category_confusion_matrix.png"),
    )

    merged_cm_path = create_confusion_matrix_plot(
        merged_true,
        merged_pred,
        merged_labels,
        "Merged Categories Confusion Matrix",
        figsize=(10, 8),
        file_path=os.path.join(img_dir, "merged_confusion_matrix.png"),
    )

    type_cm_path = create_confusion_matrix_plot(
        true_types,
        pred_types,
        type_labels,
        "Type Confusion Matrix",
        figsize=(8, 6),
        file_path=os.path.join(img_dir, "type_confusion_matrix.png"),
    )

    # Metrics bar charts
    cat_metrics_path = create_metrics_bar_chart(
        cat_metrics,
        "Category Classification Metrics",
        file_path=os.path.join(img_dir, "category_metrics.png"),
    )

    merged_metrics_path = create_metrics_bar_chart(
        merged_metrics,
        "Merged Categories Classification Metrics",
        file_path=os.path.join(img_dir, "merged_metrics.png"),
    )

    type_metrics_path = create_metrics_bar_chart(
        type_metrics,
        "Type Classification Metrics",
        file_path=os.path.join(img_dir, "type_metrics.png"),
    )

    # Distribution plots
    cat_dist_path = create_distribution_plot(
        true_categories,
        pred_categories,
        "Category Distribution: True vs Predicted",
        file_path=os.path.join(img_dir, "category_distribution.png"),
    )

    merged_dist_path = create_distribution_plot(
        merged_true,
        merged_pred,
        "Merged Categories Distribution: True vs Predicted",
        file_path=os.path.join(img_dir, "merged_distribution.png"),
    )

    type_dist_path = create_distribution_plot(
        true_types,
        pred_types,
        "Type Distribution: True vs Predicted",
        file_path=os.path.join(img_dir, "type_distribution.png"),
    )

    # 6. Generate HTML report
    print("Generating HTML report...")

    html_report = generate_html_report(
        total=total,
        category_accuracy=category_accuracy,
        merged_accuracy=merged_accuracy,
        type_accuracy=type_accuracy,
        cat_report=cat_report,
        merged_report=merged_report,
        type_report=type_report,
        cat_misclassified=cat_misclassified,
        merged_misclassified=merged_misclassified,
        type_misclassified=type_misclassified,
        img_paths={
            "category_cm": os.path.join("images", "category_confusion_matrix.png"),
            "merged_cm": os.path.join("images", "merged_confusion_matrix.png"),
            "type_cm": os.path.join("images", "type_confusion_matrix.png"),
            "category_metrics": os.path.join("images", "category_metrics.png"),
            "merged_metrics": os.path.join("images", "merged_metrics.png"),
            "type_metrics": os.path.join("images", "type_metrics.png"),
            "category_dist": os.path.join("images", "category_distribution.png"),
            "merged_dist": os.path.join("images", "merged_distribution.png"),
            "type_dist": os.path.join("images", "type_distribution.png"),
        },
    )

    with open(
        os.path.join(output_dir, "evaluation_report.html"), "w", encoding="utf-8"
    ) as f:
        f.write(html_report)

    # Also save a simple text summary
    with open(
        os.path.join(output_dir, "evaluation_summary.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("=== Classification Evaluation Summary ===\n\n")
        f.write(f"Total examples: {total}\n\n")
        f.write(f"Category Accuracy: {category_accuracy:.2%}\n")
        f.write(f"Merged Accuracy: {merged_accuracy:.2%}\n")
        f.write(f"Type Accuracy: {type_accuracy:.2%}\n\n")
        f.write("See the HTML report for detailed analysis and visualizations.\n")

    print(f"Evaluation complete. Report saved to {output_dir}/evaluation_report.html")
    print(f"Category Accuracy: {category_accuracy:.2%}")
    print(f"Merged Accuracy: {merged_accuracy:.2%}")
    print(f"Type Accuracy: {type_accuracy:.2%}")


def generate_html_report(
    total,
    category_accuracy,
    merged_accuracy,
    type_accuracy,
    cat_report,
    merged_report,
    type_report,
    cat_misclassified,
    merged_misclassified,
    type_misclassified,
    img_paths,
):
    """Generate HTML report with all evaluation information"""

    # Format the classification reports for HTML
    def format_metrics_table(report_dict):
        html = """
        <table class="table table-striped">
          <thead>
            <tr>
              <th>Label</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-score</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
        """

        # Add rows for each class
        for label, metrics in report_dict.items():
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                html += f"""
                <tr>
                  <td>{label}</td>
                  <td>{metrics["precision"]:.4f}</td>
                  <td>{metrics["recall"]:.4f}</td>
                  <td>{metrics["f1-score"]:.4f}</td>
                  <td>{metrics["support"]}</td>
                </tr>
                """

        # Add summary metrics
        for avg_type in ["macro avg", "weighted avg"]:
            if avg_type in report_dict:
                html += f"""
                <tr class="table-info">
                  <td>{avg_type}</td>
                  <td>{report_dict[avg_type]["precision"]:.4f}</td>
                  <td>{report_dict[avg_type]["recall"]:.4f}</td>
                  <td>{report_dict[avg_type]["f1-score"]:.4f}</td>
                  <td>{report_dict[avg_type]["support"]}</td>
                </tr>
                """

        # Add accuracy
        if "accuracy" in report_dict:
            html += f"""
            <tr class="table-success">
              <td>accuracy</td>
              <td colspan="3">{report_dict["accuracy"]:.4f}</td>
              <td>{report_dict["macro avg"]["support"]}</td>
            </tr>
            """

        html += """
          </tbody>
        </table>
        """
        return html

    # Format misclassified examples
    def format_misclassified(misclassified_list):
        if not misclassified_list:
            return "<p>No misclassified examples.</p>"

        html = """
        <div class="accordion" id="misclassifiedAccordion">
        """

        for i, item in enumerate(misclassified_list):
            html += f"""
            <div class="accordion-item">
              <h2 class="accordion-header" id="heading{i}">
                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
                        data-bs-target="#collapse{i}" aria-expanded="false" aria-controls="collapse{i}">
                  {item["true_label"]} → {item["predicted_label"]}
                </button>
              </h2>
              <div id="collapse{i}" class="accordion-collapse collapse" aria-labelledby="heading{i}">
                <div class="accordion-body">
                  <strong>Question:</strong> {item["question"]}<br>
                  <strong>True Label:</strong> {item["true_label"]}<br>
                  <strong>Predicted Label:</strong> {item["predicted_label"]}
                </div>
              </div>
            </div>
            """

        html += """
        </div>
        """
        return html

    # Build the full HTML report
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Dutch Question Classifier Evaluation</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
      <style>
        body {{ padding: 20px; }}
        .metric-card {{ margin-bottom: 20px; }}
        img {{ max-width: 100%; height: auto; }}
        .tab-content {{ padding: 20px 0; }}
        .nav-tabs {{ margin-bottom: 20px; }}
      </style>
    </head>
    <body>
      <div class="container">
        <h1 class="mb-4">Dutch Question Classifier Evaluation Report</h1>
        
        <div class="row mb-4">
          <div class="col-md-4">
            <div class="card text-white bg-primary metric-card">
              <div class="card-header">Category Accuracy</div>
              <div class="card-body">
                <h5 class="card-title">{category_accuracy:.2%}</h5>
                <p class="card-text">Classification accuracy across all 10 original categories</p>
              </div>
            </div>
          </div>
          <div class="col-md-4">
            <div class="card text-white bg-success metric-card">
              <div class="card-header">Merged Accuracy</div>
              <div class="card-body">
                <h5 class="card-title">{merged_accuracy:.2%}</h5>
                <p class="card-text">Accuracy with Code/Concept Comprehension unified</p>
              </div>
            </div>
          </div>
          <div class="col-md-4">
            <div class="card text-white bg-info metric-card">
              <div class="card-header">Type Accuracy</div>
              <div class="card-body">
                <h5 class="card-title">{type_accuracy:.2%}</h5>
                <p class="card-text">Accuracy when grouping into instrumental, executive, other</p>
              </div>
            </div>
          </div>
        </div>
        
        <ul class="nav nav-tabs" id="myTab" role="tablist">
          <li class="nav-item" role="presentation">
            <button class="nav-link active" id="category-tab" data-bs-toggle="tab" 
                   data-bs-target="#category" type="button" role="tab" aria-selected="true">
              Categories
            </button>
          </li>
          <li class="nav-item" role="presentation">
            <button class="nav-link" id="merged-tab" data-bs-toggle="tab" 
                   data-bs-target="#merged" type="button" role="tab" aria-selected="false">
              Merged Categories
            </button>
          </li>
          <li class="nav-item" role="presentation">
            <button class="nav-link" id="type-tab" data-bs-toggle="tab" 
                   data-bs-target="#type" type="button" role="tab" aria-selected="false">
              Types
            </button>
          </li>
        </ul>
        
        <div class="tab-content" id="myTabContent">
          <!-- Category Tab -->
          <div class="tab-pane fade show active" id="category" role="tabpanel">
            <h2>Category Evaluation</h2>
            
            <h3>Confusion Matrix</h3>
            <div class="mb-4">
              <img src="{img_paths['category_cm']}" alt="Category Confusion Matrix" class="img-fluid">
            </div>
            
            <div class="row">
              <div class="col-md-6">
                <h3>Category Distribution</h3>
                <img src="{img_paths['category_dist']}" alt="Category Distribution" class="img-fluid">
              </div>
              <div class="col-md-6">
                <h3>Classification Metrics</h3>
                <img src="{img_paths['category_metrics']}" alt="Category Metrics" class="img-fluid">
              </div>
            </div>
            
            <h3>Detailed Metrics</h3>
            {format_metrics_table(cat_report)}
            
            <h3>Misclassified Examples ({len(cat_misclassified)} of {total})</h3>
            {format_misclassified(cat_misclassified)}
          </div>
          
          <!-- Merged Categories Tab -->
          <div class="tab-pane fade" id="merged" role="tabpanel">
            <h2>Merged Categories Evaluation</h2>
            
            <h3>Confusion Matrix</h3>
            <div class="mb-4">
              <img src="{img_paths['merged_cm']}" alt="Merged Categories Confusion Matrix" class="img-fluid">
            </div>
            
            <div class="row">
              <div class="col-md-6">
                <h3>Category Distribution</h3>
                <img src="{img_paths['merged_dist']}" alt="Merged Categories Distribution" class="img-fluid">
              </div>
              <div class="col-md-6">
                <h3>Classification Metrics</h3>
                <img src="{img_paths['merged_metrics']}" alt="Merged Categories Metrics" class="img-fluid">
              </div>
            </div>
            
            <h3>Detailed Metrics</h3>
            {format_metrics_table(merged_report)}
            
            <h3>Misclassified Examples ({len(merged_misclassified)} of {total})</h3>
            {format_misclassified(merged_misclassified)}
          </div>
          
          <!-- Types Tab -->
          <div class="tab-pane fade" id="type" role="tabpanel">
            <h2>Types Evaluation</h2>
            
            <h3>Confusion Matrix</h3>
            <div class="mb-4">
              <img src="{img_paths['type_cm']}" alt="Type Confusion Matrix" class="img-fluid">
            </div>
            
            <div class="row">
              <div class="col-md-6">
                <h3>Type Distribution</h3>
                <img src="{img_paths['type_dist']}" alt="Type Distribution" class="img-fluid">
              </div>
              <div class="col-md-6">
                <h3>Classification Metrics</h3>
                <img src="{img_paths['type_metrics']}" alt="Type Metrics" class="img-fluid">
              </div>
            </div>
            
            <h3>Detailed Metrics</h3>
            {format_metrics_table(type_report)}
            
            <h3>Misclassified Examples ({len(type_misclassified)} of {total})</h3>
            {format_misclassified(type_misclassified)}
          </div>
        </div>
        
        <footer class="mt-5 pt-5 text-muted border-top">
          &copy; {pd.Timestamp.now().year} JELAI Classifier Evaluation
        </footer>
      </div>
    </body>
    </html>
    """

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of the classifier"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to ground truth Excel file"
    )
    parser.add_argument(
        "--output", "-o", default="evaluation_report", help="Output directory path"
    )

    args = parser.parse_args()
    evaluate_classifier(args.input, args.output)


if __name__ == "__main__":
    main()
