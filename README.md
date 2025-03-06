# JELAI Classifier

This repository contains the source code and resources for part of my Master's Thesis project. The classifier is built using Transformers (RobBERT) and PyTorch, and it categorizes questions into various categories with a custom training routine.

## Overview
The JELAI Classifier:
- Reads and preprocesses questions from CSV files.
- Splits data into training and validation sets.
- Computes class weights to mitigate label imbalance.
- Trains a multi-class model with a custom Trainer using weighted loss.
- Supports inference on new data and file-based batch classification.
- Evaluates model performance and generates a comprehensive evaluation report.

## Repository Contents
- **classifier.py**: Main script for training, evaluating, and inference.
- **evaluate.py**: Script for running evaluation against ground truth data.
- **data folder**: Folder containing all the training and evaluation data used for this classifier
- **evaluation_results folder**: Folder containing all the evaluation results after inference on our model
- **utility folder**: Folder containing the hyperparameter tuning and token_length scripts

## Setup & Requirements
Ensure you have Python ≥3.8 and install the necessary libraries:
```bash
pip install -r requirements.txt
```

## Usage

### Training & Inference
1. To **train** a new model, set the `DO_TRAIN` variable in `classifier.py` to `True` and run:
   ```bash
   python classifier.py
   ```
2. To **load** an existing model for inference, set `DO_TRAIN` to `False` and run the same script.

### Evaluation
To evaluate the classifier and generate a report, run:

```bash
# Run the evaluation with ground truth data
python evaluate.py --input data/ground_truth.xlsx --output evaluation_results
```

The script will:

1. Generate visualization plots:
   - Confusion matrices for categories, merged categories, and types
   - Distribution plots comparing true vs. predicted labels
   - Performance metrics charts (precision, recall, F1)

2. Calculate detailed metrics:
   - Accuracy, precision, recall, and F1-score
   - Support counts for each class
   - Weighted and macro averages

3. Provide error analysis:
   - Lists of misclassified examples
   - Grouped by category and type

4. Create an interactive HTML report:
   - Summary cards with overall metrics
   - Tabbed interface for exploring different evaluation dimensions
   - Interactive elements for exploring misclassified examples
   - Detailed tables with all metrics

## Thesis Context
This classifier is developed as part of my Master's Thesis, classifying student questions that have been asked to our LLM. The open-source repository reflects the research and development process and is shared to foster further study and collaboration.

## Acknowledgements
Special thanks to the developers of the Transformers library and the open-source community for their contributions.
