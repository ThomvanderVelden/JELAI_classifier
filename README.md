# JELAI Classifier

This repository contains the source code and resources for part of my Master's Thesis project on multi-class classification of Dutch questions. The classifier is built using Transformers (RobBERT) and PyTorch, and it categorizes questions into various categories with a custom training routine to handle imbalanced data.

## Overview
The JELAI Classifier:
- Reads and preprocesses questions from CSV files.
- Splits data into training and validation sets.
- Computes class weights to mitigate label imbalance.
- Trains a multi-class model with a custom Trainer using weighted loss.
- Supports inference on new data and file-based batch classification.
- Evaluates model performance and generates an evaluation report.

## Repository Contents
- **multi_classifier.py**: Main script for training, evaluating, and inference.
- **evaluate.py**: Script for running evaluation against ground truth data.
- **questions_before_midterm.csv**: Training data.
- **questions_after_midterm.csv**: Data for inference.
- **ground_truth.xlsx**: Labels for evaluation.
- **predictions.txt**: Output file for model predictions.
- **.gitignore**: Ignored files and directories (e.g., model checkpoints, results).

## Setup & Requirements
Ensure you have Python ≥3.8 and install the necessary libraries:
```bash
pip install -r requirements.txt
```

## Usage

### Training & Inference
1. To **train** a new model, set the `DO_TRAIN` variable in `multi_classifier.py` to `True` and run:
   ```bash
   python multi_classifier.py
   ```
2. To **load** an existing model for inference, set `DO_TRAIN` to `False` and run the same script.

### Evaluation
To evaluate the classifier and generate a report, run:
```bash
python evaluate.py
```
The evaluation report will be saved as `evaluation_report.txt`.

## Thesis Context
This classifier is developed as part of my Master's Thesis, classifying student questions that have been asked to our LLM. The open-source repository reflects the research and development process and is shared to foster further study and collaboration.

## Acknowledgements
Special thanks to the developers of the Transformers library and the open-source community for their valuable contributions.
