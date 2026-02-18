# DL Competition 01 — Image Classification (PyTorch)

## Goal

Train an image classification model using PyTorch on the provided labeled datasets, then generate predictions for a blind/unlabeled test split (`comp_test`) and submit them as a CSV file.

You will work with four datasets:

* train (labeled): used to fit the model
* val (labeled): used for model selection / hyperparameter tuning
* test (labeled): used only as an internal benchmark (do not tune on it repeatedly)
* comp_test (unlabeled): blind set used for the final competition ranking

The final ranking is computed by comparing your comp_test predictions vs. hidden ground-truth labels (classification accuracy).

---

## Dataset

This competition uses an Intel-style scene classification dataset with 6 classes.

Typical class names (verify with dataset.class_to_idx in your notebook):

* buildings
* forest
* glacier
* mountain
* sea
* street

### Kaggle paths (reference)

If you are running on Kaggle, the notebook assumes:

Train: /kaggle/input/intel-image-classification/seg_train/seg_train
Val/Test source: /kaggle/input/intel-image-classification/seg_test/seg_test
Blind competition test: /kaggle/input/intel-image-classification/seg_pred/seg_pred

If you run locally, adapt paths accordingly. The expected folder structure is the standard ImageFolder format: one subfolder per class for labeled splits.

---

## Preprocessing (baseline)

A reference preprocessing pipeline is:

* Convert to grayscale (1 channel)
* Resize to 150×150
* Convert to tensor
* Normalize with mean=0.5, std=0.5 (single channel)

You may modify preprocessing, but keep it consistent between training and inference.

---

## What you must do

1. Implement and train a PyTorch model for 6-class image classification.
2. Use val to tune your configuration (architecture, optimizer, LR, regularization, augmentation, etc.).
3. (Optional but recommended) report your final test performance once, as a sanity check.
4. Run inference on comp_test and export a submission CSV.
5. Submit your CSV via the Hugging Face competition Space (required).

---

## Submission format (CSV)

Your file must be named predictions.csv and contain exactly two columns:

id,pred

Example:

id,pred
10004.jpg,2
10010.jpg,5

---

# Deliverables (Required)

You must submit three separate items for grading.

## 1) Notebook (.ipynb)

A single notebook containing:

* Data loading and preprocessing
* Model definition (PyTorch)
* Training loop (loss, optimizer, backpropagation)
* Validation logic
* Final inference on comp_test
* Generation of predictions.csv

Notebook requirements:

* Runs end-to-end without manual edits
* Clearly shows the final model used for submission
* Saves the CSV file in the correct format

---

## 2) Network Configuration Document (.md or .pdf)

A short document describing your final model configuration.

It must include:

* Architecture description (layers, activations, output layer)
* Input size and preprocessing
* Loss function
* Optimizer and hyperparameters:

  * Learning rate
  * Batch size
  * Number of epochs
  * Weight decay (if used)
  * Scheduler (if used)
* Regularization methods (dropout, augmentation, early stopping, etc.)
* Model selection strategy (how you chose the final model)

The explanation must be clear enough for someone else to reproduce your results.

---

## 3) Declaration of AI Use (.md or .pdf)

A short written statement answering:

* Did you use AI tools (ChatGPT, Copilot, etc.)? (Yes/No)
* For what tasks? (debugging, explanation, code generation, etc.)
* Which parts of the work are fully yours?
* Any external resources used (include links if applicable)

Transparency is required. AI assistance is allowed, but it must be declared.

---

# Base Notebook (Kaggle) — Copy, Edit, and Run

Starter notebook:
https://www.kaggle.com/code/juanmartinezv4399/dl-competition01

### Steps in Kaggle

1. Open the starter notebook.
2. Click Copy & Edit (or Fork Notebook).
3. Rename using: DL-Competition01_LastName_FirstName
4. Ensure dataset is attached via Add Data.
5. Complete sections marked TODO:

   * Model definition
   * Training loop / hyperparameters
   * Validation
   * Inference on comp_test
   * CSV export

---

## Required outputs from the notebook

Your notebook must produce:

* Training/validation logs
* predictions.csv with columns id,pred

---

## Exporting files from Kaggle

1. Ensure predictions.csv is saved under /kaggle/working/
2. Download from the Output panel:

   * Final notebook (.ipynb)
   * predictions.csv

Do not submit a notebook that only partially runs.

---

# Competition Submission via Hugging Face (Required)

Submission portal:
https://huggingface.co/spaces/MLEAFIT/DLComp0120261

---

## What to submit on Hugging Face

Upload only:

predictions.csv

Requirements:

* Exactly two columns: id,pred
* Filenames must match comp_test exactly
* Predictions must be integers from 0 to 5

---

## Submission steps

1. Log in to your Hugging Face account.
2. Open the Space link.
3. Upload predictions.csv.
4. Confirm submission is accepted.

---

## Evaluation

Metric: Accuracy
Ranking is computed using hidden ground-truth labels.

---

# Final Checklist

* Notebook runs end-to-end
* CSV generated automatically
* CSV format correct
* CSV uploaded to Hugging Face
* Notebook + config doc + AI declaration submitted

---

