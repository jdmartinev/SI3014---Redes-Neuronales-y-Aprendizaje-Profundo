# DL Competition 01 — Image Classification (PyTorch)

## Goal
Train an image classification model **using PyTorch** on the provided labeled datasets, then generate **predictions for a blind/unlabeled test split (`comp_test`)** and submit them as a CSV file.

You will work with four datasets:
- **train** (labeled): used to fit the model
- **val** (labeled): used for model selection / hyperparameter tuning
- **test** (labeled): used only as an internal benchmark (do not tune on it repeatedly)
- **comp_test** (unlabeled): **blind set** used for the final competition ranking

The final ranking is computed by comparing your `comp_test` predictions vs. hidden ground-truth labels (classification **accuracy**).

---

## Dataset
This competition uses an Intel-style scene classification dataset with **6 classes**.

Typical class names (verify with `dataset.class_to_idx` in your notebook):
- `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`

### Kaggle paths (reference)
If you are running on Kaggle, the notebook assumes:
- Train: `/kaggle/input/intel-image-classification/seg_train/seg_train`
- Val/Test source: `/kaggle/input/intel-image-classification/seg_test/seg_test`
- Blind competition test: `/kaggle/input/intel-image-classification/seg_pred/seg_pred`

> If you run locally, adapt paths accordingly. The expected folder structure is the standard `ImageFolder` format: one subfolder per class for labeled splits.

---

## Preprocessing (baseline)
A reference preprocessing pipeline is:
- Convert to **grayscale** (1 channel)
- Resize to **150×150**
- Convert to tensor
- Normalize with mean=0.5, std=0.5 (single channel)

You may modify preprocessing, but keep it consistent between training and inference.

---

## What you must do
1. **Implement and train a PyTorch model** for 6-class image classification.
2. Use **val** to tune your configuration (architecture, optimizer, LR, regularization, augmentation, etc.).
3. (Optional but recommended) report your final **test** performance once, as a sanity check.
4. Run inference on **comp_test** and export a submission CSV.

---

## Submission format (CSV)
You must submit a file named (or equivalent) `predictions.csv` with **exactly two columns**:

| column | type | description |
|---|---|---|
| `id` | string | image filename (exactly as read from `comp_test`) |
| `pred` | int | predicted class index (0–5) |

Example:
```csv
id,pred
10004.jpg,2
10010.jpg,5
```

## Deliverables

You must submit **three separate items**:

---

### 1) Notebook (Required)

A single `.ipynb` file containing:

- Data loading and preprocessing
- Model definition (PyTorch)
- Training loop (loss, optimizer, backpropagation)
- Validation logic
- Final inference on `comp_test`
- Generation of `predictions.csv`

The notebook must:
- Run end-to-end without manual edits
- Clearly show your final model used for submission
- Save the CSV file in the correct format

---

### 2) Network Configuration Document (Required)

A short document (`.md` or `.pdf`) describing your final model configuration.

It must include:

- Architecture description (layers, activations, output layer)
- Input size and preprocessing
- Loss function
- Optimizer and hyperparameters:
  - Learning rate
  - Batch size
  - Number of epochs
  - Weight decay (if used)
  - Scheduler (if used)
- Regularization methods (dropout, augmentation, early stopping, etc.)
- Model selection strategy (how you chose the final model)

The explanation must be clear enough for someone else to reproduce your results.

---

### 3) Declaration of AI Use (Required)

A short written statement answering:

- Did you use AI tools (ChatGPT, Copilot, etc.)? (Yes/No)
- For what tasks? (debugging, explanation, code generation, etc.)
- Which parts of the work are fully yours?
- Any external resources used (include links if applicable)

Transparency is required. AI assistance is allowed, but it must be declared.

## Base Notebook (Kaggle) — Copy, Edit, and Run

Use the official starter notebook as your base template:

- **Starter notebook (Kaggle):** https://www.kaggle.com/code/juanmartinezv4399/dl-competition01

You must **copy it in Kaggle**, adapt it to your approach, and submit **your edited version** as the final notebook deliverable.

### Steps in Kaggle

1. Open the starter notebook link above.
2. Click **Copy & Edit** (or **Fork Notebook**, depending on Kaggle UI).
3. Rename your notebook using the format:
   - `DL-Competition01_<LastName>_<FirstName>`
4. Verify dataset attachment:
   - In the right panel, go to **Add Data** and ensure the competition dataset is attached.
5. Edit the notebook sections marked as **TODO**:
   - **Model definition**
   - **Training loop / hyperparameters**
   - **Validation + model selection**
   - **Inference on `comp_test`**
   - **CSV export**

### Required outputs from the notebook

Your notebook must produce:
- Training/validation logs (loss + accuracy, or equivalent)
- A saved submission file:
  - `predictions.csv` with columns: `id,pred`

### Exporting files from Kaggle

Before submitting:
1. Ensure `predictions.csv` is created by the notebook (typically saved under `/kaggle/working/`).
2. In Kaggle, open the **Output** panel and download:
   - Your final notebook (`.ipynb`)
   - `predictions.csv`

> Do not submit a notebook that only partially runs. It must run end-to-end on Kaggle.
