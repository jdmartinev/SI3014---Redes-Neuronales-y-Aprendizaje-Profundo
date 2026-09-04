# SI3014 — Tiny ImageNet Classification Competition

This repository contains the materials used for the **SI3014 Tiny ImageNet Classification Competition**.

The activity focuses on image classification with neural networks and is designed to compare a fully connected neural network (FCNN) with a convolutional neural network (CNN) using the Tiny ImageNet dataset.

## Kaggle Competition

Join the competition here:

**[SI3014 — Tiny ImageNet Classification Competition](https://www.kaggle.com/t/ad937bc76fe34f78ba4c94771c52604d)**

Participants must join the competition before accessing the competition data from a Kaggle notebook.

## Starter Notebook

A starter notebook is available here:

**[SI3014 Tiny ImageNet Competition — Starter Notebook](https://www.kaggle.com/code/juanmartinezv4399/si3014-tinyimagenet-competition)**

Recommended workflow:

1. Join the Kaggle competition.
2. Open the starter notebook.
3. Click **Copy & Edit**.
4. Add the competition as an input using **Add Input → Competitions**.
5. Run the setup cells and complete the activity.

## Competition Task

Students must build and compare:

1. **FCNN / MLP** — without convolutional layers.
2. **CNN** — using convolutional layers.

After the comparison, the CNN can be improved to obtain the best possible validation and leaderboard performance.

The competition contains **200 classes** and uses **classification accuracy** as the evaluation metric.

## Repository Contents

This repository may include:

```text
.
├── README.md
├── notebooks/
│   └── starter notebook
├── data/
│   ├── dataset documentation
│   └── CSV metadata
└── competition/
    ├── evaluation information
    └── supporting materials
```

The official competition data and hidden test labels are managed through Kaggle.

## Submission Format

Predictions must be submitted as a CSV file with the following structure:

```csv
id,pred
val_9240.JPEG,55
val_9308.JPEG,74
val_3262.JPEG,40
```

where `pred` is an integer class label from `0` to `199`.

## Prize

🏆 The top performers on the final leaderboard will receive **exclusive SI3014 competition pins**.
