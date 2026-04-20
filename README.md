# Nature Scene Image Classification

A comparison of three machine learning models — **Naive Bayes**, **Decision Tree**, and **MLP Neural Network** — applied to image classification across three natural landscape categories: desert, forest, and snow.

**Course:** Artificial Intelligence — ENCS3340  
**Institution:** Birzeit University, Department of Electrical and Computer Engineering  

---

## Problem Overview

Given a labeled dataset of nature images, the goal is to train and evaluate three different classifiers and compare their performance using accuracy, precision, recall, F1-score, and confusion matrices.

---

## Dataset

The dataset contains **553 images** divided into three balanced classes:

| Class | Images | Description |
|-------|--------|-------------|
| Desert | 186 | Sand dunes, dry landscapes, rocky areas, bright sunlight |
| Forest | 182 | Trees, plants, greenery — day and night shots |
| Snow | 185 | Snowy mountains, glaciers, winter scenes |

All images are resized to **32×32 pixels**, converted to RGB, flattened into 1D arrays, and normalized to `[0, 1]`.

**Dataset (Google Drive):** https://drive.google.com/drive/folders/1FVUNX-jsDwtXf12ntU5V4Du7lK_hdTlz

> Before running, update `dataset_location` in the script to point to your local copy of the dataset.

---

## Models

### Naive Bayes (`GaussianNB`)
Probabilistic classifier based on Bayes' theorem, assuming pixel features are independent. Fast to train and used as a baseline. Gaussian NB assumes pixel values follow a normal distribution.

### Decision Tree (`DecisionTreeClassifier`)
Rule-based model that splits data on pixel values using entropy as the splitting criterion. Grown fully with no depth limit. The tree is also visualized to show how decisions are made at each node.

### MLP Neural Network (`MLPClassifier`)
Feedforward neural network with two hidden layers (128 → 64 neurons), ReLU activation, Adam optimizer, and early stopping. Trained for up to 1000 iterations.

---

## Results

### Overall Accuracy

| Model | Accuracy |
|-------|----------|
| Naive Bayes | 0.7986 |
| Decision Tree | 0.8201 |
| MLP Neural Network | **0.8633** |

### Per-Class Performance (best model — MLP)

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Desert | 0.89 | 0.89 | 0.89 |
| Forest | 0.83 | 0.76 | 0.80 |
| Snow | 0.86 | 0.93 | 0.90 |

The MLP outperformed both other models across all classes. The Decision Tree performed well but showed more confusion between snow and forest. Naive Bayes had the most errors, particularly on desert images, due to its independence assumption not holding for pixel data.

### Architecture Tuning (MLP)

Reducing the MLP to a single hidden layer (128 neurons) dropped accuracy from **0.8633 to 0.8201**, matching the Decision Tree. Forest recall fell significantly (0.76 → 0.65), showing that the deeper architecture captures more complex patterns in the data.

---

## Output

The script prints per-class precision, recall, and F1-score for each model, then displays:
- Confusion matrix heatmaps for all three models
- A full decision tree visualization

---

## Running the Project

1. Download the dataset from the link above
2. Update the dataset path in the script:
   ```python
   dataset_location = r"path/to/your/dataset"
   ```
3. Install dependencies:
   ```bash
   pip install numpy pillow scikit-learn matplotlib seaborn
   ```
4. Run:
   ```bash
   python code.py
   ```

---

## Files

| File | Description |
|------|-------------|
| `code.py` | Main source code |
| `report.pdf` | Project report |

---

## Requirements

- Python 3.x
- numpy, pillow, scikit-learn, matplotlib, seaborn
