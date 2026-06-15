# Session 10 — Cross-Validation, Logistic Regression & Model Evaluation

## 🎯 What I Learned

### Cross-Validation Methods
- **Holdout Validation (Train-Test Split)** — Simple 70-30 or 80-20 split, can use stratification for imbalanced datasets
- **LOOCV (Leave-One-Out Cross-Validation)** — Train on n-1 samples, test on 1 sample, repeat n times. Expensive but uses all data
- **K-Fold Cross-Validation** — Split data into K equal folds, each fold gets turns as test set. K=10 is standard
- **Repeated K-Fold** — Repeat K-Fold multiple times with different shuffles for more reliable estimates
- **Leave-P-Out** — Test on P samples, repeat C(n,P) times. Grows factorially — rarely used in practice

### Logistic Regression
- Used for **binary classification** (Yes/No, True/False, 0/1)
- **Types**: Binomial (binary), Multinomial (multi-class), Ordinal (ordered categories)
- **Decision boundary**: `model.predict_proba(X) > threshold → class 1, else class 0`
- **Sigmoid function** squashes outputs to probabilities between 0 and 1

### Learning Curves
- **Purpose**: Diagnose bias vs variance problems
- **Manual implementation** — Train on increasing data sizes, plot accuracy vs training size
- **Sklearn `learning_curve`** — Built-in function with cross-validation
- **Interpretation**:
  - High training accuracy, low test accuracy → Overfitting (high variance)
  - Both low accuracy → Underfitting (high bias)

### Evaluation Metrics
- **Precision** — Of predicted positives, how many were actually positive
- **Recall** — Of actual positives, how many were correctly predicted
- **F1-Score** — Harmonic mean of precision and recall
- **Accuracy** — Overall correct predictions / total predictions
- **Confusion Matrix** — Shows TP, TN, FP, FN

## 📁 Files
- `Session_10.ipynb` — Cross-validation methods, logistic regression, learning curves, MNIST classification
- `mnist_784.csv` — MNIST handwritten digits dataset (70,000 images × 785 columns: 784 pixels + class label)

## 🔑 Key Concepts Covered

| Concept | Method/Tool | Description |
|---------|------------|-------------|
| K-Fold CV | `KFold(n_splits=3)` | Splits data into 3 equal folds |
| Stratified K-Fold | `StratifiedKFold(n_splits=3)` | Maintains class distribution across folds |
| LOOCV | `LeaveOneOut()` | Leaves 1 sample out each iteration (n iterations) |
| Leave-P-Out | `LeavePOut(p=3)` | Leaves P samples out (C(n,P) iterations) |
| Cross-Validation | `cross_val_score()` | Returns array of scores, one per fold |
| Detailed CV | `cross_validate()` | Returns train scores, test scores, fit times |
| Logistic Regression | `LogisticRegression()` | Linear model for classification with sigmoid |
| Learning Curve | `learning_curve()` | Shows performance vs training set size |
| Confusion Matrix | `confusion_matrix()` | Table of true vs predicted classes |
| Classification Report | `classification_report()` | Precision, recall, f1-score per class |
| Digits Dataset | `load_digits()` | 8×8 pixel handwritten digits (1797 samples) |

## 📊 Results Summary

### K-Fold Cross-Validation (3 folds on 12 samples)

| Fold | Train Indices | Test Indices |
|------|---------------|--------------|
| 1 | [4,5,6,7,8,9,10,11] | [0,1,2,3] |
| 2 | [0,1,2,3,8,9,10,11] | [4,5,6,7] |
| 3 | [0,1,2,3,4,5,6,7] | [8,9,10,11] |

### Leave-One-Out CV (12 iterations)
- Each iteration: Train on 11 samples, test on 1
- Total: 12 training runs

### Leave-P-Out (p=3, n=12)
- Number of iterations: C(12,3) = 220 iterations

Notice each fold has balanced class distribution (50% class 0, 50% class 1)

### Cross-Validation Scores (Random Forest on 12 samples)

| Metric | Score 1 | Score 2 | Score 3 | Mean |
|--------|---------|---------|---------|------|
| Accuracy | 0.667 | 1.0 | 0.8 | 0.822 |
| F1 | 0.667 | 1.0 | 0.8 | 0.822 |

### Logistic Regression on Digits Dataset (1797 samples, 64 features)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 1.00 | 1.00 | 1.00 | 36 |
| 1 | 0.88 | 0.97 | 0.92 | 36 |
| 2 | 0.97 | 0.97 | 0.97 | 35 |
| 3 | 0.97 | 1.00 | 0.99 | 37 |
| 4 | 1.00 | 0.97 | 0.99 | 36 |
| 5 | 0.92 | 0.89 | 0.90 | 37 |
| 6 | 0.97 | 1.00 | 0.99 | 36 |
| 7 | 0.97 | 0.94 | 0.96 | 36 |
| 8 | 0.94 | 0.86 | 0.90 | 35 |
| 9 | 0.97 | 0.97 | 0.97 | 36 |

**Overall Accuracy: 96%**

### Logistic Regression on MNIST (70,000 samples, 784 features)
- **Accuracy**: 91%
- **Note**: Convergence warning due to large dataset — increased `max_iter=300`

### Learning Curve Results (Digits Dataset)

| Training Size | Samples | Accuracy |
|---------------|---------|----------|
| 1% | 17 | 0.766 |
| 2% | 35 | 0.825 |
| 3% | 53 | 0.860 |
| 4% | 71 | 0.867 |
| 5% | 89 | 0.903 |
| 10% | 179 | 0.930 |
| 20% | 359 | 0.947 |

**Observations**: Accuracy improves rapidly up to 10% of data, then plateaus

### Learning Curve Parameters (sklearn)

learning_curve(
model, X, y,
train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5],
cv=5,
scoring='accuracy',
verbose=1,
n_jobs=-1
)


### cross_val_score vs cross_validate

| Function | Returns | Use Case |
|----------|---------|----------|
| `cross_val_score` | Array of scores | Quick performance estimate |
| `cross_validate` | Dict with test_score, train_score, fit_time, score_time | Detailed analysis, training performance |

## 🧠 Key Takeaways

### Cross-Validation Rules
- **Always use StratifiedKFold for classification** — preserves class distribution
- **Never shuffle time series** — preserves temporal order
- **K-Fold with K=10** is the industry standard
- **LOOCV** is deterministic but computationally expensive (n trainings)
- **Leave-P-Out** is rarely practical — factorial growth is prohibitive

### Logistic Regression Essentials
- Linear decision boundary in feature space
- Outputs probabilities via sigmoid function
- **L2 regularization (Ridge)** is default (prevents overfitting)
- Set `max_iter` higher for large datasets (default=100 often insufficient)
- **Multi-class** uses one-vs-rest (OvR) or multinomial

### Learning Curve Interpretation
- **Training accuracy high, test low** → Overfitting (high variance)
- **Both training and test low** → Underfitting (high bias)
- **Both high and close** → Good fit
- **Gap between curves** → Variance issue
- **Steep improvement slope** → Need more data

### Bias-Variance Tradeoff
- **High Bias (Underfitting)**: Model too simple, misses patterns
- **High Variance (Overfitting)**: Model too complex, memorizes noise
- **Goal**: Find sweet spot between bias and variance

### Cross-Validation Best Practices
1. Use StratifiedKFold for classification (maintains class ratios)
2. Set `shuffle=True` with `random_state` for reproducibility
3. Use `n_jobs=-1` to parallelize across CPU cores
4. Save cross-validation results to compare models
5. For small datasets, consider LOOCV or increased folds
6. For large datasets, use fewer folds (3-5) to save time

