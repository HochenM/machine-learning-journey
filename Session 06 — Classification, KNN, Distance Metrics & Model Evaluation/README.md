# Session 06 — Classification, KNN, Distance Metrics & Model Evaluation

## 🎯 What I Learned
- Built a **KNN classifier** to predict class labels for new data points
- Understood distance metrics: **Euclidean (L2)** and **Manhattan (L1)**
- Understood similarity metrics: **Cosine** and **Jaccard**
- Explored KNN hyperparameters: `n_neighbors`, `weights`, `algorithm`, `p`, `metric`, `n_jobs`
- Used `LocalOutlierFactor` to detect outliers in data
- Calculated and interpreted **accuracy, precision, recall, F1-score**
- Built and read a **confusion matrix**
- Used `classification_report` for all metrics in one line
- Understood the precision-recall tradeoff in medical diagnosis
- Applied `stratify=y` for balanced train-test splits in classification

## 📁 Files
- `main.py` — KNN classification, outlier detection, confusion matrix, metrics
- `outputs/` — Cluster plots, confusion matrix visualizations

## 🔑 Key Concepts Covered

| Concept | Method/Tool | Description |
|---------|------------|-------------|
| KNN Classifier | `KNeighborsClassifier(n_neighbors=5)` | Classifies by majority vote of K nearest neighbors |
| Distance Metrics | `metric='euclidean'`, `metric='manhattan'` | How "closeness" is measured |
| Similarity Metrics | Cosine, Jaccard | How "similar" two vectors/sets are |
| Local Outlier Factor | `LocalOutlierFactor(n_neighbors=5)` | Detects outliers (-1 = outlier, 1 = normal) |
| KNN Algorithms | `'auto'`, `'kd_tree'`, `'ball_tree'`, `'brute'` | Search method for finding neighbors |
| Accuracy | `(TP + TN) / Total` | Overall correctness |
| Precision | `TP / (TP + FP)` | When model says YES, how often is it right? |
| Recall (TPR) | `TP / (TP + FN)` | How many actual YES did model find? |
| F1 Score | `2 × P × R / (P + R)` | Harmonic mean of Precision and Recall |
| Confusion Matrix | `confusion_matrix(y_true, y_pred)` | Table of TP, TN, FP, FN |

## 📊 Results Summary

### Cancer Detection Example
| Metric | Value | Interpretation |
|--------|:---:|---------|
| Accuracy | 0.90 | 90% correct overall |
| Precision | 0.83 | 83% of "cancer" calls were correct |
| Recall | 1.00 | ALL cancers found — zero missed! |
| F1 | 0.91 | Excellent balance |

### Blobs Classification (K=3)
| Class | Precision | Recall | F1 |
|:---:|:---:|:---:|:---:|
| 0 | 1.00 | 0.50 | 0.67 |
| 1 | 0.67 | 1.00 | 0.80 |
| 2 | 1.00 | 1.00 | 1.00 |


## 🧠 Key Takeaways
KNN doesn't "learn" weights — it just memorizes data and compares distances

n_neighbors=1 → noisy; n_neighbors=50 → underfits; odd numbers avoid ties

weights='distance' gives closer neighbors more influence

algorithm='brute' checks ALL points (slow but exact); 'kd_tree' is faster for low dimensions

Cosine ignores magnitude — good for text; Euclidean uses raw values — good for measurements

LocalOutlierFactor uses fit_predict(), not predict() — it's a detector, not a classifier

stratify=y preserves class ratios in train/test splits

In cancer detection, Recall > Precision — better to false-alarm than miss a cancer

classification_report gives all metrics at once with per-class breakdown

F1 = 0.91 means excellent balance between finding cancers and avoiding false alarms




## 🔧 Key Methods Used

```python
# KNN
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Outlier Detection
lof = LocalOutlierFactor(n_neighbors=5)
outliers = lof.fit_predict(heights)  # -1 = outlier, 1 = normal

# Metrics
accuracy_score(y_true, y_pred)
precision_score(y_true, y_pred)
recall_score(y_true, y_pred)
f1_score(y_true, y_pred)
confusion_matrix(y_true, y_pred)
classification_report(y_true, y_pred)
