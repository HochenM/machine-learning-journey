# 📘 Session 11 – Support Vector Machines (SVM) & Hyperparameter Tuning

## 🚀 Overview
This repository contains my implementation and learning of **Support Vector Machines (SVM)** and **hyperparameter tuning techniques** such as **GridSearchCV** and **RandomizedSearchCV**.

The session includes theoretical understanding, model comparison, feature engineering, and a real-world Bitcoin price prediction case study.

---

## 📌 Topics Covered

### 🔹 Support Vector Machines (SVM)
- Hyperplane and decision boundary
- Support vectors and margin maximization
- Hard margin vs soft margin
- Kernel trick (Linear, Polynomial, RBF)
- Regularization parameter (C)
- Gamma parameter behavior
- NuSVC concept

---

### 🔹 Model Comparison
- Logistic Regression vs SVM
- Linear SVM vs Kernel SVM
- Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix

---

### 🔹 Hyperparameter Tuning
- GridSearchCV (exhaustive search)
- RandomizedSearchCV (random search)
- Cross-validation using StratifiedKFold
- Best parameter selection and model refitting

---

### 🔹 Time Series Case Study (BTC Prediction)
- Bitcoin price direction prediction
- Feature engineering:
  - Returns
  - Simple Moving Averages (SMA 7, SMA 30)
  - Time-based features (Year, Month, Day, Weekday)
- Train/test split without shuffling (time-series safe)

---

## 🧠 Key Learnings

- SVM is highly sensitive to feature scaling
- RBF kernel handles non-linear patterns effectively
- Time-series data must not be randomly shuffled
- Raw price values are not informative → use relative/ratio features
- Proper feature engineering significantly improves performance
- Pipelines prevent data leakage and improve workflow safety

---

## 📊 Models Used

- Logistic Regression
- Linear SVM (LinearSVC)
- Kernel SVM (SVC with linear, RBF, poly, sigmoid kernels)
- GridSearchCV optimized SVM model
- RandomizedSearchCV (Logistic Regression tuning)

---

## ⚙️ Tech Stack

- Python 🐍
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- yfinance

---

## 📈 Results Summary
SVM (RBF kernel) performed better than Logistic Regression in most cases
Hyperparameter tuning improved model performance significantly
Feature scaling and engineered ratios improved BTC prediction stability

## ⚠️ Important Insights
❌ Raw price features lead to poor generalization
❌ SVM without scaling performs poorly
❌ Random split is invalid for time-series data
✅ Stationary features improve predictive power
✅ Pipelines ensure clean and safe ML workflow

## 🔥 Future Improvements
Add LSTM / deep learning comparison
Include advanced indicators (RSI, MACD)
Use walk-forward validation for time-series
Deploy model using Flask or FastAPI

