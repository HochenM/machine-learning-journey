📘 Session 11 – Support Vector Machines (SVM) & Hyperparameter Tuning
🚀 Overview

This repository contains my learning and implementation of Support Vector Machines (SVM) and hyperparameter tuning techniques including GridSearchCV and RandomizedSearchCV.

The session includes:

Theoretical understanding of SVM
Comparison with Logistic Regression
Kernel tricks (Linear, RBF)
Model evaluation using classification metrics
Hyperparameter optimization
Time-series based BTC prediction case study
Proper ML pipeline improvements (scaling + feature engineering)
📌 Topics Covered
🔹 Support Vector Machines (SVM)
Hyperplane, margin, and support vectors
Hard margin vs Soft margin
Kernel trick (Linear, Polynomial, RBF)
C and Gamma parameters
NuSVC concept
🔹 Model Comparison
Logistic Regression vs SVM
Linear SVM vs Kernel SVM
Performance evaluation using:
Accuracy
Precision / Recall / F1-score
Confusion Matrix
🔹 Hyperparameter Tuning
GridSearchCV (exhaustive search)
RandomizedSearchCV (random search)
Cross-validation (StratifiedKFold)
🔹 Real-world Case Study
Bitcoin price movement prediction
Feature engineering:
Returns
Moving averages (SMA 7 & 30)
Time features (Year, Month, Day)
Time-series train/test split (no shuffling)
🧠 Key Insights Learned
SVM performance heavily depends on feature scaling
RBF kernel is powerful for nonlinear data
Time-series data must avoid shuffling to prevent leakage
Absolute price values are not useful → stationary features perform better
Hyperparameter tuning significantly improves model performance
📊 Models Used
Logistic Regression
Linear SVM (LinearSVC)
Kernel SVM (SVC with RBF, poly, sigmoid)
GridSearchCV optimized SVM
RandomizedSearchCV Logistic Regression
⚙️ Tech Stack
Python 🐍
NumPy
Pandas
Matplotlib / Seaborn
Scikit-learn
yfinance (for BTC data)
📁 Project Structure
Session-11-SVM-Hyperparameter-Tuning/
│
├── BTC.csv
├── session11_svm.ipynb
├── README.md
▶️ How to Run
# Clone repository
git clone https://github.com/your-username/your-repo.git

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn yfinance

# Run notebook
jupyter notebook
📈 Results Summary
SVM (RBF kernel) outperformed Logistic Regression on most datasets
Hyperparameter tuning improved accuracy significantly
Feature scaling + engineered ratios improved BTC prediction stability
⚠️ Important Lessons
❌ Using raw financial prices causes poor generalization
❌ Ignoring scaling breaks SVM performance
❌ Random train/test split in time-series leads to data leakage
✅ Stationary features improve model stability
✅ Pipelines ensure clean ML workflows
🔥 Future Improvements
Add LSTM / Deep Learning comparison
Use more advanced feature engineering (RSI, MACD)
Apply walk-forward validation for BTC prediction
Deploy model as API (Flask/FastAPI)
