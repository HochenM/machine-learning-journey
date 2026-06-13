# Session 09 — Feature Engineering, Polynomial Regression, ML Pipelines & Model Persistence

## 🎯 What I Learned
- **Feature Engineering vs Feature Extraction** — Engineering creates new columns from domain knowledge; Extraction transforms data into new space (PCA, t-SNE)
- **Technical Indicators** — Created **Simple Moving Average (SMA_30)** as a trend indicator for BTC price prediction
- **Time Series Feature Engineering** — Returns (Close - Open) and Log Returns for capturing price movements
- **5 ML Input-Output Approaches** — Single/Multiple features → Single/Multiple targets, plus sequencing/sliding window
- **Polynomial Regression** — Created polynomial features (X, X², X³...) to model non-linear relationships
- **Feature Scaling** — Scale X (features) but NOT y (target) — original scale is meaningful for interpretation
- **ML Pipelines** — Built automated workflows connecting preprocessing → model training → prediction
- **ColumnTransformer** — Applied different preprocessing to different column types (numeric → scale, text → TF-IDF)
- **Model Persistence** — Saved and loaded trained models using Python's `pickle` module
- **Multi-Output Time Series** — Predicted next 1, 2, and 3 days' closing prices simultaneously

## 📁 Files
- `Session_9.ipynb` — Feature engineering, polynomial regression, pipelines, ColumnTransformer, pickling
- `BTC-USD.csv` — Bitcoin historical data (2014-2024, 3555 rows × 7 columns)
- `data.csv` — Sample dataset for ColumnTransformer demo (day, amount, count, result)
- `pipeline.pkl` — Saved scikit-learn pipeline with preprocessing + KNeighborsClassifier

## 🔑 Key Concepts Covered

| Concept | Method/Tool | Description |
|---------|------------|-------------|
| Simple Moving Average | `df['Close'].rolling(30).mean()` | Smooths price data, shows trend direction |
| Returns | `df['Close'] - df['Open']` | Captures daily price change |
| Log Returns | `np.log(df['Return'])` | Normalizes returns for percentage comparisons |
| Single Feature → Single Target | `X = [Close]` → `y = [t+1]` | Basic time series prediction |
| Multi-Output Regression | `y = [t+1, t+2, t+3]` | Predicts multiple future time steps |
| Sliding Window (Sequencing) | Loop creating `sequence_len` windows | Converts time series to supervised learning |
| Polynomial Features | `PolynomialFeatures(degree=2)` | Creates X², X³ terms for non-linear relationships |
| Feature Scaling | `StandardScaler()` | Standardizes features (mean=0, std=1) |
| ML Pipeline | `Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())])` | Chains preprocessing and model |
| ColumnTransformer | `ColumnTransformer([('scaler', StandardScaler(), numeric_cols), ('encode', TfidfVectorizer(), text_cols)])` | Different preprocessing per column type |
| Model Persistence | `pickle.dump(model, file)` | Saves trained model to disk |
| Feature Selection (Lasso) | `Lasso()` | Drives unimportant features to zero |

## 📊 Results Summary

### Time Series Feature Engineering on BTC

| Feature | Formula | Purpose |
|---------|---------|---------|
| Return | Close - Open | Daily price movement |
| Log Return | log(Close - Open) | Normalized returns |
| SMA_30 | rolling(30).mean() | 30-day trend indicator |

### ML Input-Output Approaches

| Approach | X Shape | y Shape | When to Use |
|----------|---------|---------|--------------|
| Single → Single | (n, 1) | (n, 1) | Simple prediction |
| Single → Multiple | (n, 1) | (n, 3) | Multi-step forecast |
| Multiple → Single | (n, 4) | (n, 1) | Using all OHLC data |
| Multiple → Multiple | (n, 4) | (n, 3) | Full information + multi-step |
| Sequencing (Window=7) | (n-7, 7) | (n-7, 1) | Uses past 7 days to predict next |

### Polynomial Features

| Original Features (n) | Degree (d) | Polynomial Features Count | Formula |
|:---------------------:|:----------:|:-------------------------:|---------|
| 1 | 2 | 3 | 1, x, x² |
| 2 | 2 | 6 | 1, x₁, x₂, x₁², x₁x₂, x₂² |
| 3 | 2 | 10 | C(3+2, 2) = 10 |
| 4 | 2 | 15 | C(4+2, 2) = 15 |

### Pipeline with ColumnTransformer

| Column Type | Preprocessing | Example Columns |
|-------------|---------------|-----------------|
| Numeric | StandardScaler | amount, count |
| Text/Categorical | TfidfVectorizer | day (mon, tue, wed...) |

## 🧠 Key Takeaways

### Feature Engineering vs Feature Extraction
- **Engineering**: Domain knowledge-based (SMA, Returns, Ratios). Original features stay.
- **Extraction**: Mathematical transformation (PCA, t-SNE). Original features replaced.

### Time Series Rules
- **Don't shuffle** when splitting — preserves chronological order
- **SMA adds context** — price above/below SMA indicates trend
- **Multi-output gets harder** — t+1 easiest, t+3 hardest
- **Sequencing turns 1D → 2D** — makes time series work with sklearn

### Polynomial Regression
- Use when relationship between X and y is **curved/non-linear**
- Formula: y = b₀ + b₁x + b₂x² + b₃x³ + ...
- Higher degree = more overfitting risk
- Number of features = C(n + d, d) where C is combination

### Scaling Rules
- ✅ **Scale X** — Distance-based algorithms (KNN, SVM) and gradient descent need it
- ❌ **Don't scale y** — Original scale is meaningful (e.g., dollars for price prediction)

### Pipeline Benefits
- Prevents **training-serving skew** — same logic for training and prediction
- Automates the entire workflow
- Makes code cleaner and reproducible

### ColumnTransformer Rules
- Each transformer gets **column names or indices**
- Unspecified columns are **dropped** by default
- Use `sparse_threshold=0` to return dense array (avoid sparse matrix issues)
- Perfect for mixed data types (numeric + text + categorical)

### Pickle Best Practices
- ✅ Always use `with open(file, 'wb')` — auto-closes file
- ✅ Save the entire pipeline, not just the model
- ✅ Check model type after loading: `type(pipeline)`
- ⚠️ Avoid `open()` without `with` — risk of memory leaks

## 🔧 Key Methods Used

```python
# Feature Engineering - SMA (Simple Moving Average)
df["SMA_30"] = df["Close"].rolling(30).mean()

# Returns and Log Returns
df["Return"] = df["Close"] - df["Open"]
df["LogReturn"] = np.log(df["Return"])  # Warning: log(negative) = NaN

# Multi-Output Time Series (predict next 3 days)
df["CloseTomorrow_1"] = df["Close"].shift(-1)
df["CloseTomorrow_2"] = df["Close"].shift(-2)
df["CloseTomorrow_3"] = df["Close"].shift(-3)

X = df[["Open", "High", "Low", "Close"]]  # 4 features
y = df[["CloseTomorrow_1", "CloseTomorrow_2", "CloseTomorrow_3"]]  # 3 targets

# Sliding Window (Sequencing) - Single Feature
sequence_len = 7
X, y = [], []
for i in range(0, n_samples - sequence_len):
    X.append(df["Close"].iloc[i:i+sequence_len].tolist())  # shape: (n-7, 7)
    y.append(df["Close"].iloc[i + sequence_len])

# Sliding Window - Multiple Targets
sequence_x, sequence_y = 7, 3
X, y = [], []
for i in range(0, n_samples - sequence_x - sequence_y + 1):
    X.append(df["Close"].iloc[i:i+sequence_x])
    y.append(df["Close"].iloc[i+sequence_x : i+sequence_x+sequence_y])

# Polynomial Features
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)  # Adds x², x₁x₂ terms

# Number of polynomial features
from math import comb
n_features = comb(n_original_features + degree, degree)

# Basic Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier(n_neighbors=5))
])

# ColumnTransformer - Apply different preprocessing to different columns
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

pre_process = ColumnTransformer([
    ("scaler", StandardScaler(), ["amount", "count"]),          # Numeric → Scale
    ("encode", TfidfVectorizer(), "day")                        # Text → TF-IDF
], sparse_threshold=0)  # Force dense array output

# Pipeline with ColumnTransformer
full_pipeline = Pipeline([
    ('pre_processing', pre_process),
    ('clf', KNeighborsClassifier(n_neighbors=1))
])

# Fit and predict
full_pipeline.fit(X_train, y_train)
predictions = full_pipeline.predict(X_test)

# Auto-detect column types
text_columns = df.select_dtypes(include=["object", "string"])
numeric_columns = df.select_dtypes(include=["number", "int", "float"])

# ColumnTransformer with dynamic column selection
pre_process = ColumnTransformer([
    ("scaler", StandardScaler(), numeric_columns),
    ("encode", TfidfVectorizer(), text_columns)
])

# Save model with pickle (BEST PRACTICE)
import pickle

with open('pipeline.pkl', 'wb') as file:
    pickle.dump(full_pipeline, file)

# Load model
with open('pipeline.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)

# Check loaded object
print("Type:", type(loaded_pipeline))
print("Steps:", loaded_pipeline.named_steps)

# Alternative pickling (without 'with' - need manual close)
file = open('data.pkl', 'wb')
pickle.dump(data, file)
file.close()  # DON'T FORGET THIS!

loaded = pickle.load(open('data.pkl', 'rb'))
