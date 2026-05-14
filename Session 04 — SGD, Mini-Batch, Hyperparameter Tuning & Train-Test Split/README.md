# Session 04 — SGD, Mini-Batch, Hyperparameter Tuning & Train-Test Split

## 🎯 What I Learned
- Implemented **Stochastic Gradient Descent (SGD)** from scratch with manual shuffle
- Built **Mini-Batch SGD** to balance speed and stability
- Compared GD, SGD, and Mini-Batch approaches
- Used `SGDRegressor` from sklearn with `learning_rate='constant'`
- Performed **hyperparameter tuning** using manual grid search over lr and max_iter
- Fixed randomness with `random_state=42` for reproducible results
- Split data into train/test sets using `train_test_split`
- Understood when to use `shuffle=False` (time series) vs `shuffle=True`
- Applied `stratify=y` for balanced class splits in classification
- Evaluated models on test data — not just training data

## 📁 Files
- `main.py` — SGD, Mini-Batch SGD, SGDRegressor, hyperparameter tuning, train-test split
- `outputs/` — Cost function plots and regression line visualizations

## 🔑 Key Concepts Covered

| Concept | Description |
|---------|-------------|
| SGD | Updates theta using ONE random data point per step |
| Mini-Batch SGD | Updates theta using a small batch (e.g., 2 points) |
| `SGDRegressor` | sklearn's implementation of SGD for regression |
| Hyperparameter Tuning | Grid search to find best lr and max_iter |
| `random_state` | Fixes randomness for reproducible results |
| Train-Test Split | Separate data for training vs evaluation |
| `shuffle=True/False` | True for random data, False for time series |
| `stratify=y` | Preserves class ratios in classification splits |
| Overfitting Detection | Train MSE << Test MSE = overfitting |

## 📊 Hyperparameter Tuning Results

| lr | Best max_iter | MSE |
|:---:|:---:|------|
| 0.01 | 150 | 0.3729 |
| 0.02 | 150 | 0.3888 |
| **0.03** | **150** | **0.3658** 🏆 |

**Winner: `lr=0.03, max_iter=150`**





## 🧠 Key Takeaways
SGD updates per sample → noisy but escapes local minima; Mini-Batch balances speed and stability

random_state=42 ensures same shuffle order every run → reproducible tuning

Grid search systematically finds best hyperparameters — lr=0.03, max_iter=150 won

MSE stops improving after convergence → extra iterations don't help (MSE flatlines)

Test MSE (not train MSE) is the real measure of model quality

shuffle=False preserves time order — critical for stock prices, blood pressure, etc.

stratify=y keeps class balance in train/test splits — prevents skewed evaluation

Hyperparameter tuning is essential — good lr can cut MSE by 30% vs bad lr

## 🔧 Key Methods Used

```python
# SGD from scratch
for epoch in range(max_iter):
    indices = np.random.permutation(m)
    X_shuffled, y_shuffled = X[indices], y[indices]
    for i in range(m):
        x_i, y_i = X_shuffled[i:i+1], y_shuffled[i:i+1]
        error = x_i @ theta - y_i
        theta = theta - lr * x_i.T @ error

# sklearn SGDRegressor
model = SGDRegressor(learning_rate='constant', eta0=0.03, max_iter=150)
model.fit(X, y)

# Hyperparameter tuning (grid search)
for lr in [0.01, 0.02, 0.03]:
    for max_it in [50, 150, 200, 500, 1000]:
        model = SGDRegressor(eta0=lr, max_iter=max_it)
        mse = mean_squared_error(y, model.fit(X, y).predict(X))

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)
mse_test = mean_squared_error(y_test, model.predict(x_test))  # What matters!
