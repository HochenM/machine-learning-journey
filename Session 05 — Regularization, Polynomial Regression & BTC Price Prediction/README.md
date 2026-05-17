# Session 05 — Regularization, Polynomial Regression & BTC Price Prediction

## 🎯 What I Learned
- Applied **L1 (Lasso), L2 (Ridge), and Elastic Net** regularization to prevent overfitting
- Used `SGDRegressor` with `penalty`, `alpha`, `warm_start`, `early_stopping`, `tol`, `n_iter_no_change`
- Understood `fit_intercept=True` adds bias `b` to `y = b + aX`
- Built a BTC price prediction model (tomorrow's close from today's OHLC)
- Created time-series target using `shift(-1)` and cleaned with `dropna()`
- Split time-series data with `shuffle=False` to preserve chronological order
- Diagnosed **overfitting** by comparing train vs test MSE
- Diagnosed **underfitting** when Ridge couldn't learn the pattern
- Used `warm_start=True` to visualize cost function over iterations
- Applied **polynomial regression** (`np.polyfit` + `np.poly1d`) on BTC data
- Performed degree tuning by testing degrees 1-30 and finding the best RMSE
- Learned that higher polynomial degree = more overfitting risk

## 📁 Files
- `main.py` — Regularization, BTC prediction, polynomial regression
- `bitcoin2026.csv` — BTC price data
- `outputs/` — Cost function plots, polynomial fits, and BTC prediction visualizations

## 🔑 Key Concepts Covered

| Concept | Method/Tool | Description |
|---------|------------|-------------|
| L1 Regularization | `Lasso(alpha=0.5)` or `SGDRegressor(penalty='l1')` | Drives some weights to exactly 0 |
| L2 Regularization | `Ridge(alpha=0.5)` or `SGDRegressor(penalty='l2')` | Shrinks all weights toward 0 |
| Warm Start | `warm_start=True` | Continues training from previous solution |
| Early Stopping | `early_stopping=True` | Stops when validation score stops improving |
| Time-Series Split | `shuffle=False` | Preserves chronological order |
| Polynomial Features | `np.polyfit(X, y, degree)` | Creates X, X², ..., Xⁿ features |
| Polynomial Function | `np.poly1d(...)` | Creates callable function from coefficients |
| Overfitting Detection | Train MSE << Test MSE | Model memorized training data |
| Underfitting Detection | Both train and test MSE high | Model too simple |

## 📊 Results Summary

### Regularization on BTC

| Model | Train MSE | Test MSE | Verdict |
|-------|-----------|----------|---------|
| Ridge (L2) | High | High | Underfit — too much penalty |
| Lasso (L1) | Low | Higher | Slight overfit — acceptable for BTC scale |

### Polynomial Degree Tuning

| Degree Range | Best Degree | RMSE Trend |
|:---:|:---:|------|
| 1-30 | 7 | Above 11 = overfitting |
| 1-20 | 7 | Above 11 = bad |
| 1-10 | 7 | **Degree 7 = best balance** |


## 🧠 Key Takeaways
fit_intercept=True adds bias b → model learns y = b + aX

fit_intercept=False forces line through origin → y = aX

Ridge (L2) shrinks all weights — can cause underfitting with high alpha

Lasso (L1) drives some weights to zero — can overfit if alpha too small

shuffle=False is mandatory for time series — preserves temporal order

warm_start=True lets you see how model improves iteration by iteration

Polynomial degree 7 was optimal for BTC — degree > 11 caused overfitting

MSE values look huge for BTC because prices are in tens of thousands

Use RMSE or R² to judge model quality when target values are large

Early stopping prevents wasting iterations when model stops improving

## 🔧 Key Methods Used

```python
# SGD with L1
model = SGDRegressor(penalty='l1', alpha=0.5, warm_start=True, early_stopping=True)

# Lasso & Ridge
model = Lasso(alpha=0.5, max_iter=1000)
model = Ridge(alpha=0.5)

# Time-series target
df["TomorrowClose"] = df["Close"].shift(-1)
df.dropna(inplace=True)

# Train-test split (no shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, shuffle=False)

# Polynomial fit
model = np.poly1d(np.polyfit(X_train, y_train, 7))
y_pred = model(X_test)
