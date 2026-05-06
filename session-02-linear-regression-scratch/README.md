# Session 02 — Linear Regression from Scratch & Cost Function

## 🎯 What I Learned
- Built a **manual linear regression solver** without any ML libraries
- Understood the concept of **iterative parameter tuning** (a and b in y = ax + b)
- Calculated **cost function (RMSE)** to evaluate line fit quality
- Visualized how changing slope (a) and intercept (b) affects the prediction line
- Compared **explicit programming** (manual grid search) vs **implicit programming** (scikit-learn)
- Used `LinearRegression()` from sklearn to verify my manual solution
- Understood model coefficients (`coef_` = slope, `intercept_` = y-intercept)
- Plotted cost function curves to visualize optimizer convergence
- Used `mean_squared_error` from sklearn.metrics to evaluate final model

## 📁 Files
- `main.py` — Manual linear regression implementation and sklearn comparison
- `outputs/` — Cost function plots and regression line visualizations

## 🔑 Key Concepts Covered

| Concept | Description |
|---------|-------------|
| **Solver/Optimizer** | Manual tuning of parameters (a, b) to minimize error |
| **RMSE** (Root Mean Squared Error) | Measures average prediction error — lower is better |
| **MSE** (Mean Squared Error) | RMSE² — heavily penalizes large errors |
| **Cost Function** | Tracks RMSE across iterations — forms a U-shape |
| **Explicit Programming** | Manually defining rules and parameters (grid search) |
| **Implicit Programming (AI)** | Letting the algorithm learn from data (sklearn) |
| **model.coef_** | Slope of the best-fit line |
| **model.intercept_** | Y-intercept of the best-fit line |
| **mean_squared_error()** | sklearn function to calculate MSE |

## 📊 Final Results

| Iteration | Method | a (slope) | b (intercept) | RMSE |
|-----------|--------|-----------|---------------|------|
| 1 | Manual guess | 0.0 | 0.0 | **34.59** |
| 2 | Manual guess | 0.1 | 0.5 | **9.09** |
| Best | Grid search (explicit) | 0.11 | 5.0 | **2.69** |
| Optimal | Sklearn (implicit) | 0.103 | 8.665 | **1.61** |

## 📈 Cost Function Analysis

The cost function values form a **U-shaped curve**, proving:
- RMSE starts high (bad fit) → decreases to minimum (best fit) → increases again (worse fit)
- Grid search tested 500+ combinations of `a` (-1 to 1.5) and `b` (0 to 60)
- Best combination found at index 445 out of 500+ iterations

## 🔧 Key Methods Used

```python
# Manual RMSE calculation
error = y - y_pred
squared_error = np.square(error)
mse = np.mean(squared_error)
rmse = np.sqrt(mse)

# Sklearn metrics
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
