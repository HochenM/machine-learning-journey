# Session 02 — Linear Regression from Scratch & Cost Function

## 🎯 What I Learned
- Built a manual linear regression solver without any ML libraries
- Understood iterative parameter tuning (a and b in y = ax + b)
- Calculated cost function (RMSE) to evaluate line fit quality
- Visualized how changing slope and intercept affects the prediction line
- Compared explicit programming (manual grid search) vs implicit programming (scikit-learn)
- Used `LinearRegression()` from sklearn to verify my manual solution
- Plotted cost function curves to visualize optimizer convergence

## 📁 Files
- `main.py` — Manual linear regression implementation and sklearn comparison
- `outputs/` — Cost function plots and regression line visualizations

## 🔑 Key Concepts Covered

| Concept | Description |
|---------|-------------|
| Solver/Optimizer | Manual tuning of parameters (a, b) to minimize error |
| RMSE | Measures average prediction error — lower is better |
| MSE | RMSE² — heavily penalizes large errors |
| Cost Function | Tracks RMSE across iterations — forms a U-shape |
| Explicit Programming | Manually defining rules and parameters (grid search) |
| Implicit Programming (AI) | Letting the algorithm learn from data (sklearn) |
| `model.coef_` | Slope of the best-fit line |
| `model.intercept_` | Y-intercept of the best-fit line |

## 📊 Final Results

| Iteration | Method | a (slope) | b (intercept) | RMSE |
|-----------|--------|-----------|---------------|------|
| 1 | Manual guess | 0.0 | 0.0 | 34.59 |
| 2 | Manual guess | 0.1 | 0.5 | 9.09 |
| Best | Grid search (explicit) | 0.11 | 5.0 | 2.69 |
| Optimal | Sklearn (implicit) | 0.103 | 8.665 | 1.61 |

## 📈 Cost Function Analysis
- RMSE starts high (bad fit) → decreases to minimum (best fit) → increases again (worse fit)
- Grid search tested 500+ combinations of a (-1 to 1.5) and b (0 to 60)
- Best combination found at index 445
- Final formula: **Price = 0.103 × Area + 8.665**

## 🧠 Key Takeaways
- Explicit programming (grid search) manually iterates through parameters, calculates RMSE for each combination, and picks the best — this is what `LinearRegression().fit()` does automatically
- Implicit programming (sklearn) replaces the entire grid search loop with one line: `model.fit(X, y)`
- Cost Function (RMSE) measures prediction error — starts high (34.59), drops as parameters improve (9.09), reaches minimum at best fit (2.69 manual, 1.61 sklearn)
- Sklearn uses Normal Equation (exact mathematical solution) — beats manual grid search by finding the precise minimum
- Grid search step size matters: smaller steps → better results but more computation
- `model.coef_` gives slope (0.103), `model.intercept_` gives y-intercept (8.665)
- `min(cost_fiction)` identifies the best parameters — the lowest point on the U-shaped cost curve
- Manual approach demystifies what happens inside sklearn's `fit()`
- Nested loops with `np.arange()` enable systematic parameter exploration
- Cost function curve is U-shaped — proves there's a single best solution
