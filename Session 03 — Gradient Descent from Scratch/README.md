# Session 03 — Gradient Descent from Scratch

## 🎯 What I Learned
- Implemented gradient descent algorithm manually for linear regression
- Added bias column (column of 1s) to enable intercept in matrix form
- Understood why `X_bias @ theta` equals `b + a*X`
- Built a training loop with cost function tracking (MSE)
- Tuned hyperparameters: learning rate and max iterations
- Visualized cost function curve to monitor convergence
- Diagnosed convergence issues (cost increasing instead of decreasing)
- Learned the importance of data standardization for gradient descent
- Applied gradient descent to `make_regression` random data
- Extracted final parameters (slope and intercept) from trained theta
- Plotted the best-fit regression line against scatter data

## 📁 Files
- `main.ipynb` — Jupyter notebook with all exercises
- `outputs/` — Cost function plots and regression line visualizations

## 🔑 Key Concepts Covered

| Concept | Description |
|---------|-------------|
| Bias Column | Adding column of 1s to X to allow intercept in matrix multiplication |
| `X_bias @ theta` | Matrix form of `y = b + a*X` |
| Gradient Descent | Iterative algorithm to minimize cost by updating parameters |
| Gradient Calculation | `X_bias.T @ error / n_samples` — transpose aligns dimensions |
| Learning Rate | Controls step size — too high diverges, too low converges slowly |
| Cost Function (MSE) | Measures average squared error — should decrease over iterations |
| Convergence | Cost stabilizes at minimum when model learns |
| Standardization | `(X - mean) / std` — makes features comparable for stable GD |
| `np.c_[bias, X]` | Concatenates bias column with features |
| `theta.ravel()` | Flattens theta array to extract b and a |

## 📊 Final Results

| Dataset | Learning Rate | Iterations | a (slope) | b (intercept) | Final MSE |
|---------|:---:|:---:|-----------|---------------|-----------|
| House prices (5 samples) | 0.00001 | 25 | 0.0811 | 0.0009 | 46.66 |
| Random data (80 samples) | 0.001 | 20 | -0.0101 | -0.0132 | 21.10 |

## 📈 Cost Function Analysis
- House prices: MSE dropped from 770 → 46.66 in 25 iterations, converging steadily
- Random data: MSE increased from 21.08 → 21.10 across 20 iterations — learning rate too small for this data scale
- Both datasets showed convergence or near-convergence within few iterations
- House prices data had small X values (100-500) but large y values, requiring very small learning rate (0.00001)

## 🧠 Key Takeaways
- `X_bias @ theta = b + a*X` — the bias column is the mathematical trick for intercept
- Gradient descent computes `X_bias.T @ error` — transpose aligns dimensions: (2,n) @ (n,) = (2,)
- `theta = theta - gradient * learning_rate` — parameters move opposite to gradient direction
- Learning rate must be tuned: 0.00001 worked for house prices, 0.001 was too high for random data
- Cost function should decrease — if it increases, reduce learning rate or standardize data
- `sklearn.LinearRegression()` uses Normal Equation — no standardization or learning rate needed
- Manual gradient descent helps understand what happens inside `model.fit()`
- `make_regression` generates data with different scales — needs standardization for stable GD
- `theta.ravel()` converts (2,1) array to 1D to extract b and a values
- Even 20-25 iterations can find reasonable parameters on small datasets
