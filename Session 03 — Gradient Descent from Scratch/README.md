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
## 🔧 Theta Update Formula

```python
# Gradient Descent Update Rule:
theta_new = theta_old - learning_rate * gradient

# Where gradient is:
gradient = X_bias.T @ error / n_samples

# Full expansion:
theta = theta - α * (Xᵀ * error / n)

## 📊 Final Results

| Dataset | Learning Rate | Iterations | a (slope) | b (intercept) | Final MSE |
|---------|:---:|:---:|-----------|---------------|-----------|
| House prices (5 samples) | 0.00001 | 25 | 0.0811 | 0.0009 | 46.66 |
| Random data (80 samples) | 0.001 | 20 | -0.0101 | -0.0132 | 21.10 |

## 📈 Cost Function Analysis
- House prices: MSE dropped from 770 → 46.66 in 25 iterations, converging steadily
- Random data: MSE increased from 21.08 → 21.10 across all 20 iterations — **model failed to converge**
- Despite many attempts, couldn't find a learning rate and max_iter combination that made the cost decrease on random data
- This demonstrated the real-world challenge of gradient descent: **data from `make_regression` required standardization** that wasn't applied, causing unstable gradients

## ⚠️ Challenges Faced
- Cost function kept increasing regardless of how learning rate was adjusted
- Tried various learning rates (0.00001, 0.0001, 0.001, 0.01) — none produced stable convergence
- Realized that without standardizing `make_regression` data, gradient descent becomes highly unstable
- Manually tuning hyperparameters by hand is difficult and inefficient — this is why libraries like sklearn automate this process

## 🧠 Key Takeaways
- `X_bias @ theta = b + a*X` — the bias column is the mathematical trick for intercept
- Gradient descent computes `X_bias.T @ error` — transpose aligns dimensions: (2,n) @ (n,) = (2,)
- `theta = theta - gradient * learning_rate` — parameters move opposite to gradient direction
- Learning rate must be carefully tuned — too high causes divergence, too low causes painfully slow convergence
- **Data standardization is essential** for gradient descent to work properly on arbitrary datasets
- `sklearn.LinearRegression()` avoids all these issues by using Normal Equation instead
- Failed convergence is a valuable learning experience — it teaches why ML libraries are necessary
- Manual hyperparameter tuning by trial and error is impractical for real-world problems
- `make_regression` data requires standardization because features and targets have very different scales
- Even when GD doesn't work perfectly, the mathematical principles behind it are sound
