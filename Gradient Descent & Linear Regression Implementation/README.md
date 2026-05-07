# Session 03 — Gradient Descent & Linear Regression Implementation

## 🎯 What I Learned
- Implemented gradient descent from scratch to solve linear regression
- Understood vectorized matrix operations for efficient computation
- Learned how adding a bias column enables `y = aX + b` in matrix form
- Tracked cost function (MSE & RMSE) across iterations to monitor convergence
- Tuned hyperparameters (learning rate, iterations) to optimize model performance
- Generated random regression data using `make_regression` for realistic testing
- Visualized regression lines overlaid on scatter plots to validate model fit
- Compared model behavior on small synthetic data vs larger random datasets

## 📁 Files
- `main.ipynb` — Full gradient descent implementation with visualizations

## 🔑 Key Concepts Covered

| Concept | Description |
|---------|-------------|
| Bias Term | Adding column of ones (`np.c_[np.ones(...), X]`) to handle intercept in matrix form |
| Gradient Descent | Iterative optimizer: `theta = theta - learning_rate × gradient` |
| Gradient Formula | `X_bias.T @ error / n_samples` — derivative of cost w.r.t. parameters |
| MSE | Mean Squared Error — `np.mean(np.square(y - y_pred))` |
| RMSE | Root Mean Squared Error — `np.sqrt(MSE)` |
| Learning Rate | Controls step size (0.00001, 0.001) — too small = slow, too large = unstable |
| Cost Function | Tracks loss across iterations — should decrease and plateau |
| Vectorized Operations | Matrix multiplication (`@`) instead of loops for speed |
| `make_regression` | Generates synthetic data with controlled noise and samples |

## 📊 Model Results

### Part 1: Small Synthetic Data (5 samples)

| Metric | Value |
|--------|-------|
| Learning Rate | 0.00001 |
| Iterations | 25 |
| Initial Loss (MSE) | 770.0 |
| Final Loss (MSE) | 46.66 |
| a (slope) | 0.0811 |
| b (intercept) | 0.00094 |

### Part 2: Random Generated Data (80 samples)

| Metric | Value |
|--------|-------|
| Learning Rate | 0.001 |
| Iterations | 20 |
| Initial Loss (MSE) | 21.077 |
| Final Loss (MSE) | 21.104 |

### Part 2: Hyperparameter Tuning Attempt

| Iteration | Method | Learning Rate | Max Iter | Final MSE |
|-----------|--------|---------------|----------|-----------|
| 1 | Gradient Descent | 0.001 | 20 | 21.104 |

**Note:** The loss slightly increased over iterations, indicating the learning rate may be too high or features need scaling.

## 📈 Cost Function Analysis
- **Part 1:** MSE dropped sharply from 770 → 53.9 in just 2 iterations, then plateaued at ~46.66 — shows good convergence with low learning rate
- **Part 2:** Loss slightly increased from 21.077 → 21.104 over 20 iterations — suggests suboptimal hyperparameters
- Cost function curve plotted in red for both parts to visualize convergence behavior
- RMSE also tracked in Part 2 for interpretable error measurement

## 🧠 Key Takeaways
- **Gradient descent updates parameters iteratively** — each step moves theta toward the minimum by subtracting `learning_rate × gradient`
- **Matrix formulation is elegant:** `X_bias @ theta` replaces `a×X + b` for all samples at once
- **Adding bias column** (`np.c_[np.ones(...), X]`) is the trick that makes vectorized linear regression work
- **Learning rate is critical:** 0.00001 converged well but slowly; 0.001 caused instability on random data
- **Gradient calculation** (`X_bias.T @ error / n_samples`) is the mathematical engine — it tells us which direction to move parameters
- **Cost function monitoring** reveals if your model is learning or diverging
- **Random data from `make_regression`** provides realistic testing with controlled noise levels
- **Parameter extraction** uses `theta.ravel()` to separate slope (a) and intercept (b)
- **Visual validation:** Overlaying regression line on scatter plot instantly confirms model quality
- **Small datasets (5 samples)** train fast but may not generalize; larger datasets (80 samples) require careful tuning
