# Session 01 — Data Exploration & make_blobs

## 🎯 What I Learned
- Created synthetic clustering data using `sklearn.datasets.make_blobs`
- Understood parameters: `n_samples`, `n_features`, `centers`, `center_box`
- Visualized 3D data through multiple 2D projections

## 📁 Files
- `session01_blobs.py` — Main script
- `outputs/blobs_visualization.png` — Visualization

## 🔑 Parameters

| Parameter | Meaning | Value |
|-----------|---------|-------|
| `n_samples` | Total data points | 200 |
| `n_features` | Dimensions | 3 |
| `centers` | Number of clusters (default) | 3 |
| `center_box` | Range for centers | (5, 10) |
| `cluster_std` | Spread of points | 0.5 |

## 🧠 Key Takeaways
- `make_blobs` is for testing clustering algorithms
- 3D data can be viewed via 2D projections
- `random_state=42` makes results reproducible
