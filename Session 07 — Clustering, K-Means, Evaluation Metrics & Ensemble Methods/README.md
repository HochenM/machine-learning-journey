# Session 07 — Clustering, K-Means, Evaluation Metrics & Ensemble Methods

## 🎯 What I Learned
- Implemented **K-Means clustering** (Lloyd's algorithm) with different K values
- Understood **K-Means++** initialization (spreads initial centers apart)
- Evaluated clustering quality using **Inertia** and **Silhouette Score**
- Used **Elbow Method** to find optimal K from inertia plot
- Tested K-Means on **non-spherical data** (moons, circles) to understand its limitations
- Learned when K-Means works (round, equal-sized) vs fails (irregular shapes)
- Applied **StandardScaler** to normalize BTC, ETH, Oil prices for comparison
- Aligned time-series data using **common dates** (set intersection + `isin`)
- Compared **Bagging vs Boosting vs Stacking** ensemble methods
- Used `set()`, `&`, and `.isin()` for data alignment and filtering

## 📁 Files
- `main.py` — K-Means clustering, evaluation, ensemble methods, financial data comparison
- `outputs/` — Cluster plots, elbow/silhouette curves, scaled price comparisons

## 🔑 Key Concepts Covered

| Concept | Method/Tool | Description |
|---------|------------|-------------|
| K-Means (Lloyd) | `KMeans(n_clusters=K)` | Assign → Update centers → Repeat |
| K-Means++ | Default init in sklearn | Smart center initialization (spreads apart) |
| Inertia | `model.inertia_` | Sum of squared distances to centers |
| Silhouette Score | `silhouette_score(X, labels)` | Measures cluster quality (-1 to +1) |
| Elbow Method | Plot K vs Inertia | Find optimal K at curve bend |
| Bagging | Random Forest | Parallel models, reduce variance |
| Boosting | AdaBoost, XGBoost | Sequential models, reduce bias |
| StandardScaler | `StandardScaler()` | Mean=0, Std=1 |
| Common Dates | `set(A) & set(B)`, `.isin()` | Align time-series data |

## 📊 Results Summary

### Elbow & Silhouette
| K | Inertia | Silhouette |
|:---:|---------|:---:|
| 2 | ~3500 | ~0.35 |
| **3** | **~2000** | **~0.42** 🏆 |
| 4 | ~1500 | ~0.38 |
| 5 | ~1200 | ~0.32 |

**K=3 is optimal — highest silhouette score!**

### K-Means on Non-Spherical Data
| Data | K-Means Works? |
|------|:---:|
| Blobs (round, separated) | ✅ Perfect |
| Moons (crescent shapes) | ❌ Fails |
| Circles (inner/outer) | ❌ Fails |

## 🧠 Key Takeaways
- K-Means++ initializes centers far apart → better clusters
- Inertia always decreases with K → use Elbow, not raw values
- Silhouette considers both tightness AND separation → better metric
- StandardScaler makes different assets comparable on same plot
- `set()` intersection finds common dates — essential for time-series alignment
- K-Means fails on non-spherical data → use DBSCAN or Spectral Clustering instead
- Bagging trains models in parallel (Random Forest), Boosting trains sequentially (XGBoost)
- Stacking combines different model types with a meta-learner on top
