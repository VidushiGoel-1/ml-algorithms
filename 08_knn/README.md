# 08 - K-Nearest Neighbors (KNN)

**Files:** `knn.py`, `knn.ipynb`
**Datasets:** Wine (classification), Diabetes (regression)

## What is it?
A **lazy / instance-based** algorithm: there is no training phase, it just stores the data. To predict for a new point:
1. Compute distance to **all** training points
2. Take the **K nearest**
3. **Classification:** majority vote (optionally weighted by 1/distance). **Regression:** mean (or weighted mean) of their targets.

## Distance metrics
| Metric | Formula | Notes |
|--------|---------|-------|
| **Euclidean** | √Σ(xi - yi)² | Default, straight-line distance |
| **Manhattan** | Σ\|xi - yi\| | Grid-like, more robust in high dimensions/outliers |
| **Minkowski** | (Σ\|xi - yi\|^p)^(1/p) | Generalization: p=1 Manhattan, p=2 Euclidean |
| **Chebyshev** | max\|xi - yi\| | Largest single-feature difference |
| **Cosine** | 1 - cos(θ) | Direction, not magnitude; text/embeddings |

## Choosing K
- **Small K (e.g. 1)** -> very flexible, noisy boundary, **overfits** (high variance)
- **Large K** -> smooth boundary, **underfits** (high bias)
- Pick K by cross-validation (plot of CV accuracy vs K in the code).
- Use an **odd K** for binary classification to avoid ties.
- Rule of thumb start: K ≈ √n.

## Other important points
- **Feature scaling is essential** (code shows the accuracy difference). Use `StandardScaler` or `MinMaxScaler`.
- **Curse of dimensionality:** in high dimensions all points look equally far; do feature selection/PCA.
- `weights="distance"` gives closer neighbours more influence.
- Prediction is slow on big data (O(n) per query); use `kd_tree`/`ball_tree` or approximate search.
- Sensitive to irrelevant features and class imbalance.

## What the code does
1. KNN from scratch (numpy) vs sklearn
2. Scaled vs unscaled comparison
3. Different distance metrics (and Minkowski p)
4. K vs train/CV accuracy curve
5. GridSearchCV over K, weights, metric + confusion matrix
6. KNN regression for different K
