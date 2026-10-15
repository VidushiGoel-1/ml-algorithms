# 10 - K-Means Clustering

**Files:** `kmeans.py`, `kmeans.ipynb`
**Dataset:** synthetic blobs and moons

## What is it?
An **unsupervised** algorithm that groups unlabeled data into **K clusters** so that points in the same cluster are close to its **centroid** (mean).

## Algorithm (Lloyd's)
1. Choose K and initialize K centroids
2. **Assignment:** assign each point to the nearest centroid
3. **Update:** move each centroid to the mean of its assigned points
4. Repeat 2-3 until assignments stop changing (or max iterations)

Objective minimized: **inertia** (WCSS) = `Σ ‖xi - μ(cluster of xi)‖²`. The algorithm always converges, but possibly to a local minimum.

## Initialization
- **Random:** may give poor results; run several times (`n_init`)
- **k-means++ (default):** picks initial centroids far apart -> better, faster convergence (compared in the code)

## Choosing K
| Method | How |
|--------|-----|
| **Elbow** | Plot inertia vs K; choose the "elbow" where improvement slows |
| **Silhouette score** | `(b - a) / max(a, b)` per point (a = mean intra-cluster distance, b = mean distance to nearest other cluster); ranges -1..1, higher is better |
| Domain knowledge | Business-driven number of groups |

## Assumptions and limitations
- Clusters are roughly **spherical** and of **similar size/density** (fails on moons, shown in the code)
- **Feature scaling is required** (code shows the effect)
- Sensitive to outliers (consider K-Medoids) and to K
- Euclidean distance degrades in very high dimensions (use PCA first)
- Need to specify K in advance; alternatives: DBSCAN, Hierarchical, Gaussian Mixture

## Use cases
Customer segmentation, image compression (color quantization), document clustering, anomaly detection, feature engineering.

## What the code does
1. K-Means from scratch with numpy
2. Elbow + silhouette to pick K
3. k-means++ vs random initialization
4. Final model summary and predictions on new points
5. Failure cases: non-spherical clusters and unscaled features
