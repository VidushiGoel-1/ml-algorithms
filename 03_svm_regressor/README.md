# 03 - Support Vector Regression (SVR)

**Files:** `svm_regressor.py`, `svm_regressor.ipynb`
**Dataset:** synthetic non-linear data (sine + trend + noise)

## What is it?
SVM applied to regression. Instead of minimizing squared error for every point, SVR fits a function and allows errors **up to ε (epsilon)** for free. It builds an **ε-insensitive tube** around the prediction:

- Points **inside** the tube -> no penalty
- Points **outside** the tube -> penalized; these become the **support vectors**

```
minimize  ½||w||² + C Σ (ξi + ξi*)
subject to |yi - (w·xi + b)| ≤ ε + ξi
```

Only support vectors define the model, so it is memory-efficient and robust to outliers.

## Kernels (how it handles non-linearity)
The **kernel trick** computes similarity in a higher-dimensional space without actually transforming the data.

| Kernel | Formula | Use when |
|--------|---------|----------|
| `linear` | x·x' | Data is roughly linear, many features |
| `poly` | (γ x·x' + r)^d | Polynomial-like relationships |
| `rbf` (default) | exp(-γ‖x - x'‖²) | General non-linear data - best first choice |
| `sigmoid` | tanh(γ x·x' + r) | Rarely used |

## Hyperparameters
| Parameter | Effect |
|-----------|--------|
| **C** | Penalty for points outside the tube. Large C -> fits training data tightly (overfit risk); small C -> smoother, simpler (underfit risk) |
| **epsilon** | Tube width. Large ε -> fewer support vectors, flatter model; small ε -> follows data closely |
| **gamma** (rbf/poly) | Reach of one point. Large gamma -> wiggly, overfit; small gamma -> smooth |
| **degree** (poly) | Polynomial degree |

Tune with `GridSearchCV` (done in the code) using cross-validation.

## Important
- **Feature scaling is mandatory** (SVR uses distances). Use `StandardScaler` in a pipeline.
- Training is slow on very large datasets (roughly O(n²)-O(n³)).
- Target scaling may be needed when `y` has a very large range.

## What the code does
1. Compares linear/poly/rbf kernels
2. Visualizes the effect of C, epsilon, gamma
3. Grid-searches the best hyperparameters, reports R², MSE, number of support vectors
4. Plots the fitted curve with its ε-tube
