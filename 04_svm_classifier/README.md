# 04 - Support Vector Machine (Classifier)

**Files:** `svm_classifier.py`, `svm_classifier.ipynb`
**Datasets:** blobs (separable), moons (non-linear), Breast Cancer (real)

## What is it?
SVM finds the **optimal hyperplane** that separates classes with the **maximum margin**.

- **Hyperplane:** `w·x + b = 0`
- **Margin:** distance between the two parallel boundaries `w·x + b = ±1`; width = `2 / ‖w‖`
- **Support vectors:** the points lying on/inside the margin; only they determine the boundary

```
Maximize margin  <=>  minimize ½‖w‖²   subject to  yi (w·xi + b) ≥ 1
```

## Hard margin vs soft margin
Real data overlaps, so SVM allows violations with slack variables:
```
minimize ½‖w‖² + C Σ ξi
```
- **Large C** -> narrow margin, few misclassifications (can overfit)
- **Small C** -> wide margin, tolerates errors (can underfit)

## Kernel trick
If data is not linearly separable, map it to a higher dimension where it is. The kernel computes dot-products there **without** explicitly mapping:

| Kernel | Notes |
|--------|-------|
| `linear` | Fast; good for high-dim/text data |
| `poly` | Polynomial boundary; has `degree` |
| `rbf` | Flexible, smooth, default choice |
| `sigmoid` | Neural-net-like, rarely best |

## Hyperparameter tuning
- `C`: margin softness
- `gamma` (rbf): influence radius; high -> overfit, low -> underfit
- `kernel`, `degree`
- Use `GridSearchCV` with `Pipeline(StandardScaler, SVC)` and cross-validation (done in the code).

## Practical notes
- **Always scale features.**
- Multi-class is handled internally via one-vs-one.
- `SVC(probability=True)` gives probabilities (slower, uses Platt scaling).
- Struggles with huge datasets and heavily overlapping noisy classes.

## What the code does
1. Plots the max-margin hyperplane, margins and support vectors; prints `w`, `b`, margin width
2. Compares kernels on moons data with decision boundaries
3. Shows C vs gamma grid (underfit -> overfit)
4. Tunes SVM on Breast Cancer and prints classification report + confusion matrix
