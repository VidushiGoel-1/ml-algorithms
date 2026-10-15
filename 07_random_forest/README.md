# 07 - Random Forest

**Files:** `random_forest.py`, `random_forest.ipynb`
**Datasets:** Breast Cancer (classification), Diabetes (regression)

## What is it?
An **ensemble of many decision trees**. Each tree is trained a little differently, and their predictions are combined:
- Classification -> **majority vote**
- Regression -> **average**

## Why combine trees? Bagging
**Bagging = Bootstrap AGGregatING**
1. Draw a **bootstrap sample** (random rows, *with replacement*) for each tree.
2. Train a deep tree on each sample.
3. Aggregate predictions.

Individual deep trees have low bias but high variance. Averaging many decorrelated trees keeps the low bias and **cuts the variance** -> better accuracy and much less overfitting.

## What makes it a *random* forest
On top of bagging, each split considers only a **random subset of features** (`max_features`, default `sqrt(n)` for classification). This **decorrelates** the trees; averaging correlated trees helps much less.

## How to increase accuracy / reduce overfitting
| Hyperparameter | Effect |
|----------------|--------|
| `n_estimators` | More trees -> more stable; gains flatten after ~100-300; never causes overfitting by itself |
| `max_depth`, `min_samples_leaf` | Limit tree complexity -> less overfitting |
| `max_features` | Smaller -> more diverse trees, less overfit |
| `class_weight` | Handles imbalanced classes |
| `oob_score=True` | Free validation using samples left out of each bootstrap (~37%) |

Tune with `RandomizedSearchCV` / `GridSearchCV`.

## Feature importance
Computed from the total impurity reduction each feature provides across trees. Useful for feature selection, but biased toward high-cardinality features (consider permutation importance for a fairer view).

## Pros / Cons
- High accuracy, robust to outliers/noise, little tuning, no scaling, built-in importance and OOB estimate
- Less interpretable than a single tree, slower and larger, can't extrapolate (regression)

## What the code does
1. Single tree vs Bagging vs Random Forest (CV)
2. Bagging implemented by hand (bootstrap + vote)
3. Accuracy and OOB score vs number of trees
4. RandomizedSearchCV tuning, confusion matrix
5. Feature importance plot
6. RandomForestRegressor example
