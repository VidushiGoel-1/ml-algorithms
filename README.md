# Machine Learning Algorithms (from scratch + scikit-learn)

A hands-on collection of 10 core ML algorithms. Every algorithm has its own folder with:

- a **detailed README** (intuition, maths, key concepts, hyperparameters, pros/cons, when to use)
- a runnable **`.py`** file (cells marked with `# %%`, so it also works in VS Code / Spyder)
- a **`.ipynb`** notebook generated from the same code

All datasets are built into scikit-learn, so no download is needed.

| # | Algorithm | Type | Key topics covered |
|---|-----------|------|--------------------|
| 01 | [Multiple Linear Regression](01_multiple_linear_regression) | Regression | multiple features -> output, R², MSE, coefficients |
| 02 | [Polynomial Regression](02_polynomial_regression) | Regression | non-linear data, overfitting, Ridge/Lasso regularization |
| 03 | [SVM Regressor (SVR)](03_svm_regressor) | Regression | kernels, epsilon-tube, hyperparameter tuning |
| 04 | [SVM Classifier](04_svm_classifier) | Classification | optimal hyperplane, kernel trick, hyperparameter tuning |
| 05 | [Decision Tree Regressor](05_decision_tree_regressor) | Regression | recursive splits, entropy/variance, pruning |
| 06 | [Decision Tree Classifier](06_decision_tree_classifier) | Classification | splits, overfitting, confusion matrix |
| 07 | [Random Forest](07_random_forest) | Both | bagging, accuracy boost, overfitting reduction |
| 08 | [K-Nearest Neighbors](08_knn) | Both | distance metrics, K tuning, scaling |
| 09 | [Naive Bayes](09_naive_bayes) | Classification | Bayes theorem, conditional probability |
| 10 | [K-Means](10_kmeans) | Clustering | centroids, elbow, silhouette, k-means++ |

## Setup

```bash
git clone <your-repo-url>
cd ml-algorithms
pip install -r requirements.txt
python make_notebooks.py          # (re)generates .ipynb files from the .py files
```

Run any algorithm:

```bash
cd 01_multiple_linear_regression
python multiple_linear_regression.py      # or open the .ipynb in Jupyter
```

## Common workflow used in every folder

1. Load data -> 2. Train/test split -> 3. (Scale if needed) -> 4. Fit model -> 5. Evaluate -> 6. Tune hyperparameters with cross-validation -> 7. Visualize.

## Cheat sheet: which metric?

| Task | Metrics |
|------|---------|
| Regression | R², Adjusted R², MSE, RMSE, MAE |
| Classification | Accuracy, Precision, Recall, F1, Confusion matrix |
| Clustering | Inertia (elbow), Silhouette score |
