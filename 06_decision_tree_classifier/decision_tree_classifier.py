# %% [markdown]
# # Decision Tree Classifier
# Splits using Gini impurity or Entropy / Information Gain.
# Deep trees overfit -> use pruning. Evaluate with confusion matrix, precision, recall, F1.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)

# %%
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# %% [markdown]
# ## 1. How a split is chosen: Gini, Entropy, Information Gain (manual calculation)

# %%
def gini(labels):
    _, c = np.unique(labels, return_counts=True)
    p = c / c.sum()
    return 1 - (p ** 2).sum()

def entropy(labels):
    _, c = np.unique(labels, return_counts=True)
    p = c / c.sum()
    return -(p * np.log2(p)).sum()

def information_gain(parent, left, right):
    n = len(parent)
    return entropy(parent) - (len(left) / n) * entropy(left) - (len(right) / n) * entropy(right)

feature = 0  # mean radius
thr = np.median(X_train[:, feature])
left, right = y_train[X_train[:, feature] <= thr], y_train[X_train[:, feature] > thr]
print(f"Parent  gini={gini(y_train):.3f} entropy={entropy(y_train):.3f}")
print(f"Split on '{data.feature_names[feature]}' <= {thr:.2f}")
print(f"Information gain = {information_gain(y_train, left, right):.4f}")

# %% [markdown]
# ## 2. Gini vs Entropy

# %%
for crit in ["gini", "entropy"]:
    m = DecisionTreeClassifier(criterion=crit, random_state=0).fit(X_train, y_train)
    print(f"{crit:8s} -> train acc={m.score(X_train, y_train):.3f} | test acc={m.score(X_test, y_test):.3f} | depth={m.get_depth()} leaves={m.get_n_leaves()}")

# %% [markdown]
# ## 3. Overfitting: accuracy vs tree depth

# %%
depths = range(1, 15)
tr, te = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_train, y_train)
    tr.append(m.score(X_train, y_train)); te.append(m.score(X_test, y_test))
plt.plot(depths, tr, "o-", label="train"); plt.plot(depths, te, "o-", label="test")
plt.xlabel("max_depth"); plt.ylabel("accuracy"); plt.legend()
plt.title("Train accuracy keeps rising -> overfitting; test accuracy plateaus")
plt.show()

# %% [markdown]
# ## 4. Pruning: pre-pruning (GridSearch) + post-pruning (ccp_alpha)

# %%
grid = GridSearchCV(DecisionTreeClassifier(random_state=0),
                    {"criterion": ["gini", "entropy"], "max_depth": [2, 3, 4, 5, 6, None],
                     "min_samples_leaf": [1, 3, 5, 10], "ccp_alpha": [0, 0.005, 0.01, 0.02]},
                    cv=5, scoring="accuracy").fit(X_train, y_train)
print("Best params:", grid.best_params_, "| CV acc:", round(grid.best_score_, 4))
best = grid.best_estimator_

# %% [markdown]
# ## 5. Evaluation: confusion matrix and report

# %%
y_pred = best.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, target_names=data.target_names))
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)
ConfusionMatrixDisplay(cm, display_labels=data.target_names).plot(cmap="Blues")
plt.show()

# %% [markdown]
# ## 6. Visualize the pruned tree and feature importance

# %%
plt.figure(figsize=(18, 8))
plot_tree(best, feature_names=data.feature_names, class_names=data.target_names, filled=True, max_depth=3, fontsize=8)
plt.show()

imp = best.feature_importances_
idx = np.argsort(imp)[-8:]
plt.barh(data.feature_names[idx], imp[idx]); plt.title("Top feature importances")
plt.show()
