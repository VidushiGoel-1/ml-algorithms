# %% [markdown]
# # Support Vector Machine (Classifier)
# Finds the optimal hyperplane that maximizes the margin between classes.
# Kernel trick handles non-linear data. Tune C, gamma, kernel.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs, make_moons, load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# %% [markdown]
# ## 1. Optimal hyperplane, margin and support vectors (linearly separable data)

# %%
X, y = make_blobs(n_samples=80, centers=2, cluster_std=1.0, random_state=6)
clf = SVC(kernel="linear", C=1000).fit(X, y)   # large C ~ hard margin

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=30)
ax = plt.gca()
xx, yy = np.meshgrid(np.linspace(*ax.get_xlim(), 200), np.linspace(*ax.get_ylim(), 200))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])  # margin & hyperplane
ax.scatter(*clf.support_vectors_.T, s=120, facecolors="none", edgecolors="g", lw=2, label="support vectors")
plt.legend(); plt.title("Maximum-margin hyperplane (w.x + b = 0)")
plt.show()
w, b = clf.coef_[0], clf.intercept_[0]
print("w =", w.round(3), "| b =", round(b, 3), "| margin width = 2/||w|| =", round(2 / np.linalg.norm(w), 3))

# %% [markdown]
# ## 2. Kernel trick on non-linear data (moons)

# %%
Xm, ym = make_moons(n_samples=300, noise=0.25, random_state=0)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.3, random_state=0)

def plot_boundary(model, X, y, ax, title):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 300),
                         np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=15, edgecolors="k", linewidths=0.3)
    ax.set_title(title)

fig, axes = plt.subplots(1, 4, figsize=(19, 4))
for ax, kernel in zip(axes, ["linear", "poly", "rbf", "sigmoid"]):
    m = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1)).fit(Xm_tr, ym_tr)
    plot_boundary(m, Xm_tr, ym_tr, ax, f"{kernel}: test acc={m.score(Xm_te, ym_te):.2f}")
plt.show()

# %% [markdown]
# ## 3. Effect of C and gamma (RBF)

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, C in enumerate([0.1, 100]):
    for j, g in enumerate([0.1, 1, 50]):
        m = SVC(kernel="rbf", C=C, gamma=g).fit(Xm_tr, ym_tr)
        plot_boundary(m, Xm_tr, ym_tr, axes[i, j], f"C={C}, gamma={g} | test acc={m.score(Xm_te, ym_te):.2f}")
plt.show()

# %% [markdown]
# ## 4. Hyperparameter tuning on a real dataset (Breast Cancer)

# %%
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, stratify=data.target, random_state=42)

pipe = make_pipeline(StandardScaler(), SVC())
param_grid = {"svc__kernel": ["linear", "rbf", "poly"],
              "svc__C": [0.1, 1, 10, 100],
              "svc__gamma": ["scale", 0.001, 0.01, 0.1]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1).fit(X_train, y_train)
print("Best params :", grid.best_params_)
print("Best CV acc :", round(grid.best_score_, 4))

y_pred = grid.predict(X_test)
print("Test acc    :", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, target_names=data.target_names))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=data.target_names)
plt.show()
