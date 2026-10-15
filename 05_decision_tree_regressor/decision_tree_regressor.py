# %% [markdown]
# # Decision Tree Regressor
# Recursive binary splitting: at each node choose the feature/threshold that gives the largest
# reduction in impurity (variance / MSE for regression). Leaf prediction = mean of its samples.
# Entropy is used by classification trees; we show the formula here for comparison.
# Overfitting is controlled using pre-pruning (max_depth, min_samples_leaf) and post-pruning (ccp_alpha).

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# %% [markdown]
# ## 1. Understand impurity: entropy (classification) vs variance (regression)

# %%
def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return -(p * np.log2(p)).sum()

print("Entropy of [0,0,1,1]  :", entropy([0, 0, 1, 1]))        # 1.0 -> maximum impurity
print("Entropy of [0,0,0,0]  :", entropy([0, 0, 0, 0]) + 0.0)  # 0.0 -> pure node
print("Variance (regression) :", np.var([10, 12, 11, 40]).round(2))

# %% [markdown]
# ## 2. Manual first split: try every threshold and pick the one with lowest weighted MSE

# %%
rng = np.random.RandomState(1)
X = np.sort(rng.uniform(0, 6, 120)).reshape(-1, 1)
y = np.sin(X).ravel() + rng.normal(0, 0.25, 120)

def best_split(x, y):
    best = (None, np.inf)
    for t in (x[:-1] + x[1:]) / 2:
        left, right = y[x <= t], y[x > t]
        score = (len(left) * left.var() + len(right) * right.var()) / len(y)
        if score < best[1]:
            best = (t, score)
    return best

t, s = best_split(X.ravel(), y)
print(f"Best first split: x <= {t:.3f}  (weighted MSE = {s:.4f}, before split = {y.var():.4f})")
print("sklearn root threshold:", DecisionTreeRegressor(max_depth=1).fit(X, y).tree_.threshold[0].round(3))

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_plot = np.linspace(0, 6, 400).reshape(-1, 1)

# %% [markdown]
# ## 3. Overfitting: effect of max_depth

# %%
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, d in zip(axes, [1, 3, 5, None]):
    m = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X_train, y_train)
    ax.scatter(X_train, y_train, s=10)
    ax.plot(X_plot, m.predict(X_plot), "r")
    ax.set_title(f"depth={d} | train R2={m.score(X_train, y_train):.2f}, test R2={m.score(X_test, y_test):.2f}")
plt.show()

# %% [markdown]
# ## 4. Pruning
# **Pre-pruning:** max_depth, min_samples_split, min_samples_leaf.
# **Post-pruning (cost-complexity):** grow the full tree, then remove weak branches using ccp_alpha.

# %%
full = DecisionTreeRegressor(random_state=0)
path = full.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas[:-1]
trees = [DecisionTreeRegressor(random_state=0, ccp_alpha=a).fit(X_train, y_train) for a in alphas]

plt.plot(alphas, [t.score(X_train, y_train) for t in trees], label="train R2")
plt.plot(alphas, [t.score(X_test, y_test) for t in trees], label="test R2")
plt.xlabel("ccp_alpha"); plt.legend(); plt.title("Post-pruning: pick alpha where test R2 peaks")
plt.show()
best_alpha = alphas[int(np.argmax([t.score(X_test, y_test) for t in trees]))]
print("Best ccp_alpha:", round(best_alpha, 5))

# %% [markdown]
# ## 5. Tune pre-pruning params with GridSearchCV and evaluate

# %%
grid = GridSearchCV(DecisionTreeRegressor(random_state=0),
                    {"max_depth": [2, 3, 4, 5, 6, None], "min_samples_leaf": [1, 3, 5, 10],
                     "ccp_alpha": [0, 0.001, 0.005, 0.01]},
                    cv=5, scoring="neg_mean_squared_error").fit(X_train, y_train)
best = grid.best_estimator_
pred = best.predict(X_test)
print("Best params:", grid.best_params_)
print(f"Test MSE={mean_squared_error(y_test, pred):.4f} | R2={r2_score(y_test, pred):.4f}")

# %% [markdown]
# ## 6. Visualize the tree

# %%
small = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_train, y_train)
plt.figure(figsize=(14, 6))
plot_tree(small, feature_names=["x"], filled=True, rounded=True)
plt.show()
print(export_text(small, feature_names=["x"]))
