# %% [markdown]
# # Support Vector Regression (SVR)
# SVR fits a function inside an epsilon-tube; points outside the tube become support vectors.
# Key knobs: kernel, C, epsilon, gamma.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# %% [markdown]
# ## 1. Non-linear data

# %%
rng = np.random.RandomState(42)
X = np.sort(rng.uniform(0, 10, 150)).reshape(-1, 1)
y = np.sin(X).ravel() * 3 + 0.5 * X.ravel() + rng.normal(0, 0.4, 150)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
X_plot = np.linspace(0, 10, 400).reshape(-1, 1)

# %% [markdown]
# ## 2. Effect of kernel (linear / poly / rbf)
# SVR is distance based, so we always scale features (pipeline).

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, kernel in zip(axes, ["linear", "poly", "rbf"]):
    m = make_pipeline(StandardScaler(), SVR(kernel=kernel, C=10, epsilon=0.2)).fit(X_train, y_train)
    pred = m.predict(X_test)
    ax.scatter(X_train, y_train, s=12, alpha=0.5)
    ax.plot(X_plot, m.predict(X_plot), "r", lw=2)
    ax.set_title(f"{kernel}: R2={r2_score(y_test, pred):.2f}, MSE={mean_squared_error(y_test, pred):.2f}")
plt.show()

# %% [markdown]
# ## 3. Effect of C, epsilon, gamma (RBF kernel)

# %%
configs = [dict(C=0.1, epsilon=0.2, gamma="scale"), dict(C=100, epsilon=0.2, gamma="scale"),
           dict(C=10, epsilon=1.0, gamma="scale"), dict(C=10, epsilon=0.2, gamma=50)]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, cfg in zip(axes, configs):
    m = make_pipeline(StandardScaler(), SVR(kernel="rbf", **cfg)).fit(X_train, y_train)
    ax.scatter(X_train, y_train, s=10, alpha=0.5)
    ax.plot(X_plot, m.predict(X_plot), "r", lw=2)
    ax.set_title(str(cfg), fontsize=9)
plt.show()

# %% [markdown]
# ## 4. Hyperparameter tuning with GridSearchCV

# %%
pipe = make_pipeline(StandardScaler(), SVR())
param_grid = {
    "svr__kernel": ["rbf", "poly"],
    "svr__C": [0.1, 1, 10, 100],
    "svr__epsilon": [0.05, 0.1, 0.5],
    "svr__gamma": ["scale", 0.1, 1],
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)

best = grid.best_estimator_
pred = best.predict(X_test)
print(f"Test R2 = {r2_score(y_test, pred):.3f} | Test MSE = {mean_squared_error(y_test, pred):.3f}")
print("Number of support vectors:", len(best[-1].support_), "out of", len(X_train), "training points")

# %% [markdown]
# ## 5. Visualize the epsilon-tube of the tuned model

# %%
eps = best[-1].epsilon
line = best.predict(X_plot)
plt.scatter(X_train, y_train, s=12, alpha=0.5, label="train")
plt.plot(X_plot, line, "r", label="SVR fit")
plt.fill_between(X_plot.ravel(), line - eps, line + eps, color="r", alpha=0.15, label=f"epsilon tube ({eps})")
plt.legend(); plt.title("Tuned SVR")
plt.show()
