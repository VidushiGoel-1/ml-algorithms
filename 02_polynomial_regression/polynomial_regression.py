# %% [markdown]
# # Polynomial Regression
# For non-linear data, we create polynomial features (x, x^2, x^3, ...) and fit a linear model on them.
# High degree -> overfitting. We fix it using Regularization (Ridge / Lasso) and cross-validation.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# %% [markdown]
# ## 1. Non-linear data: y = sin(2*pi*x) + noise

# %%
rng = np.random.RandomState(0)
X = np.sort(rng.rand(60, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + rng.normal(0, 0.25, 60)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_plot = np.linspace(0, 1, 300).reshape(-1, 1)

plt.scatter(X, y)
plt.title("Non-linear data")
plt.show()

# %% [markdown]
# ## 2. Underfit vs good fit vs overfit (degree 1, 4, 15)

# %%
def evaluate(model, name):
    model.fit(X_train, y_train)
    tr = mean_squared_error(y_train, model.predict(X_train))
    te = mean_squared_error(y_test, model.predict(X_test))
    print(f"{name:28s} train MSE={tr:.3f} | test MSE={te:.3f} | test R2={r2_score(y_test, model.predict(X_test)):.3f}")
    return model

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, d in zip(axes, [1, 4, 15]):
    m = evaluate(make_pipeline(PolynomialFeatures(d), LinearRegression()), f"Degree {d} (no regularization)")
    ax.scatter(X_train, y_train, label="train")
    ax.scatter(X_test, y_test, c="orange", label="test")
    ax.plot(X_plot, m.predict(X_plot), "r")
    ax.set_ylim(-2, 2); ax.set_title(f"Degree {d}"); ax.legend()
plt.show()

# %% [markdown]
# ## 3. Avoiding overfitting with regularization (degree 15 stays, but coefficients are shrunk)
# Scaling is important before regularization.

# %%
deg = 15
models = {
    "No regularization": make_pipeline(PolynomialFeatures(deg), StandardScaler(), LinearRegression()),
    "Ridge (L2) alpha=1": make_pipeline(PolynomialFeatures(deg), StandardScaler(), Ridge(alpha=1)),
    "Lasso (L1) alpha=0.01": make_pipeline(PolynomialFeatures(deg), StandardScaler(), Lasso(alpha=0.01, max_iter=100000)),
}
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, c="gray", alpha=0.6)
for name, m in models.items():
    evaluate(m, name)
    plt.plot(X_plot, m.predict(X_plot), label=name)
plt.ylim(-2, 2); plt.legend(); plt.title("Degree 15: effect of regularization")
plt.show()

lasso = models["Lasso (L1) alpha=0.01"][-1]
print("Lasso: non-zero coefficients =", int((lasso.coef_ != 0).sum()), "out of", len(lasso.coef_))

# %% [markdown]
# ## 4. Choosing the degree: validation curve using cross-validation

# %%
degrees = range(1, 16)
cv_mse = [-cross_val_score(make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression()),
                           X_train, y_train, cv=5, scoring="neg_mean_squared_error").mean() for d in degrees]
tr_mse = [mean_squared_error(y_train, make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression())
                             .fit(X_train, y_train).predict(X_train)) for d in degrees]
plt.plot(degrees, tr_mse, "o-", label="train MSE")
plt.plot(degrees, cv_mse, "o-", label="CV MSE")
plt.yscale("log"); plt.xlabel("degree"); plt.legend(); plt.title("Bias-variance: pick the degree where CV error is lowest")
plt.show()
print("Best degree by CV:", list(degrees)[int(np.argmin(cv_mse))])

# %% [markdown]
# ## 5. Tune degree + alpha together with GridSearchCV

# %%
pipe = make_pipeline(PolynomialFeatures(), StandardScaler(), Ridge())
grid = GridSearchCV(pipe,
                    {"polynomialfeatures__degree": [2, 3, 4, 6, 8, 10, 15],
                     "ridge__alpha": [0.001, 0.01, 0.1, 1, 10]},
                    cv=5, scoring="neg_mean_squared_error")
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Test MSE   :", round(mean_squared_error(y_test, grid.predict(X_test)), 4))
print("Test R2    :", round(r2_score(y_test, grid.predict(X_test)), 4))
