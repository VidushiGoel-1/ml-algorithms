# %% [markdown]
# # Multiple Linear Regression
# Multiple features (X1..Xn) -> one continuous output (y).
# We evaluate with R^2, MSE, RMSE and interpret the coefficients.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# %% [markdown]
# ## 1. Load data (Diabetes dataset: 10 features -> disease progression score)

# %%
data = load_diabetes(as_frame=True)
X, y = data.data, data.target
print(X.shape)
X.head()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ## 2. Train the model  (y = b0 + b1*x1 + ... + bn*xn)

# %%
model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept (b0):", round(model.intercept_, 3))
coef_df = pd.DataFrame({"feature": X.columns, "coefficient": model.coef_}).sort_values("coefficient")
print(coef_df.to_string(index=False))

# %% [markdown]
# ## 3. Evaluate: R^2, Adjusted R^2, MSE, RMSE, MAE

# %%
y_pred = model.predict(X_test)
n, p = X_test.shape
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
mse = mean_squared_error(y_test, y_pred)

print(f"R^2          : {r2:.4f}")
print(f"Adjusted R^2 : {adj_r2:.4f}")
print(f"MSE          : {mse:.2f}")
print(f"RMSE         : {np.sqrt(mse):.2f}")
print(f"MAE          : {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Train R^2    : {model.score(X_train, y_train):.4f}  (compare with test R^2 to detect overfitting)")

# %% [markdown]
# ## 4. Predict output for a new sample

# %%
new_sample = X_test.iloc[[0]]
print("Predicted:", model.predict(new_sample)[0].round(2), "| Actual:", y_test.iloc[0])

# %% [markdown]
# ## 5. Which feature matters most? Compare coefficients on standardized features
# Raw coefficients are not comparable when features have different scales.

# %%
scaler = StandardScaler().fit(X_train)
m_std = LinearRegression().fit(scaler.transform(X_train), y_train)
std_coef = pd.Series(m_std.coef_, index=X.columns).sort_values()
std_coef.plot(kind="barh", title="Standardized coefficients (impact of 1 std change)")
plt.show()

# %% [markdown]
# ## 6. Diagnostics: predicted vs actual, residuals

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(y_test, y_pred, alpha=0.7)
ax[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--")
ax[0].set(xlabel="Actual", ylabel="Predicted", title="Predicted vs Actual")
residuals = y_test - y_pred
ax[1].scatter(y_pred, residuals, alpha=0.7)
ax[1].axhline(0, color="r", ls="--")
ax[1].set(xlabel="Predicted", ylabel="Residual", title="Residual plot (should look random)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Multicollinearity check (correlation between features)

# %%
corr = X.corr()
print(corr.abs().where(~np.eye(len(corr), dtype=bool)).max().sort_values(ascending=False).head(4))
