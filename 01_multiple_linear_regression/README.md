# 01 - Multiple Linear Regression

**Files:** `multiple_linear_regression.py`, `multiple_linear_regression.ipynb`
**Dataset:** Diabetes (10 features -> disease progression score)

## What is it?
Linear regression with **more than one input feature**. It predicts a continuous output `y` as a weighted sum of the features:

```
y = b0 + b1*x1 + b2*x2 + ... + bn*xn + error
```

- `b0` = intercept (prediction when all features are 0)
- `bi` = **coefficient**: change in `y` when `xi` increases by 1 unit, **keeping all other features fixed**

## How does it learn?
It finds coefficients that minimize the **sum of squared errors** (Ordinary Least Squares):

```
Loss = Σ (y_actual - y_predicted)²
Closed form:  β = (XᵀX)⁻¹ Xᵀ y
```

## Evaluation metrics
| Metric | Formula | Meaning |
|--------|---------|---------|
| **MSE** | mean((y - ŷ)²) | Average squared error; punishes big errors; units are y² |
| **RMSE** | √MSE | Same units as y, easier to interpret |
| **MAE** | mean(\|y - ŷ\|) | Average absolute error |
| **R²** | 1 - SS_res / SS_tot | Fraction of variance explained (1 = perfect, 0 = same as predicting the mean, can be negative) |
| **Adjusted R²** | 1 - (1-R²)(n-1)/(n-p-1) | R² penalized for adding useless features |

R² always increases when you add features, so use **Adjusted R²** to compare models with different feature counts.

## Interpreting coefficients
- Sign: positive -> feature increases output; negative -> decreases.
- Magnitude is only comparable if features are **standardized** (the code shows this).
- Large coefficients on correlated features (e.g. `s1`, `s2`) are a sign of **multicollinearity**.

## Assumptions (check these!)
1. **Linearity** between features and target
2. **Independence** of errors
3. **Homoscedasticity**: constant variance of residuals (residual plot should look like random noise)
4. **Normality** of residuals
5. **No multicollinearity** between features

## Pros / Cons
- Simple, fast, interpretable
- Sensitive to outliers, can't capture non-linearity, breaks with highly correlated features
- Fix: Ridge/Lasso (see 02), polynomial features, feature selection

## What the code does
1. Loads data and splits train/test
2. Fits `LinearRegression`, prints intercept and coefficients
3. Computes R², Adjusted R², MSE, RMSE, MAE (and compares train vs test R²)
4. Predicts the output for a new sample
5. Standardized coefficient bar chart
6. Predicted-vs-actual and residual plots
7. Multicollinearity check
