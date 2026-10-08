# 02 - Polynomial Regression

**Files:** `polynomial_regression.py`, `polynomial_regression.ipynb`
**Dataset:** synthetic noisy sine curve (clearly non-linear)

## What is it?
When data is **non-linear**, a straight line underfits. Polynomial regression adds powers of the feature:

```
y = b0 + b1*x + b2*x² + b3*x³ + ... + bd*x^d
```

It is still a **linear model** (linear in the coefficients). We just transform `x -> [x, x², ..., x^d]` using `PolynomialFeatures` and then run ordinary linear regression. With several features it also creates interaction terms (`x1*x2`).

## Degree and the bias-variance trade-off
| Degree | Behaviour | Problem |
|--------|-----------|---------|
| Too low (1) | Misses the curve | **Underfitting** (high bias) |
| Just right (3-5 here) | Follows the pattern, ignores noise | Good generalization |
| Too high (15) | Wiggles through every training point | **Overfitting** (high variance): train error tiny, test error huge |

## How to avoid overfitting
1. **Choose the degree with cross-validation** (validation curve in the code).
2. **Regularization**: add a penalty on coefficient size so high-degree terms are shrunk.
   - **Ridge (L2):** `Loss = MSE + α Σ bi²` -> shrinks all coefficients smoothly, never exactly zero.
   - **Lasso (L1):** `Loss = MSE + α Σ |bi|` -> can set coefficients to exactly 0 (automatic feature selection).
   - **ElasticNet:** mix of both.
   - Larger `α` = stronger penalty = simpler model. `α = 0` is plain linear regression.
3. **More data** or **less noisy data**.
4. **Scale features** (StandardScaler) before regularizing, since high powers have huge values.

## Practical notes
- Always use a `Pipeline` (PolynomialFeatures -> StandardScaler -> Model) so that scaling happens inside cross-validation (no data leakage).
- Never judge by training error; compare **train vs test/CV MSE**.
- Extrapolation outside the training range is dangerous for polynomials.

## What the code does
1. Generates non-linear data
2. Shows underfit (d=1), good fit (d=4), overfit (d=15)
3. Fixes degree-15 with Ridge and Lasso
4. Validation curve over degrees 1-15
5. `GridSearchCV` over degree **and** alpha
