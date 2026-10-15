# 05 - Decision Tree Regressor

**Files:** `decision_tree_regressor.py`, `decision_tree_regressor.ipynb`
**Dataset:** synthetic noisy sine wave (1 feature, so the tree is easy to visualize)

## What is it?
A tree of if/else questions. Each internal node tests `feature <= threshold`, each leaf stores a prediction (the **mean** of the training targets that reach it). The result is a **piecewise-constant** function.

## Recursive splitting (greedy, top-down)
1. Start with all data at the root.
2. Try every feature and every possible threshold.
3. Pick the split that gives the **largest impurity reduction**.
4. Repeat recursively on left and right children until a stopping rule is hit.

For regression, impurity = **MSE / variance** of the node:
```
score = (n_left/n) * MSE(left) + (n_right/n) * MSE(right)     -> choose split with the lowest score
```
The code implements this search by hand and checks it against scikit-learn.

## Entropy (and Gini)
Used by **classification** trees (see 06) but important to know:
```
Entropy = -Σ pi log2(pi)     (0 = pure node, 1 = max impurity for 2 classes)
Gini    = 1 - Σ pi²
Information Gain = Entropy(parent) - weighted Entropy(children)
```
Regression trees use variance reduction instead because the target is continuous.

## Overfitting and pruning
An unrestricted tree memorizes the data (one leaf per sample). Control it by **pruning**:

**Pre-pruning (early stopping)**
- `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`

**Post-pruning (cost-complexity pruning)**
- Grow the full tree, then remove branches whose benefit is small.
- Minimizes `R(T) + α·|leaves|`; larger `ccp_alpha` -> smaller tree.
- `cost_complexity_pruning_path` gives candidate alphas; pick the one with best validation score (done in the code).

## Pros / Cons
- Easy to interpret/visualize, no scaling needed, handles non-linearity
- High variance (small data change -> different tree), can't extrapolate beyond training range, piecewise-flat predictions
- Fix: ensembles (Random Forest, see 07)

## What the code does
1. Entropy/variance demo
2. Manual best-split search vs sklearn
3. Effect of `max_depth`
4. Post-pruning with `ccp_alpha`
5. GridSearchCV tuning, MSE and R²
6. Tree plot and text rules
