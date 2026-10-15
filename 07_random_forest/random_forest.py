# %% [markdown]
# # Random Forest
# Many decision trees trained on bootstrap samples (bagging) with random feature subsets.
# Predictions are combined by voting (classification) / averaging (regression).
# Result: higher accuracy and much lower variance (overfitting) than a single tree.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, r2_score

# %%
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, stratify=data.target, random_state=42)

# %% [markdown]
# ## 1. Single tree vs Bagging vs Random Forest (5-fold CV on training data)

# %%
models = {
    "Single Decision Tree": DecisionTreeClassifier(random_state=0),
    "Bagging (100 trees)": BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=0),
    "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, random_state=0),
}
for name, m in models.items():
    scores = cross_val_score(m, X_train, y_train, cv=5)
    m.fit(X_train, y_train)
    print(f"{name:27s} CV acc={scores.mean():.4f} (+/- {scores.std():.4f}) | train={m.score(X_train, y_train):.3f} | test={m.score(X_test, y_test):.3f}")

# %% [markdown]
# ## 2. Bagging by hand: bootstrap samples + majority vote

# %%
rng = np.random.RandomState(0)
trees, n = [], len(X_train)
for _ in range(25):
    idx = rng.choice(n, n, replace=True)                       # bootstrap sample (with replacement)
    trees.append(DecisionTreeClassifier(max_features="sqrt", random_state=rng.randint(1e6)).fit(X_train[idx], y_train[idx]))
votes = np.array([t.predict(X_test) for t in trees])
manual_pred = (votes.mean(axis=0) >= 0.5).astype(int)           # majority vote
print("Manual forest accuracy (25 trees):", round(accuracy_score(y_test, manual_pred), 4))
print("Average single-tree accuracy     :", round(np.mean([accuracy_score(y_test, v) for v in votes]), 4))

# %% [markdown]
# ## 3. More trees -> better and more stable (with OOB score)

# %%
n_trees = [1, 5, 10, 25, 50, 100, 200, 300]
test_acc, oob = [], []
for k in n_trees:
    rf = RandomForestClassifier(n_estimators=k, oob_score=True, bootstrap=True, random_state=0, n_jobs=-1).fit(X_train, y_train)
    test_acc.append(rf.score(X_test, y_test))
    oob.append(rf.oob_score_ if k >= 10 else np.nan)   # OOB needs enough trees
plt.plot(n_trees, test_acc, "o-", label="test accuracy")
plt.plot(n_trees, oob, "s--", label="OOB score (free validation)")
plt.xlabel("n_estimators"); plt.legend(); plt.show()

# %% [markdown]
# ## 4. Hyperparameter tuning (reduces overfitting & increases accuracy)

# %%
param_dist = {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10, 15],
              "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2", 0.5]}
search = RandomizedSearchCV(RandomForestClassifier(random_state=0, n_jobs=-1), param_dist,
                            n_iter=15, cv=5, scoring="accuracy", random_state=0).fit(X_train, y_train)
best = search.best_estimator_
print("Best params:", search.best_params_)
y_pred = best.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, target_names=data.target_names))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=data.target_names, cmap="Greens")
plt.show()

# %% [markdown]
# ## 5. Feature importance

# %%
imp = best.feature_importances_
idx = np.argsort(imp)[-10:]
plt.barh(data.feature_names[idx], imp[idx]); plt.title("Random Forest: top 10 features")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. Random Forest as a Regressor (Diabetes data)

# %%
Xd, yd = load_diabetes(return_X_y=True)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.2, random_state=0)
rfr = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(Xd_tr, yd_tr)
print("RF regressor test R2:", round(r2_score(yd_te, rfr.predict(Xd_te)), 4))
