# %% [markdown]
# # K-Nearest Neighbors (KNN)
# Lazy, instance-based learner: prediction depends on the K closest training points.
# Classification -> majority vote, Regression -> average of neighbors' values.
# Important: feature scaling, distance metric, value of K.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_diabetes
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, r2_score, mean_squared_error

# %%
data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, stratify=data.target, random_state=42)

# %% [markdown]
# ## 1. KNN from scratch (Euclidean distance + majority vote)

# %%
def knn_predict(X_tr, y_tr, x, k=5):
    dist = np.sqrt(((X_tr - x) ** 2).sum(axis=1))
    nearest = np.argsort(dist)[:k]
    return np.bincount(y_tr[nearest]).argmax()

sc = StandardScaler().fit(X_train)
Xs_tr, Xs_te = sc.transform(X_train), sc.transform(X_test)
manual = np.array([knn_predict(Xs_tr, y_train, x, k=5) for x in Xs_te])
print("Scratch KNN accuracy :", round((manual == y_test).mean(), 4))
print("sklearn KNN accuracy :", round(KNeighborsClassifier(5).fit(Xs_tr, y_train).score(Xs_te, y_test), 4))

# %% [markdown]
# ## 2. Scaling matters: KNN is distance based

# %%
raw = KNeighborsClassifier(5).fit(X_train, y_train).score(X_test, y_test)
scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(X_train, y_train).score(X_test, y_test)
print(f"Without scaling: {raw:.3f} | With scaling: {scaled:.3f}")

# %% [markdown]
# ## 3. Distance metrics

# %%
for metric in ["euclidean", "manhattan", "chebyshev", "cosine"]:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(5, metric=metric, algorithm="brute"))
    print(f"{metric:10s} CV acc = {cross_val_score(m, X_train, y_train, cv=5).mean():.4f}")
for p in [1, 2, 3]:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(5, metric="minkowski", p=p))
    print(f"minkowski p={p}: CV acc = {cross_val_score(m, X_train, y_train, cv=5).mean():.4f}")

# %% [markdown]
# ## 4. Tuning K: small K -> overfit (high variance), large K -> underfit (high bias)

# %%
ks = range(1, 31)
cv_acc = [cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(k)), X_train, y_train, cv=5).mean() for k in ks]
train_acc = [make_pipeline(StandardScaler(), KNeighborsClassifier(k)).fit(X_train, y_train).score(X_train, y_train) for k in ks]
plt.plot(ks, train_acc, label="train"); plt.plot(ks, cv_acc, label="cross-val")
plt.xlabel("K"); plt.ylabel("accuracy"); plt.legend(); plt.title("Choosing K")
plt.show()
print("Best K:", list(ks)[int(np.argmax(cv_acc))])

# %% [markdown]
# ## 5. GridSearchCV (K, weights, metric) and final evaluation

# %%
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
grid = GridSearchCV(pipe, {"kneighborsclassifier__n_neighbors": range(1, 21),
                           "kneighborsclassifier__weights": ["uniform", "distance"],
                           "kneighborsclassifier__metric": ["euclidean", "manhattan"]}, cv=5).fit(X_train, y_train)
print("Best:", grid.best_params_)
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred, target_names=data.target_names))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=data.target_names)
plt.show()

# %% [markdown]
# ## 6. KNN Regression (Diabetes) - average of K neighbors

# %%
Xd, yd = load_diabetes(return_X_y=True)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.2, random_state=0)
for k in [1, 5, 10, 20, 40]:
    r = make_pipeline(StandardScaler(), KNeighborsRegressor(k)).fit(Xd_tr, yd_tr)
    print(f"K={k:2d} -> test R2={r2_score(yd_te, r.predict(Xd_te)):.3f}, MSE={mean_squared_error(yd_te, r.predict(Xd_te)):.1f}")
