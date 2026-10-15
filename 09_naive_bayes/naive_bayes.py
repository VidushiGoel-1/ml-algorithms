# %% [markdown]
# # Naive Bayes
# Classifier based on Bayes' theorem + "naive" assumption that features are conditionally independent given the class.
#   P(class | x1..xn)  is proportional to  P(class) * Product_i P(xi | class)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# %% [markdown]
# ## 1. Bayes theorem by hand (tiny example)
# P(Disease)=1%, test sensitivity P(+|D)=95%, false-positive rate P(+|not D)=5%.

# %%
p_d, p_pos_d, p_pos_nd = 0.01, 0.95, 0.05
p_pos = p_pos_d * p_d + p_pos_nd * (1 - p_d)
print("P(Disease | +ve test) =", round(p_pos_d * p_d / p_pos, 4), " <- posterior is only ~16% because the prior is tiny")

# %% [markdown]
# ## 2. Gaussian Naive Bayes from scratch (continuous features, Iris)

# %%
X, y = load_iris(return_X_y=True)
names = load_iris().target_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

class MyGaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.prior = np.array([(y == c).mean() for c in self.classes])
        self.mean = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.var = np.array([X[y == c].var(axis=0) + 1e-9 for c in self.classes])
        return self

    def _log_posterior(self, x):
        # log P(c) + sum_i log N(x_i | mean_ci, var_ci)   (logs avoid numerical underflow)
        log_lik = -0.5 * (np.log(2 * np.pi * self.var) + (x - self.mean) ** 2 / self.var).sum(axis=1)
        return np.log(self.prior) + log_lik

    def predict(self, X):
        return self.classes[[np.argmax(self._log_posterior(x)) for x in X]]

mine = MyGaussianNB().fit(X_train, y_train)
sk = GaussianNB().fit(X_train, y_train)
print("From scratch accuracy:", round(accuracy_score(y_test, mine.predict(X_test)), 4))
print("sklearn accuracy     :", round(accuracy_score(y_test, sk.predict(X_test)), 4))
print("Learned class priors :", mine.prior.round(3))
print("Class means:\n", mine.mean.round(2))

# %%
y_pred = sk.predict(X_test)
print(classification_report(y_test, y_pred, target_names=names))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=names)
plt.show()
print("Posterior probabilities for first test flower:", sk.predict_proba(X_test[:1]).round(3))

# %% [markdown]
# ## 3. Multinomial Naive Bayes for text (spam detection) with Laplace smoothing

# %%
texts = ["win free money now", "free lottery winner claim prize", "limited offer win cash", "urgent claim your free reward",
         "cheap loans approved instantly", "meeting at 10 tomorrow", "please review the project report",
         "lunch with the team today", "assignment submission deadline tomorrow", "can we reschedule the class",
         "notes for the ML lecture attached", "project meeting agenda attached"]
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # 1 = spam, 0 = ham

vec = CountVectorizer()
Xt = vec.fit_transform(texts)
for alpha in [1.0, 0.0001]:   # alpha=1 -> Laplace smoothing
    nb = MultinomialNB(alpha=alpha).fit(Xt, labels)
    msg = ["claim your free prize now", "project report meeting tomorrow", "free meeting"]
    probs = nb.predict_proba(vec.transform(msg))[:, 1]
    print(f"alpha={alpha}:", {m: round(float(p), 3) for m, p in zip(msg, probs)})

# %% [markdown]
# ## 4. Why smoothing is needed: a word never seen in spam would give P=0 and kill the whole product

# %%
nb = MultinomialNB(alpha=1).fit(Xt, labels)
top = np.argsort(nb.feature_log_prob_[1] - nb.feature_log_prob_[0])[-5:]
print("Words most indicative of spam:", vec.get_feature_names_out()[top])
