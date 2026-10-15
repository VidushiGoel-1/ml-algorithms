# 09 - Naive Bayes

**Files:** `naive_bayes.py`, `naive_bayes.ipynb`
**Datasets:** Iris (continuous features), small spam/ham text set

## What is it?
A probabilistic classifier built on **Bayes' theorem**:

```
P(C | x) = P(x | C) · P(C) / P(x)
posterior = likelihood × prior / evidence
```

- **Prior P(C):** how common the class is
- **Likelihood P(x | C):** how likely these feature values are if the class is C
- **Posterior P(C | x):** what we want; predict the class with the highest posterior

## Why "naive"?
Computing `P(x1, ..., xn | C)` jointly is hard, so we **assume features are conditionally independent given the class**:

```
P(x1..xn | C) = P(x1|C) · P(x2|C) · ... · P(xn|C)
Prediction = argmax_C  P(C) · Π P(xi | C)
```
`P(x)` is the same for every class, so it is ignored when comparing. The assumption is rarely true, yet the classifier often works surprisingly well, especially for text.

## Conditional probability refresher
`P(A|B)` = probability of A given B happened = `P(A ∩ B) / P(B)`. The code includes a medical-test example showing why a positive test on a rare disease still gives a low posterior (base-rate effect).

## Variants
| Variant | Features | Likelihood model |
|---------|----------|------------------|
| **GaussianNB** | continuous | Normal distribution with per-class mean/variance |
| **MultinomialNB** | counts (word counts) | Multinomial; classic for text |
| **BernoulliNB** | binary (word present/absent) | Bernoulli |
| **ComplementNB** | imbalanced text | Variant of Multinomial |

## Laplace (additive) smoothing
If a word never appears with a class in training, `P(word|class) = 0` and the whole product becomes 0. Fix: add `alpha` (usually 1) to every count:
```
P(w|C) = (count(w, C) + α) / (total words in C + α·|V|)
```

## Numerical tip
Multiplying many small probabilities underflows, so implementations sum **log-probabilities** (the from-scratch Gaussian NB does this).

## Pros / Cons
- Very fast, works with little data, handles many features, gives probabilities, great text baseline
- Independence assumption is unrealistic, probabilities are often poorly calibrated, GaussianNB assumes normal features

## What the code does
1. Bayes theorem calculation by hand
2. Gaussian Naive Bayes **from scratch** vs sklearn on Iris (priors, means, accuracy, confusion matrix, posteriors)
3. Multinomial NB spam classifier with CountVectorizer
4. Effect of Laplace smoothing and most spam-indicative words
