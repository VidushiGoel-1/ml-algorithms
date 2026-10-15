# 06 - Decision Tree Classifier

**Files:** `decision_tree_classifier.py`, `decision_tree_classifier.ipynb`
**Dataset:** Breast Cancer Wisconsin (30 features, benign/malignant)

## What is it?
A flowchart-like model: each node asks a question about one feature, branches follow the answer, and each leaf outputs a class (majority class of samples in it; `predict_proba` gives class fractions).

## How splits are chosen
At each node the algorithm picks the feature/threshold that makes the child nodes as **pure** as possible.

| Criterion | Formula |
|-----------|---------|
| **Gini impurity** | `1 - Σ pi²` |
| **Entropy** | `-Σ pi log2(pi)` |
| **Information Gain** | `Entropy(parent) - Σ (nchild/n) Entropy(child)` |

Both criteria usually give similar trees; Gini is slightly faster. The code calculates a split's information gain by hand.

## Overfitting
Left unrestricted, the tree reaches 100% training accuracy but generalizes worse. Symptoms: large gap between train and test accuracy, very deep tree.

**Prevent with:**
- `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes` (pre-pruning)
- `ccp_alpha` (post-pruning)
- Cross-validation to select them
- Ensembles like Random Forest

## Confusion matrix & metrics
```
                 Predicted 0   Predicted 1
Actual 0            TN            FP
Actual 1            FN            TP
```
| Metric | Formula | Question it answers |
|--------|---------|---------------------|
| Accuracy | (TP+TN)/all | Overall how many correct? |
| Precision | TP/(TP+FP) | Of predicted positives, how many are right? |
| Recall | TP/(TP+FN) | Of actual positives, how many did we catch? |
| F1 | 2PR/(P+R) | Balance of both |

For medical tasks, **recall** of the dangerous class is often the priority.

## Pros / Cons
- Interpretable, no scaling, handles numeric + categorical (after encoding), gives feature importance
- Unstable and overfits easily, biased toward features with many split points, axis-aligned boundaries only

## What the code does
1. Manual Gini/entropy/information gain
2. Gini vs entropy comparison
3. Train/test accuracy vs depth plot (overfitting)
4. GridSearchCV pruning
5. Confusion matrix, classification report
6. Tree plot and feature importances
