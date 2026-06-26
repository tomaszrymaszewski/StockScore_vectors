"""
Supervised sector-classification metric: SVM + k-fold CV on the embeddings.

This is the protocol Dolphin et al. (2024, arXiv:2407.18645) use for their sector-
classification leaderboard (Table 1), where they also report a best-effort reimplementation
of Sarmah et al. (node2vec on a correlation graph) at F1 0.61 / 61% accuracy. It is a THIRD,
distinct lens on embedding quality:

  * hit-rate / MRR        -> unsupervised retrieval (nearest neighbors share the label?)
  * V-measure / NMI / ARI -> unsupervised clustering agreement (KMeans vs labels)
  * this module           -> SUPERVISED: can a classifier read the label off the embedding?

It lets the paper place a number next to that shared leaderboard. Note the comparison is only
strictly apples-to-apples on Dolphin's public 611-stock 2000-2018 dataset; on other universes
it is a metric-aligned regime check, not a head-to-head.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC


def svm_classification(
    embeddings: dict[str, list[float]],
    labels: dict[str, str],
    *,
    seed: int = 0,
    n_splits: int = 5,
    min_per_class: int | None = None,
    C: float = 10.0,
    kernel: str = "rbf",
) -> dict[str, float]:
    """Weighted-F1 and accuracy of an SVM classifying GICS labels from the embedding.

    Stratified k-fold CV (default 5, matching Dolphin). Classes with fewer than `n_splits`
    members are dropped (stratified CV needs at least one per fold); the count of dropped
    stocks is returned so the loss of coverage at fine granularity is visible rather than
    silent. Embeddings are L2-normalized first (cosine geometry).
    """
    tickers = [t for t in embeddings if t in labels]
    if len(tickers) < n_splits + 1:
        return {}
    X = np.asarray([embeddings[t] for t in tickers], dtype=np.float64)
    y = np.asarray([labels[t] for t in tickers], dtype=object)

    floor = max(min_per_class or n_splits, n_splits)
    counts = Counter(y)
    keep = np.array([counts[v] >= floor for v in y])
    n_dropped = int((~keep).sum())
    X, y = X[keep], y[keep]
    if len(set(y.tolist())) < 2:
        return {}

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    X = X / norms

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = SVC(kernel=kernel, C=C, gamma="scale")
    pred = cross_val_predict(clf, X, y, cv=cv)
    return {
        "n_stocks": int(len(y)),
        "n_classes": int(len(set(y.tolist()))),
        "n_dropped": n_dropped,
        "f1_weighted": float(f1_score(y, pred, average="weighted")),
        "accuracy": float(accuracy_score(y, pred)),
    }
