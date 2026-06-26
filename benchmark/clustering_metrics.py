"""
Clustering-agreement metrics: V-measure, NMI, ARI vs GICS labels.

V-measure / NMI / ARI on a KMeans clustering of the embedding vs GICS labels are the
standard way prior correlation-network embedding studies report cluster quality. Computing
them here lets the paper reference that literature on common terms, alongside the retrieval
hit-rate that is our preferred primary metric.

Methodological caveat the paper should state: **V-measure (and homogeneity) inflate as the
number of clusters grows** -- at sub-industry granularity there are only ~3 stocks per
class, so homogeneity is trivially high. That is precisely why a retrieval hit-rate is the
more honest headline. `clustering_metrics` reports homogeneity and completeness separately
so this inflation is visible rather than hidden inside V-measure.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)


def _aligned(embeddings: dict[str, list[float]], labels: dict[str, str]):
    tickers = [t for t in embeddings if t in labels]
    X = np.asarray([embeddings[t] for t in tickers], dtype=np.float64)
    y = np.asarray([labels[t] for t in tickers], dtype=object)
    return tickers, X, y


def cluster_kmeans(X: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    # Cosine geometry: L2-normalize so Euclidean KMeans approximates spherical k-means.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    Xn = X / norms
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return km.fit_predict(Xn)


def clustering_metrics(
    embeddings: dict[str, list[float]],
    labels: dict[str, str],
    *,
    n_clusters: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Cluster the embedding and score agreement with GICS labels.

    `n_clusters` defaults to the number of distinct GICS classes at this level (the
    natural, literature-standard choice).
    """
    tickers, X, y = _aligned(embeddings, labels)
    if len(tickers) < 2:
        return {}
    classes = sorted(set(y.tolist()))
    if n_clusters is None:
        n_clusters = len(classes)
    n_clusters = max(2, min(n_clusters, len(tickers)))

    pred = cluster_kmeans(X, n_clusters, seed=seed)
    return {
        "n_stocks": len(tickers),
        "n_gics_classes": len(classes),
        "n_clusters": n_clusters,
        "v_measure": float(v_measure_score(y, pred)),
        "nmi": float(normalized_mutual_info_score(y, pred)),
        "ari": float(adjusted_rand_score(y, pred)),
        "homogeneity": float(homogeneity_score(y, pred)),
        "completeness": float(completeness_score(y, pred)),
    }


def clustering_metrics_all_levels(
    embeddings: dict[str, list[float]],
    labels_by_level: dict[str, dict[str, str]],
    *,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    return {
        level: clustering_metrics(embeddings, labels, seed=seed)
        for level, labels in labels_by_level.items()
    }
