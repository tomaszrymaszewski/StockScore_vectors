"""Embedding families: node2vec, spectral (Laplacian Eigenmaps), netmf, and raw baselines.

Ported from train_vectors.EmbeddingTrainer and benchmark/embedding_methods.py.
All embedders emit the same `{ticker: vector}` dict.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh, svds

from .config import Config

logger = logging.getLogger(__name__)

# Embedders that ignore the graph and read the returns matrix directly.
RAW_EMBEDDERS = {"raw_corr", "raw_returns"}
# Embedders whose output does not depend on the seed.
DETERMINISTIC_EMBEDDERS = {"spectral", "netmf", "raw_corr", "raw_returns"}


def _weighted_adjacency(G: nx.Graph) -> tuple[list[str], np.ndarray]:
    """Symmetric, non-negative weighted adjacency in a fixed node order."""
    nodes = list(G.nodes())
    W = nx.to_numpy_array(G, nodelist=nodes, weight="weight", dtype=np.float64)
    W = np.abs(W)
    np.fill_diagonal(W, 0.0)
    W = np.maximum(W, W.T)
    return nodes, W


# ---------------------------------------------------------------------------
# Node2Vec
# ---------------------------------------------------------------------------


def node2vec_embeddings(G: nx.Graph, config: Config) -> dict[str, list[float]]:
    """Random-walk + skip-gram embedding (the library default)."""
    from node2vec import Node2Vec

    p, q = config.walk_p, config.walk_q
    logger.info("Training Node2Vec (p=%s, q=%s, dim=%d, seed=%d)...",
                p, q, config.dimensions, config.seed)

    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)
        logger.warning("Removed %d isolated nodes.", len(isolates))

    if not G.nodes():
        return {}

    # Seed every source of randomness. workers=1 is required for gensim
    # Word2Vec determinism (multi-worker training is non-deterministic).
    np.random.seed(config.seed)
    n2v = Node2Vec(
        G, dimensions=config.dimensions,
        walk_length=config.walk_length, num_walks=config.num_walks,
        p=p, q=q, workers=1, quiet=True, seed=config.seed,
    )
    model = n2v.fit(window=config.window, min_count=1, seed=config.seed)
    return {t: model.wv[t].tolist() for t in model.wv.index_to_key}


# ---------------------------------------------------------------------------
# Laplacian Eigenmaps (spectral embedding)
# ---------------------------------------------------------------------------


def laplacian_eigenmaps(G: nx.Graph, dim: int, seed: int = 0) -> dict[str, list[float]]:
    """Normalized-Laplacian eigenmap embedding (Belkin & Niyogi, 2003).

    A closed-form GLOBAL spectral decomposition -- no random walks, no
    stochastic training.
    """
    nodes, W = _weighted_adjacency(G)
    n = len(nodes)
    if n == 0:
        return {}
    dim = int(min(dim, n - 1))
    d = W.sum(axis=1)
    d[d <= 0] = 1e-12
    dinv_sqrt = 1.0 / np.sqrt(d)
    A = W * np.outer(dinv_sqrt, dinv_sqrt)
    A = 0.5 * (A + A.T)

    k = int(min(n - 1, dim + 5))
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n)
    try:
        vals, vecs = eigsh(A, k=k, which="LA", v0=v0)
    except Exception:
        vals, vecs = np.linalg.eigh(A)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    nontrivial = np.where(vals < 1.0 - 1e-6)[0]
    if nontrivial.size >= dim:
        cols = nontrivial[:dim]
    else:
        cols = np.arange(1, min(dim + 1, vecs.shape[1]))
    emb = dinv_sqrt[:, None] * vecs[:, cols]
    return {nodes[i]: emb[i].astype(float).tolist() for i in range(n)}


# ---------------------------------------------------------------------------
# NetMF -- DeepWalk/node2vec as explicit matrix factorization (Qiu et al., WSDM 2018)
# ---------------------------------------------------------------------------


def netmf(G: nx.Graph, dim: int, window: int = 10, seed: int = 0,
          b: float = 1.0) -> dict[str, list[float]]:
    """Closed-form skip-gram matrix factorization (small-window NetMF)."""
    nodes, W = _weighted_adjacency(G)
    n = len(nodes)
    if n == 0:
        return {}
    dim = int(min(dim, n - 1))
    d = W.sum(axis=1)
    d[d <= 0] = 1e-12
    vol = float(d.sum())
    P = W / d[:, None]
    T = max(1, int(window))

    S = np.zeros_like(P)
    Pr = np.eye(n)
    for _ in range(T):
        Pr = Pr @ P
        S += Pr
    M = (vol / (b * T)) * (S / d[None, :])
    M = np.log(np.maximum(M, 1.0))
    M = np.nan_to_num(M)

    k = int(min(dim, min(M.shape) - 1))
    if k < 1:
        return {nodes[i]: [0.0] for i in range(n)}
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(min(M.shape))
    try:
        U, s, _ = svds(M, k=k, v0=v0)
    except Exception:
        U, s, _ = np.linalg.svd(M, full_matrices=False)
        U, s = U[:, :k], s[:k]
    order = np.argsort(s)[::-1]
    emb = U[:, order] * np.sqrt(np.maximum(s[order], 0.0))[None, :]
    return {nodes[i]: emb[i].astype(float).tolist() for i in range(n)}


# ---------------------------------------------------------------------------
# Raw-feature baselines (no embedding)
# ---------------------------------------------------------------------------


def raw_correlation_vectors(returns: pd.DataFrame) -> dict[str, list[float]]:
    """Each stock IS its row of the Pearson correlation matrix (self-corr zeroed)."""
    C = np.nan_to_num(returns.corr().values).astype(np.float64)
    np.fill_diagonal(C, 0.0)
    tickers = list(returns.columns)
    return {t: C[i].tolist() for i, t in enumerate(tickers)}


def raw_return_vectors(returns: pd.DataFrame) -> dict[str, list[float]]:
    """Each stock IS its standardized return time series."""
    X = returns.values.T.astype(np.float64)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1e-12
    Xn = (X - mu) / sd
    tickers = list(returns.columns)
    return {t: Xn[i].tolist() for i, t in enumerate(tickers)}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def build_embeddings(config: Config, returns: pd.DataFrame,
                     graph: nx.Graph | None = None) -> dict[str, list[float]]:
    """Produce `{ticker: vector}` for `config.embedder`.

    Graph-based families share the exact graph the GraphBuilder produces for
    this config, so only the embedding step differs between families.
    """
    embedder = config.embedder

    if embedder == "raw_corr":
        return raw_correlation_vectors(returns)
    if embedder == "raw_returns":
        return raw_return_vectors(returns)

    if graph is None:
        from .graph import GraphBuilder
        graph = GraphBuilder(config).build(returns)

    if embedder == "node2vec":
        return node2vec_embeddings(graph, config)

    # node2vec drops isolates inside its trainer; do the same so all families
    # see one graph.
    isolates = list(nx.isolates(graph))
    if isolates:
        graph.remove_nodes_from(isolates)

    if embedder == "spectral":
        return laplacian_eigenmaps(graph, config.dimensions, seed=config.seed)
    if embedder == "netmf":
        return netmf(graph, config.dimensions, window=config.window,
                     seed=config.seed)
    raise ValueError(f"Unknown embedder {embedder!r}")
