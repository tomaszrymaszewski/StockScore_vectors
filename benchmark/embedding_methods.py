"""
Embedding families beyond Node2Vec -- so the paper's "does an embedding beat raw features?"
question is asked of more than one method.

Node2Vec is one point in a large design space (random-walk sampling + skip-gram). If only
Node2Vec is tested, "embeddings don't beat raw features" is really "Node2Vec doesn't". This
module adds three structurally different representations that all emit the SAME
`{ticker: vector}` dict the rest of the harness consumes, so hit-rate / SVM-F1 / V-measure /
significance all score them unchanged:

  * **node2vec**  (delegated to EmbeddingTrainer)  -- local random walks + skip-gram.
  * **spectral**  Laplacian Eigenmaps             -- GLOBAL spectral geometry of the graph.
  * **netmf**     DeepWalk-as-matrix-factorization -- the closed-form skip-gram matrix
                  (Qiu et al. 2018, WSDM), so "is Node2Vec's value the walks, or the objective?"
  * **raw_corr / raw_returns**  the no-embedding baselines -- a stock IS its correlation row
                  (resp. its standardized return series). This is the honest yardstick: if no
                  embedding beats these, the embedding is a lossy compression. Makes the
                  previously prose-only "raw features ~0.77" number reproducible.

spectral and netmf are deterministic given the graph (the graph is seed-independent; only
Node2Vec injects walk randomness), so their seed-variance is ~0 -- a feature, not a bug. The
raw-feature baselines are likewise seed-free.

Everything routes through `build_embeddings(cfg, returns, seed)`, dispatched on `cfg.embedder`.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd
from scipy.sparse.linalg import eigsh, svds

# Embedders that ignore the graph and read the returns matrix directly.
_RAW_EMBEDDERS = {"raw_corr", "raw_returns"}
# Graph-free embedders (no GraphBuilder graph needed): raw baselines + Dolphin contrastive.
_GRAPH_FREE = _RAW_EMBEDDERS | {"contrastive"}
# Embedders whose output does not depend on the seed (compute once, reuse across seeds).
# (contrastive has a random init + minibatch shuffling, so it IS seed-dependent.)
_DETERMINISTIC = {"spectral", "netmf", "raw_corr", "raw_returns"}


def is_graph_free(embedder: str) -> bool:
    return embedder in _GRAPH_FREE


def is_deterministic(embedder: str) -> bool:
    return embedder in _DETERMINISTIC


def _weighted_adjacency(G: nx.Graph) -> tuple[list[str], np.ndarray]:
    """Symmetric, non-negative weighted adjacency in a fixed node order."""
    nodes = list(G.nodes())
    W = nx.to_numpy_array(G, nodelist=nodes, weight="weight", dtype=np.float64)
    W = np.abs(W)                      # weights are |corr|-style; guard any sign
    np.fill_diagonal(W, 0.0)
    W = np.maximum(W, W.T)             # enforce symmetry
    return nodes, W


# ---------------------------------------------------------------------------
# Laplacian Eigenmaps (spectral embedding)
# ---------------------------------------------------------------------------


def laplacian_eigenmaps(G: nx.Graph, dim: int, seed: int = 0) -> dict[str, list[float]]:
    """Normalized-Laplacian eigenmap embedding (Belkin & Niyogi, 2003).

    Embed each node by the bottom non-trivial eigenvectors of the symmetric normalized
    Laplacian L = I - D^{-1/2} W D^{-1/2} -- equivalently the TOP eigenvectors of the
    normalized adjacency A = D^{-1/2} W D^{-1/2}. The single largest mode (eigenvalue ~1,
    the sqrt(degree) vector) and any per-component constant modes are dropped; the next
    `dim` eigenvectors, mapped back by D^{-1/2}, are the coordinates.

    A genuinely different family from Node2Vec: a closed-form GLOBAL spectral decomposition,
    no random walks, no stochastic training.
    """
    nodes, W = _weighted_adjacency(G)
    n = len(nodes)
    if n == 0:
        return {}
    dim = int(min(dim, n - 1))
    d = W.sum(axis=1)
    d[d <= 0] = 1e-12
    dinv_sqrt = 1.0 / np.sqrt(d)
    A = W * np.outer(dinv_sqrt, dinv_sqrt)     # symmetric normalized adjacency
    A = 0.5 * (A + A.T)                          # symmetrize against round-off

    k = int(min(n - 1, dim + 5))                # a few extra to discard trivial modes
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n)
    try:
        vals, vecs = eigsh(A, k=k, which="LA", v0=v0)
    except Exception:
        vals, vecs = np.linalg.eigh(A)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # Drop trivial near-1 modes (global + per-connected-component constants), then take `dim`.
    nontrivial = np.where(vals < 1.0 - 1e-6)[0]
    if nontrivial.size >= dim:
        cols = nontrivial[:dim]
    else:                                        # fall back: skip only the single top mode
        cols = np.arange(1, min(dim + 1, vecs.shape[1]))
    emb = dinv_sqrt[:, None] * vecs[:, cols]     # generalized-eigvec coordinates
    return {nodes[i]: emb[i].astype(float).tolist() for i in range(n)}


# ---------------------------------------------------------------------------
# NetMF -- DeepWalk/node2vec as explicit matrix factorization (Qiu et al., WSDM 2018)
# ---------------------------------------------------------------------------


def netmf(G: nx.Graph, dim: int, window: int = 10, seed: int = 0, b: float = 1.0
          ) -> dict[str, list[float]]:
    """Closed-form skip-gram matrix factorization (small-window NetMF).

    DeepWalk/node2vec implicitly factorize
        M = (vol/(b*T)) * (sum_{r=1..T} P^r) D^{-1},   P = D^{-1} W,
    with T = window, b = #negative samples. We factorize log(max(M,1)) by truncated SVD:
    embedding = U sqrt(S). This isolates the skip-gram OBJECTIVE from the random-walk SAMPLING
    -- if NetMF tracks Node2Vec, the walks are not where the value is.
    """
    nodes, W = _weighted_adjacency(G)
    n = len(nodes)
    if n == 0:
        return {}
    dim = int(min(dim, n - 1))
    d = W.sum(axis=1)
    d[d <= 0] = 1e-12
    vol = float(d.sum())
    P = W / d[:, None]                           # row-stochastic transition matrix
    T = max(1, int(window))

    S = np.zeros_like(P)
    Pr = np.eye(n)
    for _ in range(T):
        Pr = Pr @ P
        S += Pr
    M = (vol / (b * T)) * (S / d[None, :])       # (vol/bT) * S * D^{-1}
    M = np.log(np.maximum(M, 1.0))               # log(max(M,1)); sub-1 entries -> 0
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
# Raw-feature baselines (no embedding) -- the honest yardstick
# ---------------------------------------------------------------------------


def raw_correlation_vectors(returns: pd.DataFrame) -> dict[str, list[float]]:
    """Each stock IS its row of the Pearson correlation matrix (n-dimensional feature).

    The "raw correlation features fed straight to the classifier" baseline. Cosine similarity
    of two correlation rows is the second-order "do these stocks correlate with the same
    things" signal -- a strong, embedding-free representation.

    The diagonal (each stock's self-correlation = 1.0) is zeroed: it is a unique per-stock
    giveaway that leaks identity into the SVM and inflates the score (~+0.05 F1 on the Dolphin
    set; 0.774 -> 0.720). Removing it is what makes this a HONEST raw-feature yardstick rather
    than a flattered one.
    """
    C = np.nan_to_num(returns.corr().values).astype(np.float64)
    np.fill_diagonal(C, 0.0)        # remove self-correlation leak
    tickers = list(returns.columns)
    return {t: C[i].tolist() for i, t in enumerate(tickers)}


def raw_return_vectors(returns: pd.DataFrame) -> dict[str, list[float]]:
    """Each stock IS its standardized return time series (T-dimensional feature).

    Cosine of two standardized series is (up to centering) their return correlation, so this
    is the rawest possible "no model at all" representation.
    """
    X = returns.values.T.astype(np.float64)      # (n_stocks, T)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1e-12
    Xn = (X - mu) / sd
    tickers = list(returns.columns)
    return {t: Xn[i].tolist() for i, t in enumerate(tickers)}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def build_embeddings(cfg, returns: pd.DataFrame, seed: int) -> dict[str, list[float]]:
    """Produce `{ticker: vector}` for `cfg.embedder`, reusing GraphBuilder for graph methods.

    `cfg.embedder` (default "node2vec") selects the family. Graph-based families share the
    EXACT graph GraphBuilder produces for this cfg (same matrix / co-movement removal / filter),
    so only the embedding step differs -- a fair, isolated comparison.
    """
    embedder = getattr(cfg, "embedder", "node2vec")

    if embedder == "raw_corr":
        return raw_correlation_vectors(returns)
    if embedder == "raw_returns":
        return raw_return_vectors(returns)
    if embedder == "contrastive":
        from benchmark.contrastive_embedding import train_contrastive_embedding
        return train_contrastive_embedding(returns, dim=cfg.dimensions, seed=seed)

    # Graph-based families: build the graph once, then embed.
    from train_vectors import GraphBuilder, EmbeddingTrainer

    if embedder == "node2vec":
        graph = GraphBuilder(cfg).build(returns)
        return EmbeddingTrainer(cfg).train(graph)

    graph = GraphBuilder(cfg).build(returns)
    # node2vec drops isolates inside its trainer; do the same so all families see one graph.
    isolates = list(nx.isolates(graph))
    if isolates:
        graph.remove_nodes_from(isolates)

    if embedder == "spectral":
        return laplacian_eigenmaps(graph, cfg.dimensions, seed=seed)
    if embedder == "netmf":
        return netmf(graph, cfg.dimensions, window=getattr(cfg, "window", 10), seed=seed)
    raise ValueError(f"Unknown embedder {embedder!r}")
