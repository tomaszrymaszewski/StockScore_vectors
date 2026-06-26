"""
Retrieval-metric core -- the single source of truth for the paper's evaluation.

This replaces the two divergent, single-number implementations that existed before
(`metrics.NetworkMetrics.calculate_hit_rate`, k=3 sub-industry, and
`test_gics.calculate_sector_hit_rate`, k=5 sector via Pinecone). Everything here runs
locally on a plain `{ticker: vector}` dict, so it is agnostic to *how* the representation
was produced -- Node2Vec embeddings, the raw-correlation baseline, or any competitor
graph all flow through the same scoring.

The load-bearing object is the **per-stock neighbor-match matrix** returned by
`neighbor_match_frame`: one row per query stock, columns `match_1..match_K` indicating
whether the i-th nearest neighbor shares the query's label. Every aggregate metric
(hit-rate@k, success@k, MRR) and every significance test (paired Wilcoxon, McNemar,
bootstrap) is derived from this per-stock substrate -- so method-vs-method comparisons are
properly paired on the same stocks.

Definitions
-----------
* hit_rate@k  : mean over stocks of (#top-k neighbors sharing the label) / k.
                This is the paper's headline "hit-rate".
* success@k   : fraction of stocks with >= 1 same-label neighbor in the top-k.
* mrr@k       : mean reciprocal rank of the first same-label neighbor (0 if none in top-k).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_KS = (1, 3, 5, 10)


def _aligned_matrix(
    embeddings: dict[str, list[float]], labels: dict[str, str]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Restrict to tickers present in BOTH embeddings and labels.

    Returns (tickers, vectors, label_array) in a consistent order. Restricting the
    universe to labeled tickers guarantees every retrieved neighbor also has a label,
    so a "miss" is a genuine cross-class neighbor, never a missing annotation.
    """
    tickers = [t for t in embeddings if t in labels]
    if not tickers:
        return [], np.empty((0, 0)), np.empty((0,), dtype=object)
    vectors = np.asarray([embeddings[t] for t in tickers], dtype=np.float64)
    label_arr = np.asarray([labels[t] for t in tickers], dtype=object)
    return tickers, vectors, label_arr


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    unit = vectors / norms
    return unit @ unit.T


def neighbor_match_frame_from_similarity(
    tickers: list[str],
    similarity: np.ndarray,
    label_arr: np.ndarray,
    max_k: int,
) -> pd.DataFrame:
    """Per-stock neighbor-match matrix given a square similarity matrix.

    `similarity[i, j]` is "closeness" of j to i (higher = nearer); the diagonal is
    ignored (a stock is never its own neighbor). Works for cosine similarity of
    embeddings or a raw correlation matrix alike.

    Returns a DataFrame indexed by ticker with boolean columns `match_1..match_max_k`
    and a `label` column.
    """
    n = len(tickers)
    if n == 0:
        return pd.DataFrame()
    max_k = min(max_k, n - 1)

    sim = similarity.copy()
    np.fill_diagonal(sim, -np.inf)

    # Top-k neighbor indices per row, ranked nearest-first.
    # argpartition for speed, then sort the small top-k slice by similarity.
    part = np.argpartition(-sim, kth=max_k - 1, axis=1)[:, :max_k]
    rows = np.arange(n)[:, None]
    order = np.argsort(-sim[rows, part], axis=1)
    topk = part[rows, order]  # (n, max_k) neighbor indices, nearest first

    neighbor_labels = label_arr[topk]  # (n, max_k)
    matches = neighbor_labels == label_arr[:, None]  # (n, max_k) bool

    frame = pd.DataFrame(
        matches, index=tickers, columns=[f"match_{i + 1}" for i in range(max_k)]
    )
    frame.insert(0, "label", label_arr)
    return frame


def neighbor_match_frame(
    embeddings: dict[str, list[float]],
    labels: dict[str, str],
    max_k: int = max(DEFAULT_KS),
) -> pd.DataFrame:
    """Per-stock neighbor-match matrix for an embedding under cosine similarity."""
    tickers, vectors, label_arr = _aligned_matrix(embeddings, labels)
    if not tickers:
        return pd.DataFrame()
    sim = _cosine_similarity_matrix(vectors)
    return neighbor_match_frame_from_similarity(tickers, sim, label_arr, max_k)


def raw_correlation_match_frame(
    correlation: np.ndarray,
    corr_tickers: list[str],
    labels: dict[str, str],
    max_k: int = max(DEFAULT_KS),
    absolute: bool = False,
) -> pd.DataFrame:
    """Per-stock neighbor-match matrix for the RAW-CORRELATION baseline.

    The reviewer's primary null hypothesis: skip Node2Vec entirely and take each stock's
    top-k most-correlated peers straight from the correlation matrix. Same scoring, same
    labels, same downstream significance machinery -- so the embedding's contribution is
    exactly the margin over this.

    `absolute=True` ranks by |corr| (peers that co-move OR counter-move); default ranks by
    signed correlation (most positively co-moving peers), which is the natural "similar
    stock" notion and the harder baseline to beat.
    """
    keep = [i for i, t in enumerate(corr_tickers) if t in labels]
    tickers = [corr_tickers[i] for i in keep]
    if not tickers:
        return pd.DataFrame()
    sub = correlation[np.ix_(keep, keep)].astype(np.float64)
    sub = np.nan_to_num(sub)
    if absolute:
        sub = np.abs(sub)
    label_arr = np.asarray([labels[t] for t in tickers], dtype=object)
    return neighbor_match_frame_from_similarity(tickers, sub, label_arr, max_k)


# ---------------------------------------------------------------------------
# Aggregate metrics derived from a per-stock match frame
# ---------------------------------------------------------------------------


def _match_columns(frame: pd.DataFrame, k: int) -> np.ndarray:
    cols = [f"match_{i + 1}" for i in range(k)]
    cols = [c for c in cols if c in frame.columns]
    return frame[cols].to_numpy(dtype=bool)


def per_stock_hit_fraction(frame: pd.DataFrame, k: int) -> pd.Series:
    """Per-stock fraction of top-k neighbors sharing the label (continuous, for Wilcoxon)."""
    m = _match_columns(frame, k)
    return pd.Series(m.mean(axis=1), index=frame.index, name=f"hit_frac@{k}")


def per_stock_success(frame: pd.DataFrame, k: int) -> pd.Series:
    """Per-stock boolean: at least one same-label neighbor in the top-k (for McNemar)."""
    m = _match_columns(frame, k)
    return pd.Series(m.any(axis=1), index=frame.index, name=f"success@{k}")


def per_stock_reciprocal_rank(frame: pd.DataFrame, k: int) -> pd.Series:
    """Per-stock reciprocal rank of the first same-label neighbor within top-k (else 0)."""
    m = _match_columns(frame, k)
    rr = np.zeros(m.shape[0])
    for i in range(m.shape[0]):
        hit = np.flatnonzero(m[i])
        if hit.size:
            rr[i] = 1.0 / (hit[0] + 1)
    return pd.Series(rr, index=frame.index, name=f"rr@{k}")


def hit_rate(frame: pd.DataFrame, k: int) -> float:
    """Paper's headline hit-rate@k: mean over stocks of the per-stock hit fraction."""
    return float(per_stock_hit_fraction(frame, k).mean()) if len(frame) else 0.0


def success_at_k(frame: pd.DataFrame, k: int) -> float:
    return float(per_stock_success(frame, k).mean()) if len(frame) else 0.0


def mrr(frame: pd.DataFrame, k: int) -> float:
    return float(per_stock_reciprocal_rank(frame, k).mean()) if len(frame) else 0.0


@dataclass
class MetricBundle:
    n_stocks: int
    hit_rate: dict[int, float]
    success_at_k: dict[int, float]
    mrr: dict[int, float]

    def to_flat_dict(self, prefix: str = "") -> dict[str, float]:
        out: dict[str, float] = {f"{prefix}n_stocks": self.n_stocks}
        for k, v in self.hit_rate.items():
            out[f"{prefix}hit_rate@{k}"] = v
        for k, v in self.success_at_k.items():
            out[f"{prefix}success@{k}"] = v
        for k, v in self.mrr.items():
            out[f"{prefix}mrr@{k}"] = v
        return out


def metric_bundle(frame: pd.DataFrame, ks: tuple[int, ...] = DEFAULT_KS) -> MetricBundle:
    """Compute hit-rate, success@k, and MRR for every k from one per-stock match frame."""
    if len(frame) == 0:
        return MetricBundle(0, {k: 0.0 for k in ks}, {k: 0.0 for k in ks}, {k: 0.0 for k in ks})
    return MetricBundle(
        n_stocks=len(frame),
        hit_rate={k: hit_rate(frame, k) for k in ks},
        success_at_k={k: success_at_k(frame, k) for k in ks},
        mrr={k: mrr(frame, k) for k in ks},
    )


def per_subindustry_breakdown(
    frame: pd.DataFrame, k: int = 3
) -> pd.DataFrame:
    """hit-rate@k grouped by the query's label.

    Reveals whether the headline mean is carried by a few easy clusters (banks, REITs,
    utilities) or is broad-based.
    """
    if len(frame) == 0:
        return pd.DataFrame()
    frac = per_stock_hit_fraction(frame, k)
    g = frac.groupby(frame["label"])
    out = pd.DataFrame(
        {"n_stocks": g.size(), f"hit_rate@{k}": g.mean()}
    ).sort_values(f"hit_rate@{k}", ascending=False)
    return out
