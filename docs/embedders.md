# Embedding families

The graph/matrix construction is shared by all graph-based families — only the embedding
step changes. Select with `embedder=`.

| `embedder` | Family | Deterministic? | Output dim |
|---|---|---|---|
| `"node2vec"` (default) | Random walks + skip-gram | seeded | `dimensions` |
| `"spectral"` | Laplacian Eigenmaps | yes | `dimensions` |
| `"netmf"` | DeepWalk as matrix factorization | yes | `dimensions` |
| `"raw_corr"` | No embedding: correlation-row features | yes | n_stocks |
| `"raw_returns"` | No embedding: standardized return series | yes | n_days |

## node2vec

Local biased random walks (parameters `walk_p`, `walk_q`, `walk_length`, `num_walks`)
fed to a gensim Word2Vec skip-gram (`window`, `dimensions`). The library default; the
method evaluated in the paper. Stochastic but fully seeded (`seed`; `workers=1` for
determinism).

```python
embeddings = train_embeddings(data)   # node2vec by default
```

## spectral

Laplacian Eigenmaps (Belkin & Niyogi, 2003): embed each node by the bottom non-trivial
eigenvectors of the symmetric normalized Laplacian. A closed-form **global** spectral
decomposition — no walks, no stochastic training. A structurally different family from
Node2Vec; useful when you want a deterministic embedding of the same graph.

```python
embeddings = train_embeddings(data, embedder="spectral", dimensions=32)
```

## netmf

NetMF (Qiu et al., WSDM 2018): DeepWalk/node2vec **implicitly** factorize a closed-form
matrix `M = (vol/(b·T)) · (Σ Pʳ) · D⁻¹`; NetMF factorizes `log(max(M, 1))` directly by
truncated SVD. This isolates the skip-gram *objective* from the random-walk *sampling*:
if NetMF tracks Node2Vec on your data, the walks are not where the value is. Uses
`window` as T. Dense O(n³)-ish — for small/medium universes.

```python
embeddings = train_embeddings(data, embedder="netmf", dimensions=32)
```

## raw_corr / raw_returns — the honest baselines

No embedding at all:

- `raw_corr` — a stock **is** its row of the Pearson correlation matrix (the diagonal
  self-correlation is zeroed to remove the identity leak). Cosine similarity of two rows
  answers "do these stocks correlate with the same things?"
- `raw_returns` — a stock **is** its standardized return time series. Cosine of two
  standardized series is (up to centering) their correlation.

These are the yardstick: if no embedding beats them on your task, the embedding is a
lossy compression, not a gain. The paper's verdict: every embedding family clearly beats
`raw_corr` on fine-grained retrieval, while the strong `raw_returns` baseline ties the
best embedding on sector classification — the embedding's value over raw returns is
**compression** (64 numbers instead of a T-day series), not raw accuracy.

```python
baseline = train_embeddings(data, embedder="raw_corr")
```

!!! note
    `raw_*` embedders ignore the graph entirely (`TrainResult.graph` is `None`) and
    output data-shaped vectors (n_stocks / n_days dimensions), not `dimensions`.
