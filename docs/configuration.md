# Configuration

Every parameter lives on the [`Config`](api.md#stock_embeddings.Config) dataclass and can
be passed as a keyword argument to `train_embeddings()` / `fit()`. All defaults come from
the research benchmark behind the
[paper](https://dx.doi.org/10.2139/ssrn.7591058).

## Embedding

| Parameter | Default | Description |
|---|---|---|
| `embedder` | `"node2vec"` | Embedding family: `"node2vec"`, `"spectral"`, `"netmf"`, `"raw_corr"`, `"raw_returns"`. See [Embedding families](embedders.md). |
| `dimensions` | `64` | Embedding dimensionality (ignored by `raw_*` embedders, which are data-shaped). |
| `seed` | `42` | Random seed for walks, Word2Vec init, and eigensolvers. |

## Graph construction

| Parameter | Default | Description |
|---|---|---|
| `graph_filtering` | `"threshold"` | How the similarity matrix becomes a graph: `"threshold"`, `"mst"`, `"knn"`, `"signed_knn"`. |
| `correlation_threshold` | `0.65` | Connect stocks if \|correlation\| > threshold (threshold mode only). With `use_precision_matrix`, use a small value (~0.02–0.1). |
| `repair_islands` | `True` | **Forced Connectivity** (the paper's contribution): repair under-connected nodes up to `min_degree`. Set `False` for a plain threshold graph. |
| `min_degree` | `3` | Target minimum degree used by Forced Connectivity. |
| `knn_k` | `10` | Edges per node in the kNN graph (kNN modes only). |
| `mst_extra_edges_pct` | `0.05` | `0.0` = pure MST; `0.05` = add the top 5% of strongest unused edges (PMFG-style). MST mode only. |
| `mst_squared_corr` | `True` | Use correlation² as the MST edge weight (emphasizes very strong ties). |
| `mst_distance_transform` | `False` | Mantegna (1999) distance-transform MST on **signed** correlation, so strongly negative pairs are treated as far apart. Overrides the two options above. |

## Node2Vec walk parameters

| Parameter | Default | Description |
|---|---|---|
| `walk_p` | `0.8` | Return parameter. High p ⇒ avoid returning to the start ⇒ explore further. |
| `walk_q` | `7.2` | In-out parameter. High q ⇒ stay local (BFS-like); low q ⇒ go far (DFS-like). |
| `walk_length` | `20` | Length of each random walk. |
| `num_walks` | `100` | Number of walks per node. |
| `window` | `10` | Skip-gram context window (also the NetMF window). |

## Similarity matrix estimation

| Parameter | Default | Description |
|---|---|---|
| `use_precision_matrix` | `False` | `True` = Graphical Lasso partial correlation (removes indirect market noise; O(n³), small universes only). `False` = Pearson correlation. |
| `lasso_alpha` | `0.0` | `0.0` ⇒ `GraphicalLassoCV` (picks alpha by cross-validation; slowest). `> 0.0` ⇒ fixed-alpha `GraphicalLasso` (fast; higher alpha = sparser graph). |
| `market_neutral` | `False` | Regress each stock's returns on a market factor and keep the residual, so edges reflect idiosyncratic co-movement. O(n·T), scales to large universes. |
| `market_ticker` | `"SPY"` | Market factor column for `market_neutral`. If absent, an equal-weighted average of all stocks is used. |
| `cross_sectional_demean` | `False` | Subtract each day's equal-weighted cross-sectional mean return. The cheapest co-movement removal; O(n·T). |
| `spectral_clean` | `"none"` | Eigenvalue (RMT) cleaning of the correlation matrix: `"none"`, `"market"` (remove the top eigenmode — the data-driven market factor), `"topk"` (remove the top `spectral_k` modes), `"rmt"` (keep only modes above the Marchenko–Pastur noise edge). |
| `spectral_k` | `1` | Number of eigenmodes removed by `"topk"` / `"rmt"`. |

Co-movement removal options compose: preprocessing is applied to the returns first
(`cross_sectional_demean`, then `market_neutral`), then the matrix is computed
(correlation or lasso), then `spectral_clean` is applied to the correlation matrix.

!!! tip "Which co-movement removal?"
    From the benchmark: **`spectral_clean="market"` (PCA market-mode removal) is the best
    cheap, scalable option**; `market_neutral=True` is roughly equivalent. Graphical Lasso
    wins at small n but doesn't scale. RMT band reconstruction over-denoises — avoid it
    for retrieval tasks. See the [paper](research.md).

## Data preparation

| Parameter | Default | Description |
|---|---|---|
| `min_history_frac` | `0.8` | Drop tickers with less than this fraction of non-NaN price observations. |
| `min_rows` | `30` | Minimum number of return observations required to train. |
| `verbose` | `True` | Print progress logs. |
