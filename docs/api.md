# API reference

## Top-level

### `stock_embeddings.train_embeddings`

```python
train_embeddings(data, config=None, **params) -> dict[str, list[float]]
```

Train stock embeddings from a dict of `{ticker: price series}`.

| Argument | Type | Description |
|---|---|---|
| `data` | `Mapping[str, pd.Series \| Sequence[float] \| np.ndarray]` | Ticker → **price levels** (not returns). A `DatetimeIndex` is used for alignment if present. |
| `config` | `Config \| None` | Optional pre-built `Config`; keyword arguments override its fields. |
| `**params` | — | Any [`Config`](#stock_embeddingsconfig) field, e.g. `dimensions=32`, `embedder="spectral"`. Unknown names raise `TypeError`. |

**Returns:** `{ticker: embedding vector}` dict.

**Raises:** `ValueError` if the data is empty, too short (`min_rows`), or fewer than 2
tickers survive cleaning; `TypeError` for unknown parameter names.

```python
>>> embeddings = train_embeddings({"AAPL": aapl_prices, "MSFT": msft_prices})
>>> embeddings = train_embeddings(data, dimensions=32, embedder="spectral")
```

### `stock_embeddings.fit`

```python
fit(data, config=None, **params) -> TrainResult
```

Identical to `train_embeddings` but returns the full [`TrainResult`](#stock_embeddingstrainresult)
(embeddings + returns + graph + resolved config).

### `stock_embeddings.TrainResult`

Dataclass returned by `fit()`.

| Member | Type | Description |
|---|---|---|
| `embeddings` | `dict[str, list[float]]` | The primary output. |
| `returns` | `pd.DataFrame` | The aligned returns DataFrame actually used (days × tickers). |
| `graph` | `nx.Graph \| None` | The stock similarity graph; `None` for `raw_*` embedders. Nodes are tickers; edges carry a `weight` attribute. |
| `config` | `Config` | The resolved config for this run. |

Methods:

#### `TrainResult.to_dataframe()`

```python
to_dataframe() -> pd.DataFrame
```

Embeddings as a DataFrame indexed by ticker (one column per dimension).

#### `TrainResult.most_similar`

```python
most_similar(ticker: str, k: int = 10) -> list[tuple[str, float]]
```

Top-k most cosine-similar tickers to `ticker` (excluding itself), as
`[(ticker, similarity), ...]` sorted by descending similarity. Raises `KeyError` if
`ticker` is not in the trained embeddings.

### `stock_embeddings.Config`

Dataclass with every training parameter; all fields have defaults. See
[Configuration](configuration.md) for the full table.

```python
from stock_embeddings import Config

cfg = Config(dimensions=32, embedder="spectral", graph_filtering="knn")
embeddings = train_embeddings(data, config=cfg)
```

## Data preparation

### `stock_embeddings.data.prices_to_returns`

```python
prices_to_returns(data, config=None) -> pd.DataFrame
```

Build a returns DataFrame from a dict of `{ticker: price series}`. Series are
outer-joined on their index (positional if no index), forward-filled, converted to
simple returns, and cleaned: non-trading days dropped, tickers with less than
`config.min_history_frac` coverage dropped, flat-lined stocks removed. Called
internally by `fit()`; exposed for advanced use.

## Graph construction

### `stock_embeddings.graph.GraphBuilder`

```python
GraphBuilder(config: Config)
```

Builds the stock similarity graph from a returns DataFrame according to the config
(matrix estimation, co-movement removal, edge filtering, Forced Connectivity repair).

#### `GraphBuilder.build`

```python
build(returns: pd.DataFrame) -> nx.Graph
```

Pipeline: optional co-movement removal (`cross_sectional_demean`, `market_neutral`) →
similarity matrix (Pearson correlation, optionally `spectral_clean`-ed or Graphical-Lasso
partial correlation) → edge filtering (`threshold` + repair, `mst`, `knn`,
`signed_knn`). Edges carry a `weight` attribute (|correlation|-scale).

## Embedders

All functions in `stock_embeddings.embedders` emit the same `{ticker: vector}` dict and
are used internally by the dispatcher; exposed for advanced use.

### `stock_embeddings.embedders.build_embeddings`

```python
build_embeddings(config: Config, returns: pd.DataFrame, graph: nx.Graph | None = None) -> dict[str, list[float]]
```

Dispatcher: produces embeddings for `config.embedder`, building the graph first if
`graph` is `None` and the family needs one.

### `stock_embeddings.embedders.node2vec_embeddings`

```python
node2vec_embeddings(G: nx.Graph, config: Config) -> dict[str, list[float]]
```

Random-walk + skip-gram embedding (the library default). Drops isolated nodes, seeds all
randomness from `config.seed`, trains with `workers=1` for determinism.

### `stock_embeddings.embedders.laplacian_eigenmaps`

```python
laplacian_eigenmaps(G: nx.Graph, dim: int, seed: int = 0) -> dict[str, list[float]]
```

Normalized-Laplacian eigenmap embedding (Belkin & Niyogi, 2003). Closed-form global
spectral decomposition; trivial near-1 modes are dropped.

### `stock_embeddings.embedders.netmf`

```python
netmf(G: nx.Graph, dim: int, window: int = 10, seed: int = 0, b: float = 1.0) -> dict[str, list[float]]
```

Closed-form skip-gram matrix factorization (small-window NetMF, Qiu et al., WSDM 2018).
Factorizes `log(max(M, 1))` where `M = (vol/(b·T)) · (Σ_{r=1..T} Pʳ) · D⁻¹`; embedding =
`U √S` from a truncated SVD.

### `stock_embeddings.embedders.raw_correlation_vectors`

```python
raw_correlation_vectors(returns: pd.DataFrame) -> dict[str, list[float]]
```

No-embedding baseline: each stock is its Pearson-correlation row with the diagonal
(self-correlation) zeroed to remove the identity leak.

### `stock_embeddings.embedders.raw_return_vectors`

```python
raw_return_vectors(returns: pd.DataFrame) -> dict[str, list[float]]
```

No-embedding baseline: each stock is its standardized return time series.

### Module constants

| Constant | Value | Meaning |
|---|---|---|
| `RAW_EMBEDDERS` | `{"raw_corr", "raw_returns"}` | Embedders that ignore the graph and read the returns matrix directly. |
| `DETERMINISTIC_EMBEDDERS` | `{"spectral", "netmf", "raw_corr", "raw_returns"}` | Embedders whose output does not depend on the seed. |
