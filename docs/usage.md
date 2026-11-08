# Usage

## Input data

The library takes a dict of `{ticker: prices}`:

```python
data = {
    "AAPL": aapl_price_series,   # pandas Series (DatetimeIndex preferred)
    "MSFT": [101.2, 102.4, ...], # plain list of prices
    "XOM":  xom_price_array,     # numpy array
}
```

Rules:

- Values are **price levels**, not returns — the library computes simple returns itself.
- Series with a `DatetimeIndex` are aligned on dates; plain lists/arrays are aligned
  positionally (outer join, so different lengths are fine).
- Missing holes are forward-filled; tickers with less than `min_history_frac` (default
  80%) non-NaN observations are dropped; flat-lined stocks are removed.
- At least `min_rows` (default 30) return observations and 2 surviving tickers are
  required, otherwise a `ValueError` is raised.

## Two entry points

### `train_embeddings()` — just the vectors

```python
from stock_embeddings import train_embeddings

embeddings = train_embeddings(data, dimensions=32, seed=7)
# -> dict[str, list[float]]
```

### `fit()` — the full result

```python
from stock_embeddings import fit

result = fit(data, market_neutral=True)
```

`TrainResult` fields and helpers:

| Member | What |
|---|---|
| `result.embeddings` | `{ticker: vector}` — the primary output |
| `result.returns` | the aligned returns DataFrame actually used |
| `result.graph` | the stock similarity graph (`networkx.Graph`; `None` for `raw_*` embedders) |
| `result.config` | the resolved `Config` for this run |
| `result.to_dataframe()` | embeddings as a DataFrame indexed by ticker |
| `result.most_similar(ticker, k)` | top-k cosine-similar tickers with scores |

## Passing parameters

Every [`Config`](configuration.md) field is a keyword argument:

```python
embeddings = train_embeddings(
    data,
    dimensions=32,
    embedder="spectral",
    graph_filtering="knn",
    knn_k=10,
    market_neutral=True,
    spectral_clean="market",
    seed=42,
    verbose=False,          # silence progress logs
)
```

Unknown keyword arguments raise a `TypeError`. You can also pass a pre-built `Config`
and override individual fields on top of it:

```python
from stock_embeddings import Config

cfg = Config(dimensions=16, graph_filtering="knn")
embeddings = train_embeddings(data, config=cfg, seed=7)   # seed overrides cfg.seed
```

## Logging

With `verbose=True` (the default) the library prints progress via the standard
`logging` module under the `stock_embeddings` logger. Set `verbose=False` for silence,
or configure the logger yourself:

```python
import logging
logging.getLogger("stock_embeddings").setLevel(logging.WARNING)
```

## Determinism

All randomness is seeded from `Config.seed` (Node2Vec walks, Word2Vec init,
eigensolvers). Node2Vec trains with `workers=1`, which is required for gensim Word2Vec
determinism. The `spectral`, `netmf`, `raw_corr`, and `raw_returns` embedders are
deterministic regardless of seed.

## Using the vectors downstream

```python
import numpy as np

df = result.to_dataframe()
X = df.values                       # (n_stocks, dimensions) feature matrix
tickers = df.index.tolist()

# Cosine similarity between all pairs:
M = X / np.linalg.norm(X, axis=1, keepdims=True)
sim = M @ M.T
```

The vectors work with any scikit-learn estimator (KMeans, SVM, t-SNE/UMAP), any vector
database, or your own retrieval code.
