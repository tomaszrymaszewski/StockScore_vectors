# stock-embeddings

Train vector embeddings of stocks from their price history. Give the library a dict of
`{ticker: prices}`, and it builds a stock-similarity graph from return correlations and
embeds every stock into a dense vector — stocks that move together (and belong to similar
businesses) end up close in embedding space.

```python
from stock_embeddings import train_embeddings

data = {"AAPL": aapl_prices, "MSFT": msft_prices, "XOM": xom_prices, ...}
embeddings = train_embeddings(data)   # -> {"AAPL": [64 floats], "MSFT": [...], ...}
```

**Full documentation: [docs.stockscore.live](https://docs.stockscore.live/)**

## Summary

`stock-embeddings` turns raw price panels into a `{ticker: vector}` dictionary using the
pipeline from our research on correlation-graph stock embeddings:

```
prices -> returns -> correlation matrix -> graph -> embedding -> {ticker: vector}
```

Every stage is configurable with sane defaults: co-movement removal (market-neutral
residuals, cross-sectional demeaning, spectral/RMT cleaning), graph construction
(threshold + Forced Connectivity repair, MST, kNN), and the embedding family itself
(Node2Vec, Laplacian Eigenmaps, NetMF, or raw-feature baselines).

## Why this is useful

- **Semantic stock similarity.** Embedding-space neighbours capture economic similarity
  (sector, industry, business model) that raw correlation misses, because the graph +
  random walks propagate relationships beyond direct pairwise correlation.
- **Ready-made features.** The vectors are a compact, dense feature representation for
  downstream ML: sector classification, clustering, peer selection, retrieval, or risk
  models.
- **Turnkey.** One function call, no data plumbing, no graph code, no tuning required —
  defaults come from a seed-averaged, significance-tested research benchmark.
- **Honest and reproducible.** All randomness is seeded; alternative embedding families
  and no-embedding baselines are built in, so you can verify the embedding actually adds
  value on your data.

## Setup

Requires Python 3.10+.

```bash
pip install stock-embeddings
```

Or from source:

```bash
git clone https://github.com/tomaszrymaszewski/StockScore_vectors.git
cd StockScore_vectors
pip install .
```

## Quick start

```python
import pandas as pd
from stock_embeddings import train_embeddings

# Any dict of ticker -> prices works: pandas Series (optionally with a
# DatetimeIndex), lists, or numpy arrays.
prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
data = {ticker: prices[ticker].dropna() for ticker in prices.columns}

embeddings = train_embeddings(data)

# Nearest neighbours of a stock:
from stock_embeddings import fit
result = fit(data)
result.most_similar("AAPL", k=10)     # [("MSFT", 0.98), ("AMZN", 0.97), ...]
result.to_dataframe()                 # embeddings as a DataFrame
```

## Usage

Every parameter has a default; pass only what you want to change:

```python
embeddings = train_embeddings(
    data,
    dimensions=32,               # embedding size (default 64)
    embedder="spectral",         # node2vec | spectral | netmf | raw_corr | raw_returns
    graph_filtering="knn",       # threshold | mst | knn | signed_knn
    knn_k=10,
    market_neutral=True,         # regress out the market factor before building the graph
    spectral_clean="market",     # remove the market eigenmode from the correlation matrix
    seed=42,
)
```

Use `fit()` instead of `train_embeddings()` to get the full `TrainResult` — embeddings,
the returns matrix actually used, the similarity graph, and the resolved config — plus
helpers like `to_dataframe()` and `most_similar()`.

See the [documentation](https://docs.stockscore.live/) for the complete parameter
reference and guides.

## Research

This library implements the method developed in our paper
[*Stock Embeddings from Return-Correlation Graphs*](https://dx.doi.org/10.2139/ssrn.7591058),
which benchmarks how well correlation-graph embeddings recover the GICS classification of
stocks against honest baselines (raw correlation, MST, kNN, PMFG), with seed-averaged,
significance-tested results. Read the paper for how we got here — the method choices,
the defaults shipped in this library, and what the embeddings can and cannot do.

## License

MIT — see [LICENSE](LICENSE).
