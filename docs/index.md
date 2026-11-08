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

## Summary

`stock-embeddings` turns raw price panels into a `{ticker: vector}` dictionary using the
pipeline from our research on correlation-graph stock embeddings:

```
prices -> returns -> correlation matrix -> graph -> embedding -> {ticker: vector}
```

Every stage is configurable with sane defaults:

| Stage | Options |
|---|---|
| Co-movement removal | market-neutral residuals, cross-sectional demeaning, spectral/RMT eigenmode cleaning, Graphical Lasso partial correlations |
| Graph construction | correlation threshold + Forced Connectivity repair, MST (+ PMFG-style extra edges), kNN, signed kNN |
| Embedding | Node2Vec (default), Laplacian Eigenmaps, NetMF, raw-feature baselines |

## Why this is useful

- **Semantic stock similarity.** Embedding-space neighbours capture economic similarity
  (sector, industry, business model) that raw correlation misses, because the graph and
  random walks propagate relationships beyond direct pairwise correlation.
- **Ready-made features.** The vectors are a compact, dense feature representation for
  downstream ML: sector classification, clustering, peer selection, retrieval, or risk
  models.
- **Turnkey.** One function call, no data plumbing, no graph code, no tuning required —
  the defaults come from a seed-averaged, significance-tested research benchmark.
- **Honest and reproducible.** All randomness is seeded; alternative embedding families
  and no-embedding baselines are built in, so you can verify the embedding actually adds
  value on your own data.

## Research

This library implements the method developed in our paper
[*Stock Embeddings from Return-Correlation Graphs*](https://dx.doi.org/10.2139/ssrn.7591058).
See [Research paper](research.md) for what the paper shows and how it maps to the
library's defaults.
