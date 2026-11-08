# Changelog

All notable changes to `stock-embeddings` are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-08

Initial release.

### Added

- `train_embeddings(data, **params)` — train stock embeddings from a dict of
  `{ticker: price series}` (pandas Series, lists, or numpy arrays), with research-backed
  defaults for every parameter.
- `fit(data, **params)` — same pipeline, returning a full `TrainResult` (embeddings,
  returns matrix, similarity graph, resolved config) with `to_dataframe()` and
  `most_similar()` helpers.
- `Config` dataclass exposing every knob: embedding family, graph construction,
  Node2Vec walk parameters, similarity-matrix estimation, co-movement removal, and data
  preparation.
- Embedding families: `node2vec` (default), `spectral` (Laplacian Eigenmaps), `netmf`
  (DeepWalk as matrix factorization), and the honest no-embedding baselines `raw_corr`
  and `raw_returns`.
- Graph construction: correlation threshold with Forced Connectivity repair (the paper's
  contribution), MST (+ PMFG-style extra edges, Mantegna distance transform), kNN, and
  signed kNN.
- Co-movement removal: market-neutral residualization, cross-sectional demeaning,
  spectral/RMT eigenmode cleaning, and Graphical-Lasso partial correlations.
- Seeded determinism for all stochastic components.
- Test suite (`tests/`) and full documentation (`docs/`, published at
  https://docs.stockscore.live/).

[0.1.0]: https://github.com/tomaszrymaszewski/StockScore_vectors/releases/tag/v0.1.0
