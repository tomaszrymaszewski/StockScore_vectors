# Research paper

This library implements the method developed in:

> **[*Stock Embeddings from Return-Correlation Graphs*](https://dx.doi.org/10.2139/ssrn.7591058)**
> DOI: [10.2139/ssrn.7591058](https://dx.doi.org/10.2139/ssrn.7591058)

## What the paper does

The paper learns vector embeddings of equities from their return-correlation structure
(a correlation graph + Node2Vec) and evaluates how well the embedding's nearest
neighbours recover the GICS classification of each stock — a retrieval hit-rate measured
along the full GICS granularity ladder (sector → industry group → industry →
sub-industry). Every headline number is reproducible, seed-averaged, and
significance-tested against honest baselines (raw correlation, MST, kNN, PMFG).

## How the paper maps to this library

| Paper element | Library parameter |
|---|---|
| Forced Connectivity (threshold graph + minimum-degree repair — the paper's graph contribution) | `graph_filtering="threshold"`, `repair_islands=True`, `min_degree=3` (all defaults) |
| Competitor graph baselines (MST, PMFG-approx, kNN) | `graph_filtering="mst"` (+ `mst_extra_edges_pct`), `"knn"`, `"signed_knn"` |
| Co-movement removal (demean, market-neutral, PCA market-mode, RMT, Graphical Lasso) | `cross_sectional_demean`, `market_neutral`, `spectral_clean`, `use_precision_matrix` |
| Embedding-family comparison (does the verdict generalize beyond Node2Vec?) | `embedder="node2vec" \| "spectral" \| "netmf"` |
| Honest no-embedding baselines | `embedder="raw_corr" \| "raw_returns"` |
| Seed-averaged evaluation | `seed` (sweep many seeds, compare mean ± std) |

## Key findings relevant to library users

- **Embeddings beat correlation-row features** on fine-grained (sub-industry) retrieval
  by a wide margin (+0.11–0.21), across every embedding family.
- **The strong raw-return-series baseline ties the best embedding** on sector
  classification — the embedding's value over raw returns is *compression* (64 numbers
  instead of a T-day series), not raw accuracy.
- **PCA market-mode removal (`spectral_clean="market"`) is the best cheap, scalable
  co-movement removal**; market-neutral residualization is roughly equivalent.
  Graphical Lasso wins at small n but is O(n³) and doesn't scale. RMT band
  reconstruction over-denoises and hurts fine-grained retrieval.
- **Node2Vec walks are stochastic**: a single run is a single noisy sample. The paper
  sweeps many seeds and reports mean ± std — do the same when comparing configurations.

## Scope and limitations (from the paper)

- **Survivorship bias.** The benchmark universes are survivor panels; results compare
  methods under clean, idealised conditions and are not a recommendation for live
  trading.
- **GICS is not ground truth.** It is a convenient reference taxonomy; a company's
  economic identity can change while its label stays fixed.
- **Method comparison, not suitability.** The paper asks which representation recovers
  GICS structure best — not whether these embeddings improve a trading strategy. Its
  out-of-sample hedging test shows representation quality does not automatically
  translate to portfolio improvement.
