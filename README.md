# StockScore — Node2Vec stock embeddings & GICS retrieval benchmark

Learn vector embeddings of equities from their return-correlation structure (a graph +
[Node2Vec]), and evaluate how well the embedding's nearest neighbours recover the GICS
classification of each stock — a **retrieval hit-rate** measured along the full GICS
granularity ladder.

This repository contains the research/benchmark code behind the paper, designed so every
headline number is **reproducible, seed-averaged, and significance-tested**, and compared
against honest baselines (raw correlation, MST, kNN, PMFG).

[Node2Vec]: https://snap.stanford.edu/node2vec/

## What the benchmark measures

For each stock, take its top-`k` nearest neighbours in embedding space and ask what
fraction share its GICS label. Reported at four granularities:

| level            | # classes | role |
|------------------|-----------|------|
| `sector`         | 11        | anchors to the broad literature (~60% regime) |
| `industry_group` | 25        | |
| `industry`       | 74        | |
| `sub_industry`   | 159       | the fine-grained target where the method's value shows |

Metrics: **hit-rate@k**, success@k, and MRR for `k ∈ {1,3,5,10}`, plus clustering-agreement
(V-measure / NMI / ARI) to compare against prior work.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Large price CSVs are **not committed** (gitignored). Two small reference files *are* in the
repo so labels work out of the box:

- `data/gics_hierarchy.csv` — sub-industry → industry → industry group → sector rollup.
  Rebuild/validate with `python -m jobs.build_gics_hierarchy`.
- `data/Holdings_details_Russell_3000_ETF.csv` — iShares IWV holdings (ticker → sub-industry).

Regenerate the price history (needs `POLYGON_API_KEY`; copy `.env.example` → `.env`):

```bash
python -m jobs.fetch_prices --universe r3000 --period 2y
python -m jobs.fetch_prices --universe sp500 --period 5y
```

## Run the benchmark

```bash
# All methods, 10 seeds, on the Russell 3000 / 2y universe:
python -m benchmark.run_benchmark --universe r3000 --period 2y --seeds 0-9 --methods all
# Just our method:
python -m benchmark.run_benchmark --universe sp500 --period 0.5y --seeds 0-4 --methods ours
```

Results (per-method, seed-averaged hit-rates with std) are written to
`results/benchmark_summary.csv` and printed as a table. `run_benchmark` trains and scores in a
single pass; use the two-step flow below when you want to keep the trained vectors.

### Decoupled flow: save a vector space, then score it

Training the embedding is the expensive part. Generate it once to JSON, then run any stats on
that file without retraining:

```bash
# Step 1 — generate + save the vector space (per seed) to JSON:
python -m benchmark.build_vectors --universe r3000 --period 2y --seeds 0-9 \
    --method forced_connectivity --out results/vectors/fc_r3000_2y.json

# Step 2 — compute every metric from the saved JSON (no retraining):
python -m benchmark.stats_from_vectors results/vectors/fc_r3000_2y.json --out results/fc_stats.csv

# Optional — paired significance between two saved vector spaces:
python -m benchmark.build_vectors --method mst   --universe r3000 --period 2y --seeds 0-9 \
    --out results/vectors/mst_r3000_2y.json
python -m benchmark.stats_from_vectors results/vectors/fc_r3000_2y.json \
    --compare results/vectors/mst_r3000_2y.json --level sub_industry --k 3
```

Step 2 reports hit-rate@k, success@k, MRR, and V-measure/NMI/ARI at all four GICS levels
(mean ± std over the saved seeds), and `--compare` adds paired Wilcoxon + bootstrap CI and
McNemar between the two files.

### At cloud scale (Modal)

The sweep is embarrassingly parallel over (method × seed). Run it across containers:

```bash
pip install modal && modal token new
modal run modal_app.py --universe r3000 --period 2y --seeds 0-19 --methods all
```

> Node2Vec here is gensim Word2Vec — **CPU-bound** (`workers=1` for determinism). Modal's
> win is many parallel CPU containers, not a GPU. A GPU Node2Vec (PyTorch-Geometric) is a
> documented future option in `modal_app.py`.

## Reproducing the paper's claims

| Claim / table | How |
|---|---|
| **Beats raw-correlation top-k** (the null hypothesis) | `run_benchmark` reports `raw_correlation` alongside the embedding; `benchmark.significance.compare_methods` gives the paired Wilcoxon p-value + effect size. |
| **Seed variance / mean ± std** | every config is run over N seeds (`--seeds 0-19`); see `aggregate_over_seeds`. |
| **Competitor baselines under one metric** | `competitor_specs` runs MST / PMFG / kNN / threshold-no-repair / forced-connectivity through the identical pipeline. |
| **Significance** | `benchmark/significance.py`: paired Wilcoxon, McNemar, paired bootstrap CI, Holm + BH-FDR correction, effect sizes. |
| **Granularity ladder** | every run reports all four GICS levels. |
| **V-measure / NMI / ARI** | `benchmark/clustering_metrics.py` (and the note that V-measure inflates at fine granularity). |
| **Universe / covariance confound** | `benchmark/matched_sampling.py` — matched node-count comparison of R3000 vs S&P 500. |
| **Co-movement removal (graph construction)** | `benchmark/lasso_experiment.py` — raw correlation vs partial-correlation (Graphical Lasso) vs market-neutral residual, at matched density. See "Graph construction" below. |
| **Direct Dolphin et al. comparison** | `benchmark/dolphin_benchmark.py` (or `modal run modal_app.py::dolphin`) — our methods through their SVM-sector protocol on the reconstructed 611-stock benchmark. See "Direct comparison to Dolphin et al." below. |
| **Speak the literature's metrics** | `benchmark/literature_comparison.py` — V-measure (Sarmah) + SVM-F1 (Dolphin) across all saved vectorspaces → `results/literature_metrics.csv`. |

## Graph construction: removing co-movement

The graph is built from a stock×stock relationship matrix. Raw correlation mixes the common
*market* factor with *direct* relationships, so several methods strip out the shared co-movement
to leave idiosyncratic structure — trading fidelity against compute. All are composable and
plumbed through `RunSpec`.

| Method | Config flag | Cost | Scales? |
|---|---|---|---|
| Raw correlation (default) | — | O(n²) | yes |
| Cross-sectional demean | `cross_sectional_demean=True` | O(n·T) | yes |
| Market-neutral residual (SPY/proxy) | `market_neutral=True` | O(n·T) | yes |
| PCA market-mode removal | `spectral_clean="market"` (or `"topk"`, `spectral_k`) | O(n²·k) | yes |
| RMT band reconstruction | `spectral_clean="rmt"` (+ `spectral_k`) | O(n²·k) | yes |
| Partial correlation (Graphical Lasso) | `use_precision_matrix=True` (+ `lasso_alpha`) | O(n³) | no — small graphs only (O(n³); ill-conditioned at T<n) |

Controlled comparison on the S&P 500 (n=200, 5y, matched average degree ~5/10/20, 3 seeds);
**sub-industry hit@3 is the key fine-grained metric**:

| method | hit@3 sector | hit@3 sub-ind | V sector | build (n=200) |
|---|---|---|---|---|
| raw correlation | 0.68–0.74 | 0.25–0.29 | 0.53–0.57 | 0.1 s |
| cross-sectional demean | 0.66–0.70 | 0.25–0.28 | 0.47–0.49 | 0.1 s |
| market-neutral (SPY/proxy) | 0.70–0.71 | 0.27–0.30 | 0.49–0.56 | 0.1 s |
| **PCA market-mode** | 0.69–0.73 | **0.26–0.30** | 0.52–0.58 | 0.4 s |
| RMT band | 0.65 | 0.22 | 0.57–0.59 | 0.7 s |
| **Graphical Lasso** | **0.76–0.78** | **0.28–0.32** | **0.62–0.64** | 68 s (CV) / ~1 s (fixed α) |

Findings (ranking on sub-industry hit@3: **lasso > PCA-market ≈ market-neutral > correlation ≈
demean > RMT**):

- **Graphical Lasso wins at small-n matched density, but doesn't scale.** It removes *all* indirect
  paths and wins on every metric/density in the small-n test, but is O(n³) and gets ill-conditioned
  when T < n, so it's impractical on large universes (no node guard — it will just run slowly).
  `lasso_alpha` controls sparsity (fixed α ~1 s vs ~68 s CV). Precision is normalized to a partial-
  correlation scale, so use a *small* threshold (~0.02–0.1), not 0.65.
- **PCA market-mode removal is the best cheap, scalable method.** Removing the top eigenmode (the
  data-driven market factor) via a truncated solver (`eigsh`, O(n²·k)) improves sub-industry
  retrieval over raw correlation while staying neutral on sector, in ~0.4 s. This is the recommended
  co-movement removal when the lasso is infeasible. The SPY/proxy regression (`market_neutral`) is
  ≈ equivalent, marginally behind.
- **Don't over-denoise (confirmed empirically).** RMT *band* reconstruction — keep only the modes
  above the Marchenko–Pastur edge λ₊=(1+√(n/T))², dropping the market mode *and* the noise bulk — is
  the **worst** on retrieval/sub-industry: it shreds the fine sub-industry chains that live partly in
  the bulk. Removing only the top market mode is the sweet spot; reconstructing from the band alone
  goes too far.
- **Cross-sectional demeaning is too blunt** — flat per-day mean removal slightly *hurts* sector and
  doesn't help sub-industry; the beta-weighted regression (`market_neutral`) does better.

Reproduce:  `python -m benchmark.lasso_experiment --n 200 --period 5y`

## Direct comparison to Dolphin et al. (ICAIF 2024)

We reconstruct Dolphin et al.'s public benchmark (arXiv:2407.18645) from the Kaggle "Daily
Historical Stock Prices (1970–2018)" dataset (`jobs.prep_kaggle_data --dolphin`) and run our
methods through their exact protocol — **SVM + 5-fold CV sector classification (weighted F1)** on
the NASDAQ-taxonomy sector labels. Their leaderboard (all `*` rows are best-effort
reimplementations by Dolphin; only the contrastive row is a method's own number):

> Dolphin[9] 0.60\* · **Sarmah 0.61\*** (node2vec on corr) · Dolphin[10] 0.66\* · **Contrastive 0.69**

Our full-period-survivor set is ~1,113 stocks (their exact 611 filter is unknown — ours is ~2×
larger / harder; we control for that below). Results (dim=16, 5 seeds, SVM rbf C=10):

| method (dim 16) | SVM-F1 sector | hit@3 sector |
|---|---|---|
| **pca_market** (Node2Vec) | **0.741** | 0.709 |
| **market_neutral** (Node2Vec) | **0.741** | 0.702 |
| demean | 0.730 | 0.686 |
| rmt | 0.717 | 0.666 |
| corr | 0.702 | 0.651 |
| fc | 0.690 | 0.642 |
| lasso | 0.570 | 0.513 |
| netmf (matrix factorization) | 0.658 | 0.582 |
| **contrastive** (Dolphin's own method, reimpl.) | 0.657 | 0.578 |
| spectral (Laplacian eigenmaps) | 0.646 | 0.579 |
| raw return series *(no embedding)* | 0.745 | 0.635 |
| raw correlation rows, leak-free *(no embedding)* | 0.721 | 0.583 |

Two firm-up checks (from the saved vectorspaces, no retraining):
- **Matched universe size** (subsample to N=611): costs only ~0.02 F1 (pca_market 0.719 @611 vs
  0.741 @1,113) — universe size is not the driver.
- **SVM-config robustness** (kernel × C ∈ {1,10,100}): F1 ranges ≈0.68–0.75 (pca_market 0.677–0.745)
  — this is the dominant uncertainty.

**This is NOT a controlled head-to-head — do not read it as "we match/beat Dolphin."** The absolute
level is a *universe artifact*: on this clean full-survivor set *every* method — embedding or simple
baseline — scores ~0.65–0.75, above Dolphin's 0.69, because clean large-caps are easy; the same
`fc` method drops to ~0.61 on a messier 90%-coverage universe. We even **reimplemented Dolphin's own
contrastive method** (`benchmark/contrastive_embedding.py`, pure NumPy — their method is a free
embedding matrix, not a neural encoder) and reproduce it to **0.657** (vs their 0.68–0.69 on a
curated 611-set).

The honest within-dataset reading is **parity, baseline-dependent**: at matched config the leak-free
raw baselines (return series **0.745**, correlation rows **0.721**) tie the best embedding (pca_market
0.741) and sit above the contrastive method. An earlier draft reported "raw features 0.77 beat the
embedding" — that was an artifact of a **self-correlation leak** (each correlation row contains its
own 1.0; removing it drops 0.774 → 0.721) plus a favorable SVM config. With the leak fixed, no
embedding beats the strong **return-series** baseline, but *every* embedding family clearly **beats
the correlation-row** baseline (+0.11–0.21 sub-industry retrieval). So the embedding's value over
naive correlation features is real; over the return series it is compression, not accuracy. This
generalizes across embedding families and is temporally stable — see RESULTS.md §6b–6e.

Reproduce: `python -m benchmark.dolphin_benchmark` (local) or `modal run modal_app.py::dolphin`
(cloud); build the data with `python -m jobs.prep_kaggle_data --min-coverage 1.0 --dolphin`. The full
cross-family sweep: `python -m benchmark.embedding_families` / `modal run modal_app.py::families`.

## Repository layout

```
train_vectors.py      Config + DataFetcher + GraphBuilder + EmbeddingTrainer (Node2Vec)
tickers.py            Polygon universe list (used by jobs/fetch_prices)
modal_app.py          cloud fan-out of the sweep
benchmark/            # the paper's evaluation harness (no Pinecone, reproducible)
  evaluation.py           per-stock match frames; hit-rate@k, success@k, MRR; raw-corr baseline
  labels.py               GICS labels at all 4 granularities
  run_benchmark.py        parameterized entrypoint + competitor registry  (CLI, train+score)
  embedding_methods.py    embedder dispatcher: node2vec | spectral | netmf | contrastive | raw_*
  contrastive_embedding.py  Dolphin's contrastive method, reimplemented (NumPy + Adam, no torch)
  embedding_families.py   does any embedding family beat raw features? hit/SVM/V across families (CLI)
  temporal_validation.py  out-of-sample: stability across disjoint windows + future-corr test (CLI)
  build_vectors.py        step 1: train + save a vector space to JSON  (CLI)
  stats_from_vectors.py   step 2: all metrics + paired significance from saved JSON  (CLI)
  significance.py         Wilcoxon / McNemar / bootstrap / Holm / BH-FDR
  clustering_metrics.py   V-measure / NMI / ARI
  classification_metrics.py  SVM 5-fold sector classification (Dolphin et al. protocol)
  matched_sampling.py     universe/covariance confound experiment
  lasso_experiment.py     co-movement removal: correlation vs lasso vs market-neutral (CLI)
  dolphin_benchmark.py    run our methods on Dolphin's benchmark, SVM-sector protocol (CLI)
  reproduce_sarmah.py     controlled reproduction of Sarmah et al. (S&P 500 2021, V-measure) (CLI)
  sarmah_improvements.py  apply all our techniques to Sarmah's network; which beat the MST baseline (CLI)
  literature_comparison.py  V-measure (Sarmah) + SVM-F1 (Dolphin) over all saved vectorspaces (CLI)
jobs/
  build_gics_hierarchy.py   build data/gics_hierarchy.csv (with completeness check)
  fetch_prices.py           regenerate price CSVs from Polygon
  prep_kaggle_data.py       distill the Kaggle 1970-2018 dataset into project artifacts
data/   results/      inputs and outputs (gitignored)
trash/                # legacy / product code, not part of the paper (gitignored; see trash/README.md)
```

The benchmark depends only on `train_vectors.py` + `benchmark/`; data prep lives in `jobs/`.
Superseded grid-search/Pinecone scripts and the old FastAPI product have been moved to
`trash/` and are excluded from the published repo.

## Reproducibility notes

- All randomness is seeded (`Config.seed` → Node2Vec walks + Word2Vec init; `workers=1`).
- A harmless `Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'` line may
  print on some BLAS builds during interpreter shutdown; it does not affect results.
