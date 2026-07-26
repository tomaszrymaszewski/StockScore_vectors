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
(V-measure / NMI / ARI) to compare against prior work. The benchmark also includes a simple
out-of-sample **hedging / risk** test: short a stock's k nearest peers (selected on the train window)
and measure the realised residual volatility in the test window.

---

## How this repo is organized — start here

**Two documents.** This README is the *how-to + map of the code*. [RESULTS.md](RESULTS.md) is the
*findings*: every empirical claim, seed-averaged and significance-tested, each tagged with the
script that produces it. If you want to know "what did they find and is it sound", read RESULTS.md;
if you want "how do I run X / where does Y live", stay here.

**One data flow, reused by every experiment.** Each experiment is the *same* pipeline with exactly
one stage swapped out:

```
 price CSV  ─▶  returns  ─▶  relationship matrix  ─▶  graph  ─▶  embedding  ─▶  {ticker: vector}  ─▶  metrics
 data/*.csv     DataFetcher    GraphBuilder            GraphBuilder  EmbeddingTrainer  (a plain dict)     evaluation.py
 (built by      .fetch_returns ._compute_matrix        .build        /                                   clustering_metrics.py
  jobs/)                       (corr·demean·market_     (threshold·   embedding_methods                   classification_metrics.py
                               neutral·pca·rmt·lasso)   mst·knn·FC)   .build_embeddings                   significance.py
```

The **representation is always a plain `{ticker: [floats]}` dict**, so Node2Vec embeddings, the
other embedding families, the raw-feature baselines, and every competitor graph all score through
the *identical* metric + significance code — differences reflect the method, not the measurement.
Trained vector spaces are saved to `results/vectors/*.json`; every metric can be recomputed from
them without retraining.

**Where to find what:**

| You want to… | File | Key symbol |
|---|---|---|
| **— building blocks —** | | |
| see/limit every knob (graph, matrix, embedder, walk params, seed) | `train_vectors.py` | `Config` |
| compute the relationship matrix (corr · demean · market-neutral · PCA-market · RMT · lasso) | `train_vectors.py` | `GraphBuilder._compute_matrix`, `_spectral_clean`, `_residualize_market`, `_cross_sectional_demean` |
| filter it into a graph (threshold · MST · kNN · forced-connectivity repair) | `train_vectors.py` | `GraphBuilder.build`, `_rescue_islands` |
| train the Node2Vec embedding | `train_vectors.py` | `EmbeddingTrainer.train` |
| pick an embedding family (node2vec · spectral · netmf · contrastive · raw_corr · raw_returns) | `benchmark/embedding_methods.py` | `build_embeddings` (dispatcher) |
| Dolphin's contrastive method, reimplemented (NumPy, no torch) | `benchmark/contrastive_embedding.py` | `train_contrastive_embedding` |
| describe one benchmark config (everything except the seed) | `benchmark/run_benchmark.py` | `RunSpec`, `.to_config` |
| load GICS labels at all 4 granularities | `benchmark/labels.py` | `load_all_levels` |
| **— metrics —** | | |
| retrieval: hit-rate@k, success@k, MRR (the headline metric) | `benchmark/evaluation.py` | `neighbor_match_frame`, `metric_bundle` |
| clustering: V-measure / NMI / ARI (Sarmah's language) | `benchmark/clustering_metrics.py` | `clustering_metrics` |
| supervised: SVM 5-fold sector F1 (Dolphin's protocol) | `benchmark/classification_metrics.py` | `svm_classification` |
| significance: paired Wilcoxon / McNemar / bootstrap CI / Holm·BH-FDR | `benchmark/significance.py` | `compare_methods`, `wilcoxon_compare` |
| save a vector space / score a saved one (no retraining) | `benchmark/build_vectors.py` · `stats_from_vectors.py` | CLIs |
| **— the experiments (→ RESULTS.md section) —** | | |
| graph-construction & co-movement sweep (§3, §4) | `benchmark/run_benchmark.py`, `benchmark/lasso_experiment.py` | `competitor_specs`, `comovement_specs` |
| metric-disagreement under universe expansion (§5) | `benchmark/run_benchmark.py` + `literature_comparison.py` | — |
| against the literature: Dolphin SVM-sector (§6b) | `benchmark/dolphin_benchmark.py` | `METHOD_FLAGS`, `evaluate` |
| reproduce Sarmah's V-measure (§6a) | `benchmark/reproduce_sarmah.py` | CLI |
| improve on Sarmah's exact network (§6a-bis) | `benchmark/sarmah_improvements.py` | `METHODS` |
| **does any embedding family beat raw features? (§6d)** | `benchmark/embedding_families.py` | `run_family`, `score_embedding`, `summarize` |
| **temporal / out-of-sample robustness (§6e)** | `benchmark/temporal_validation.py` | `main`, `mean_future_corr` |
| **out-of-sample hedging / risk (§6g)** | `benchmark/hedging_experiment.py` | `main`, `evaluate_hedging_window` |
| **dimension & Node2Vec p/q ablations (§6f)** | `modal_app.py` | `dim_sweep`, `pq_sweep` |
| universe/size confound | `benchmark/matched_sampling.py` | `run_matched_node_count` |
| literature metrics over every saved vector space | `benchmark/literature_comparison.py` | CLI |
| **— infrastructure —** | | |
| fan any sweep out across cloud containers | `modal_app.py` | entrypoints (see "Modal entrypoints" below) |
| (re)build the data (prices · Kaggle/Dolphin set · GICS rollup) | `jobs/` | `fetch_prices`, `prep_kaggle_data`, `build_gics_hierarchy` |

The full annotated file tree, the `Config` knob reference, and the method/metric vocabularies are at
the bottom under **[Repository layout](#repository-layout)**.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

**What's committed vs not.** Large price CSVs are **gitignored** (regenerable). Two small reference
files *are* committed so GICS labels work out of the box:

- `data/gics_hierarchy.csv` — sub-industry → industry → industry group → sector rollup.
  Rebuild/validate with `python -m jobs.build_gics_hierarchy`.
- `data/Holdings_details_Russell_3000_ETF.csv` — iShares IWV holdings (ticker → GICS sub-industry);
  the source of truth for labels (`benchmark/labels.py`).

**Universes & periods.** A universe×period is a price panel `data/<universe>_<period>.csv`
(rows = dates, cols = tickers), sliced to a trailing window ending at `Config.cutoff_date`:

| universe arg | file prefix | ~stocks | source |
|---|---|---|---|
| `sp500` | `data/sp500_*.csv` | ~470 (large-cap core) | Polygon (`jobs/fetch_prices.py`) |
| `r1000` | `data/russell_1000_*.csv` | ~880 | Polygon |
| `r3000` | `data/russell_3000_*.csv` | ~2,180 | Polygon |
| Dolphin set | `data/dolphin/returns_df_611.csv` | ~1,113 full-survivors, 2000–2018 | Kaggle (`jobs/prep_kaggle_data.py`) |
| Sarmah net | `data/sarmah_2017_returns.csv` | 504, 2017 log returns | Kaggle |

Periods are `0.5y · 1y · 2y · 5y`. Regenerate the price history (needs `POLYGON_API_KEY`; copy
`.env.example` → `.env`):

```bash
python -m jobs.fetch_prices --universe r3000 --period 2y
python -m jobs.fetch_prices --universe sp500 --period 5y
# Dolphin/Kaggle benchmark set (no API key; needs the Kaggle dump in data/kaggle_raw/):
python -m jobs.prep_kaggle_data --min-coverage 1.0 --dolphin
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

The sweep is embarrassingly parallel over (method × seed). Run it across containers. `modal_app.py`
exposes one entrypoint per experiment — pass the entrypoint after `::`:

```bash
pip install modal && modal token new
# Graph-construction OR co-movement sweep (saves vectorspaces + a metrics CSV):
modal run modal_app.py::main --suite comovement --universe r3000 --period 5y --seeds 0-19
# Other experiments:
modal run modal_app.py::families  --universes sp500,r1000,r3000 --seeds 0-4   # §6d embedding families
modal run modal_app.py::dolphin   --seeds 0-4                                 # §6b Dolphin SVM-sector
modal run modal_app.py::dim_sweep --universe sp500 --dims 8,16,32,64,128      # §6f dimension sweep
modal run modal_app.py::pq_sweep  --universe sp500                            # §6f Node2Vec p/q
modal run modal_app.py::sarmah_density --seeds 0-4                            # §6a-bis density sweep
```

Every entrypoint mounts the local `data/` directory into the container (so nothing is committed or
downloaded at run time), writes a metrics CSV to `results/`, and saves trained vector spaces to
`results/vectors/`. See the **Modal entrypoints** table at the bottom for what each one does.

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
| **OOS hedging / risk reduction** | `python -m benchmark.hedging_experiment` (or `modal run modal_app.py::hedging`) — out-of-sample peer-hedge residual volatility. |

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

Annotated tree — every file, with the classes/functions worth opening (CLI = runnable with
`python -m <module> --help`):

```
train_vectors.py            THE ENGINE. One pass: returns → matrix → graph → Node2Vec embedding.
  Config                      every knob (see "Config knobs" below); RunSpec.to_config() fills it
  DataFetcher.fetch_returns   load data/<universe>_<period>.csv → cleaned daily returns
  GraphBuilder.build          returns → graph (threshold | mst | knn, + forced-connectivity repair)
    ._compute_matrix            correlation / Graphical-Lasso precision matrix
    ._spectral_clean            PCA market-mode removal & RMT band (eigsh, O(n²k))
    ._residualize_market        market-neutral (SPY/proxy) residuals
    ._cross_sectional_demean    subtract each day's cross-sectional mean
    ._rescue_islands            Forced Connectivity: repair under-degree nodes
  EmbeddingTrainer.train      graph → {ticker: vector} via gensim Node2Vec (workers=1, seeded)
  PineconeClient/Evaluator/Pipeline   LEGACY product path (not used by the benchmark)
tickers.py                  Polygon universe ticker lists (used by jobs/fetch_prices)
modal_app.py                CLOUD fan-out — one @app.local_entrypoint per experiment (see table)

benchmark/                  THE EVALUATION HARNESS (no Pinecone, fully reproducible)
  run_benchmark.py            RunSpec (a config minus the seed); train_embedding(); the method
                              registries competitor_specs() / comovement_specs(); run_raw_correlation()
                              baseline; aggregate_over_seeds()                                   (CLI)
  embedding_methods.py        build_embeddings() dispatcher → node2vec | spectral (Laplacian
                              eigenmaps) | netmf (DeepWalk-as-matrix-factorization) | contrastive |
                              raw_corr | raw_returns; the leak-free raw-feature vectorizers
  contrastive_embedding.py    Dolphin et al. (2407.18645) contrastive method, reimplemented in
                              NumPy + hand-rolled Adam (no torch): build_cooccurrence(),
                              hypothesis_sampling(), train_contrastive_embedding()
  labels.py                   load_all_levels() — GICS sector/group/industry/sub-industry per ticker
  evaluation.py               neighbor_match_frame() (per-stock substrate) → hit-rate@k, success@k,
                              MRR (metric_bundle); raw_correlation_match_frame() baseline
  clustering_metrics.py       clustering_metrics() — KMeans V-measure / NMI / ARI / homogeneity
  classification_metrics.py   svm_classification() — SVM 5-fold weighted-F1 (Dolphin's protocol)
  significance.py             compare_methods(), wilcoxon_compare(), mcnemar_compare(),
                              bootstrap_ci(), Holm + BH-FDR, effect sizes
  build_vectors.py            STEP 1 — train a method over seeds, save results/vectors/*.json   (CLI)
  stats_from_vectors.py       STEP 2 — all metrics (+ --compare significance) from a saved json (CLI)
  embedding_families.py       §6d — node2vec/spectral/netmf/contrastive vs raw, hit/SVM/V          (CLI)
  temporal_validation.py      §6e — disjoint-window stability + future-correlation OOS test        (CLI)
  hedging_experiment.py       §6g — out-of-sample peer-hedge residual volatility (economic test)      (CLI)
  dolphin_benchmark.py        §6b — our methods through Dolphin's SVM-sector protocol; METHOD_FLAGS (CLI)
  reproduce_sarmah.py         §6a — faithful Sarmah reproduction (MST + their Node2Vec, V-measure)  (CLI)
  sarmah_improvements.py      §6a-bis — every technique on Sarmah's exact network; METHODS          (CLI)
  literature_comparison.py    V-measure (Sarmah) + SVM-F1 (Dolphin) over all saved vectorspaces     (CLI)
  matched_sampling.py         universe size-vs-composition confound: run_matched_node_count()       (CLI)
  lasso_experiment.py         §4a — Graphical Lasso vs correlation at matched density (small n)      (CLI)

jobs/                       DATA PREP (not needed to read results; only to rebuild data/)
  fetch_prices.py             Polygon → data/<universe>_<period>.csv                                (CLI)
  prep_kaggle_data.py         Kaggle 1970-2018 dump → data/dolphin/ + sarmah_2017_returns.csv       (CLI)
  build_gics_hierarchy.py     build/validate data/gics_hierarchy.csv                                (CLI)

data/      price panels + the 2 committed label files (rest gitignored; see "Data")
results/   outputs (gitignored): *.csv metrics tables + vectors/*.json saved vector spaces
trash/     legacy product code (Pinecone/FastAPI/old grid search), NOT part of the paper (gitignored)
```

The benchmark depends only on `train_vectors.py` + `benchmark/`; `jobs/` is just data prep and
`modal_app.py` is just cloud orchestration of the same code.

### Config knobs (`train_vectors.py::Config`, set per-run via `RunSpec`)

| Knob | Values | Controls |
|---|---|---|
| `embedder` | `node2vec`·`spectral`·`netmf`·`contrastive`·`raw_corr`·`raw_returns` | which representation (default node2vec) |
| `graph_filtering` | `threshold`·`mst`·`knn` | how the matrix becomes a graph |
| `repair_islands` / `min_degree` | bool / int | Forced Connectivity (threshold + degree repair) |
| `knn_k` | int | edges per node in the kNN graph |
| `mst_distance_transform` / `mst_extra_edges_pct` | bool / float | Mantegna-distance MST; PMFG-style extra edges |
| `use_precision_matrix` / `lasso_alpha` | bool / float | Graphical Lasso (partial correlation); α=0 ⇒ CV |
| `cross_sectional_demean` | bool | subtract each day's cross-sectional mean |
| `market_neutral` | bool | regress out a market factor (SPY/proxy) |
| `spectral_clean` / `spectral_k` | `none`·`market`·`topk`·`rmt` / int | PCA market-mode removal / RMT band cleaning |
| `walk_p` · `walk_q` · `walk_length` · `num_walks` · `window` | floats/ints | Node2Vec random-walk + skip-gram params |
| `dimensions` · `seed` | ints | embedding size · RNG seed (sweep many, report mean±std) |
| `training_period` · `cache_file` · `cutoff_date` | str | which price panel and trailing window |

### Method & embedder vocabulary (names used throughout RESULTS.md)

- **Graph construction** (`competitor_specs`): `mst` (Sarmah-style tree) · `pmfg_approx` (MST + 5% extra
  edges) · `knn` · `threshold_norepair` · `forced_connectivity` (FC = threshold + degree repair).
- **Co-movement removal** (`comovement_specs`, all on a kNN graph, only the *matrix* differs):
  `corr` (raw) · `demean` · `market_neutral` · `pca_market` · `rmt` · `lasso`.
- **Embedding families** (`embedding_methods.build_embeddings`): `node2vec` (random walk + skip-gram) ·
  `spectral` (Laplacian eigenmaps) · `netmf` (skip-gram as matrix factorization) · `contrastive`
  (Dolphin's method) · `raw_corr` / `raw_returns` (no-embedding baselines).

### Metrics at a glance

| Metric | What | File |
|---|---|---|
| hit-rate@k · success@k · MRR | do the k nearest neighbours share the GICS label? (**headline**) | `evaluation.py` |
| V-measure · NMI · ARI | KMeans-partition agreement with labels (Sarmah's language) | `clustering_metrics.py` |
| SVM-F1 (5-fold) | can an SVM read the sector off the vectors? (Dolphin's protocol) | `classification_metrics.py` |
| Wilcoxon · McNemar · bootstrap CI · BH-FDR | paired significance with effect sizes | `significance.py` |

### Modal entrypoints (`modal run modal_app.py::<name>`)

| Entrypoint | Experiment | RESULTS.md |
|---|---|---|
| `main --suite competitor\|comovement` | graph-construction / co-movement sweep, saves vectorspaces | §3, §4 |
| `families` | embedding-family comparison (node2vec/spectral/netmf/contrastive vs raw) | §6d |
| `dolphin` | our methods through Dolphin's SVM-sector protocol | §6b |
| `dim_sweep` | embedding-dimension robustness | §6f |
| `pq_sweep` | Node2Vec p/q sensitivity (provenance check) | §6f |
| `sarmah_density` | denser kNN/FC graphs on Sarmah's network | §6a-bis |
| `hedging` | out-of-sample peer-hedge risk experiment | §6g |

## Reproducibility notes

- All randomness is seeded (`Config.seed` → Node2Vec walks + Word2Vec init; `workers=1`).
- A harmless `Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'` line may
  print on some BLAS builds during interpreter shutdown; it does not affect results.

## Scope and limitations

- **Survivorship bias.** All universes are survivor panels: stocks with missing history or delistings are dropped before analysis. The results compare methods under clean, idealised conditions and are not a recommendation for live trading.
- **GICS is not ground truth.** We use GICS as a convenient reference taxonomy, but a company's economic identity can change while its GICS label stays fixed. A different reference dictionary would yield different "correct" neighbours.
- **Method comparison, not suitability.** The paper asks "which embedding representation recovers GICS structure best?" and "do embeddings beat raw-feature baselines?", not "should these embeddings be used in a trading strategy?". The out-of-sample hedging test is included precisely to show that representation quality does not automatically translate to portfolio improvement.
