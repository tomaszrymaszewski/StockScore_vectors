"""Configuration for stock embedding training. All parameters have sane defaults."""

from dataclasses import dataclass


@dataclass
class Config:
    """Central control panel for embedding training.

    Every field can be passed as a keyword argument to
    :func:`stock_embeddings.train_embeddings`.
    """

    # -------------------------------------------------------------------------
    # Embedding family
    # -------------------------------------------------------------------------
    # "node2vec"     -> local random walks + skip-gram (default)
    # "spectral"     -> Laplacian Eigenmaps (global spectral geometry)
    # "netmf"        -> DeepWalk-as-matrix-factorization (closed-form skip-gram)
    # "raw_corr"     -> no embedding: a stock IS its correlation row
    # "raw_returns"  -> no embedding: a stock IS its standardized return series
    embedder: str = "node2vec"

    # Embedding dimensionality (ignored by raw_* embedders, which are data-shaped).
    dimensions: int = 64

    # Random seed for walks / Word2Vec init / eigensolvers.
    seed: int = 42

    # -------------------------------------------------------------------------
    # Graph construction
    # -------------------------------------------------------------------------
    # 'threshold'   = connect stocks if |correlation| > correlation_threshold
    # 'mst'         = maximum spanning tree backbone (+ optional extra edges)
    # 'knn'         = connect each stock to its K strongest-correlated peers (|corr|)
    # 'signed_knn'  = kNN ranked by SIGNED correlation (anti-correlated peers excluded)
    graph_filtering: str = "threshold"

    # Only used if graph_filtering == 'threshold'
    correlation_threshold: float = 0.65

    # Forced Connectivity: repair under-connected nodes to a minimum degree.
    repair_islands: bool = True
    min_degree: int = 3

    # Only used if graph_filtering in ('knn', 'signed_knn')
    knn_k: int = 10

    # Only used if graph_filtering == 'mst'
    mst_extra_edges_pct: float = 0.05   # 0.0 = pure MST; 0.05 = add top 5% unused edges
    mst_squared_corr: bool = True       # use corr^2 as edge weight
    # Mantegna (1999) distance-transform MST on SIGNED correlation; overrides the
    # mst_squared_corr / abs weighting above.
    mst_distance_transform: bool = False

    # -------------------------------------------------------------------------
    # Node2Vec walk parameters
    # -------------------------------------------------------------------------
    walk_p: float = 0.8                 # return parameter
    walk_q: float = 7.2                 # in-out parameter
    walk_length: int = 20
    num_walks: int = 100
    window: int = 10                    # skip-gram window (also NetMF window)

    # -------------------------------------------------------------------------
    # Similarity matrix estimation
    # -------------------------------------------------------------------------
    # True = Graphical Lasso (partial correlation); False = Pearson correlation.
    use_precision_matrix: bool = False
    # lasso_alpha = 0.0 -> GraphicalLassoCV (slow CV); > 0.0 -> fixed-alpha (fast).
    # With lasso, use a SMALL correlation_threshold (~0.02-0.1).
    lasso_alpha: float = 0.0

    # Market-neutral residualization: regress out a market factor and keep residuals.
    # Factor = `market_ticker` column if present, else equal-weighted average.
    market_neutral: bool = False
    market_ticker: str = "SPY"

    # Cross-sectional demeaning: subtract each day's equal-weighted mean return.
    cross_sectional_demean: bool = False

    # Spectral (RMT) cleaning of the correlation matrix:
    #   "none" | "market" (drop top eigenmode) | "topk" (drop top spectral_k)
    #   | "rmt" (keep only modes above the Marchenko-Pastur edge)
    spectral_clean: str = "none"
    spectral_k: int = 1

    # -------------------------------------------------------------------------
    # Data preparation
    # -------------------------------------------------------------------------
    # Drop tickers with less than this fraction of non-NaN observations.
    min_history_frac: float = 0.8
    # Minimum number of return observations required to train.
    min_rows: int = 30

    # Print progress logs.
    verbose: bool = True
