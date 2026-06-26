"""
Parameterized, Pinecone-free benchmark entrypoint.

One `RunSpec` -> train an embedding (or compute a baseline) -> score it along the full
GICS granularity ladder -> a `RunResult` carrying both aggregate metrics and the per-stock
match frames that the significance tests consume.

Design goals
------------
* **No Pinecone, no network.** Reuses `DataFetcher`/`GraphBuilder`/`EmbeddingTrainer` from
  `train_vectors.py` with `skip_pinecone=True`. The only external input is the data
  directory, taken from `$STOCKSCORE_DATA_DIR` (default "data"), so the exact same code
  runs unchanged inside a Modal container.
* **Seeds are first class.** `run_seeds` repeats a spec over many seeds and reports
  mean +/- std -- the prerequisite for every significance claim.
* **Competitors share the pipeline.** The raw-correlation baseline and the MST / kNN /
  PMFG / no-repair graph variants all flow through the identical scoring, so differences
  reflect the method, not the metric.

CLI:
    python -m benchmark.run_benchmark --universe r3000 --period 2y --seeds 0-9 --methods all
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from train_vectors import Config, DataFetcher, EmbeddingTrainer, GraphBuilder
from benchmark import evaluation as ev
from benchmark.labels import LEVELS, load_all_levels

DATA_DIR = os.environ.get("STOCKSCORE_DATA_DIR", "data")

_UNIVERSE_PREFIX = {"sp500": "sp500", "r1000": "russell_1000", "r3000": "russell_3000"}


def cache_file_for(universe: str, period: str) -> str:
    prefix = _UNIVERSE_PREFIX.get(universe)
    if prefix is None:
        raise ValueError(f"Unknown universe {universe!r}; expected one of {list(_UNIVERSE_PREFIX)}")
    return os.path.join(DATA_DIR, f"{prefix}_{period}.csv")


# ---------------------------------------------------------------------------
# Run specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    """A single benchmark configuration (everything except the seed)."""

    method: str = "forced_connectivity"  # display label for the row
    universe: str = "r3000"
    period: str = "2y"
    walk_p: float = 0.8
    walk_q: float = 7.2
    dimensions: int = 64
    threshold: float = 0.65
    graph_mode: str = "threshold"        # threshold | knn | mst
    repair_islands: bool = True
    min_degree: int = 3                   # Forced-Connectivity target min degree (sweepable)
    use_lasso: bool = False
    lasso_alpha: float = 0.0              # 0=GraphicalLassoCV; >0=fixed-alpha (higher=sparser)
    market_neutral: bool = False         # regress out market factor (SPY/proxy) before the matrix
    cross_sectional_demean: bool = False  # subtract each day's cross-sectional mean (free, O(nT))
    spectral_clean: str = "none"         # none | market | topk | rmt  (eigenvalue/RMT cleaning)
    spectral_k: int = 1                  # modes removed (market/topk) or dropped before RMT band
    knn_k: int = 10
    mst_extra_edges_pct: float = 0.0
    mst_distance_transform: bool = False  # True = Mantegna distance-transform MST (signed corr)
    walk_length: int = 20
    num_walks: int = 100
    window: int = 10
    embedder: str = "node2vec"           # node2vec | spectral | netmf | raw_corr | raw_returns

    def to_config(self, seed: int) -> Config:
        cfg = Config(training_period=self.period, skip_pinecone=True, seed=seed)
        cfg.cache_file = cache_file_for(self.universe, self.period)
        cfg.walk_p = float(self.walk_p)
        cfg.walk_q = float(self.walk_q)
        cfg.dimensions = int(self.dimensions)
        cfg.correlation_threshold = float(self.threshold)
        cfg.graph_filtering = self.graph_mode
        cfg.repair_islands = bool(self.repair_islands)
        cfg.min_degree = int(self.min_degree)
        cfg.use_precision_matrix = bool(self.use_lasso)
        cfg.lasso_alpha = float(self.lasso_alpha)
        cfg.market_neutral = bool(self.market_neutral)
        cfg.cross_sectional_demean = bool(self.cross_sectional_demean)
        cfg.spectral_clean = str(self.spectral_clean)
        cfg.spectral_k = int(self.spectral_k)
        cfg.knn_k = int(self.knn_k)
        cfg.mst_extra_edges_pct = float(self.mst_extra_edges_pct)
        cfg.mst_distance_transform = bool(self.mst_distance_transform)
        cfg.walk_length = int(self.walk_length)
        cfg.num_walks = int(self.num_walks)
        cfg.window = int(self.window)
        cfg.embedder = str(self.embedder)
        cfg.include_metadata = False
        return cfg

    def tag(self) -> dict:
        return {
            "method": self.method,
            "universe": self.universe,
            "period": self.period,
            "walk_p": self.walk_p,
            "walk_q": self.walk_q,
            "dimensions": self.dimensions,
            "threshold": self.threshold,
            "graph_mode": self.graph_mode,
            "repair_islands": self.repair_islands,
            "use_lasso": self.use_lasso,
            "embedder": self.embedder,
        }


@dataclass
class RunResult:
    """Outcome of scoring one representation against every GICS level."""

    spec: RunSpec
    seed: int | None
    n_tickers: int
    metrics: dict[str, ev.MetricBundle]              # level -> bundle
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)  # level -> per-stock matches

    def summary_row(self, ks: tuple[int, ...] = ev.DEFAULT_KS) -> dict:
        row = self.spec.tag()
        row["seed"] = self.seed
        row["n_tickers"] = self.n_tickers
        for level, bundle in self.metrics.items():
            for k in ks:
                row[f"{level}.hit_rate@{k}"] = bundle.hit_rate.get(k)
                row[f"{level}.success@{k}"] = bundle.success_at_k.get(k)
                row[f"{level}.mrr@{k}"] = bundle.mrr.get(k)
        return row


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def fetch_returns(spec: RunSpec) -> pd.DataFrame:
    cfg = spec.to_config(seed=0)
    return DataFetcher(cfg).fetch_returns(tickers=None)


def train_embedding(spec: RunSpec, seed: int, returns: pd.DataFrame | None = None):
    """Build an embedding for `spec` at `seed`. Returns (embeddings, returns).

    Dispatches on `spec.embedder` (node2vec | spectral | netmf | raw_corr | raw_returns) via
    `embedding_methods.build_embeddings`; graph-based families share the same GraphBuilder graph,
    so only the embedding step differs.
    """
    from benchmark.embedding_methods import build_embeddings

    cfg = spec.to_config(seed)
    if returns is None:
        returns = DataFetcher(cfg).fetch_returns(tickers=None)
    embeddings = build_embeddings(cfg, returns, seed)
    return embeddings, returns


def _score_all_levels(
    frame_fn, labels_by_level: dict[str, dict[str, str]], keep_frames: bool, ks
):
    metrics, frames = {}, {}
    for level in LEVELS:
        frame = frame_fn(labels_by_level[level])
        metrics[level] = ev.metric_bundle(frame, ks)
        if keep_frames:
            frames[level] = frame
    return metrics, frames


def evaluate_embedding(
    embeddings: dict,
    labels_by_level: dict[str, dict[str, str]],
    *,
    keep_frames: bool = False,
    ks: tuple[int, ...] = ev.DEFAULT_KS,
    max_k: int = max(ev.DEFAULT_KS),
):
    return _score_all_levels(
        lambda labels: ev.neighbor_match_frame(embeddings, labels, max_k),
        labels_by_level,
        keep_frames,
        ks,
    )


def run_single(
    spec: RunSpec,
    seed: int,
    labels_by_level: dict[str, dict[str, str]] | None = None,
    *,
    returns: pd.DataFrame | None = None,
    keep_frames: bool = False,
) -> RunResult:
    """Train + evaluate one spec at one seed."""
    if labels_by_level is None:
        labels_by_level = load_all_levels()
    embeddings, _ = train_embedding(spec, seed, returns=returns)
    metrics, frames = evaluate_embedding(embeddings, labels_by_level, keep_frames=keep_frames)
    n = len(metrics["sub_industry"].hit_rate) and metrics["sub_industry"].n_stocks
    return RunResult(spec=spec, seed=seed, n_tickers=n, metrics=metrics, frames=frames)


def run_raw_correlation(
    spec: RunSpec,
    labels_by_level: dict[str, dict[str, str]] | None = None,
    *,
    returns: pd.DataFrame | None = None,
    keep_frames: bool = False,
    absolute: bool = False,
) -> RunResult:
    """Raw-correlation top-k baseline (no Node2Vec) -- the reviewer's primary null.

    Uses the plain Pearson correlation of returns, independent of `use_lasso`, because the
    null hypothesis is explicitly *raw* correlation. Seed-independent.
    """
    if labels_by_level is None:
        labels_by_level = load_all_levels()
    if returns is None:
        returns = fetch_returns(spec)
    corr = np.nan_to_num(returns.corr().values)
    corr_tickers = list(returns.columns)
    metrics, frames = _score_all_levels(
        lambda labels: ev.raw_correlation_match_frame(
            corr, corr_tickers, labels, max_k=max(ev.DEFAULT_KS), absolute=absolute
        ),
        labels_by_level,
        keep_frames,
        ev.DEFAULT_KS,
    )
    rawspec = replace(spec, method="raw_correlation", graph_mode="none", repair_islands=False)
    return RunResult(rawspec, None, metrics["sub_industry"].n_stocks, metrics, frames)


def run_seeds(
    spec: RunSpec,
    seeds: list[int],
    labels_by_level: dict[str, dict[str, str]] | None = None,
    *,
    returns: pd.DataFrame | None = None,
    keep_frames: bool = False,
) -> list[RunResult]:
    """Repeat a spec across seeds (returns are fetched once and reused)."""
    if labels_by_level is None:
        labels_by_level = load_all_levels()
    if returns is None:
        returns = fetch_returns(spec)
    results = []
    for s in seeds:
        print(f"   ▶ {spec.method} seed={s}")
        results.append(
            run_single(spec, s, labels_by_level, returns=returns, keep_frames=keep_frames)
        )
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_over_seeds(results: list[RunResult], ks=ev.DEFAULT_KS) -> dict:
    """mean +/- std over seeds for each level/k metric. Replaces single-run reporting."""
    rows = [r.summary_row(ks) for r in results]
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if "@" in c]
    agg = {}
    base = results[0].spec.tag()
    agg.update(base)
    agg["n_seeds"] = len(results)
    agg["n_tickers"] = results[0].n_tickers
    for c in metric_cols:
        agg[f"{c}.mean"] = float(df[c].mean())
        agg[f"{c}.std"] = float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
    return agg


# ---------------------------------------------------------------------------
# Competitor-method registry
# ---------------------------------------------------------------------------


def competitor_specs(
    universe: str = "r3000",
    period: str = "2y",
    *,
    walk_p: float = 0.8,
    walk_q: float = 7.2,
    dimensions: int = 64,
    threshold: float = 0.65,
    knn_k: int = 10,
    use_lasso: bool = False,
) -> list[RunSpec]:
    """The graph-construction methods to compare under the identical hit-rate pipeline.

    (The raw-correlation baseline is run separately via `run_raw_correlation` since it
    involves no graph or Node2Vec.)
    """
    common = dict(
        universe=universe, period=period, walk_p=walk_p, walk_q=walk_q,
        dimensions=dimensions, use_lasso=use_lasso,
    )
    return [
        RunSpec(method="mst", graph_mode="mst", mst_extra_edges_pct=0.0, **common),
        RunSpec(method="pmfg_approx", graph_mode="mst", mst_extra_edges_pct=0.05, **common),
        RunSpec(method="knn", graph_mode="knn", knn_k=knn_k, **common),
        RunSpec(method="threshold_norepair", graph_mode="threshold",
                threshold=threshold, repair_islands=False, **common),
        RunSpec(method="forced_connectivity", graph_mode="threshold",
                threshold=threshold, repair_islands=True, **common),
    ]


def comovement_specs(
    universe: str = "sp500",
    period: str = "5y",
    *,
    walk_p: float = 0.8,
    walk_q: float = 7.2,
    dimensions: int = 64,
    knn_k: int = 10,
    include_lasso: bool = False,
    lasso_alpha: float = 0.4,
) -> list[RunSpec]:
    """Co-movement-removal methods to compare at scale (e.g. on Modal).

    All use a **kNN graph** (fixed k neighbours per node): this matches density across methods
    and is scale-invariant, so it sidesteps the fact that residual/partial-correlation matrices
    have much smaller entries than raw correlation (a fixed |value| threshold would otherwise
    give them near-empty graphs). Only the *matrix* differs between rows.

    `include_lasso` is off by default: Graphical Lasso is O(n^3) and slow on large universes (and
    ill-conditioned when T < n), so enable it deliberately.
    """
    common = dict(
        universe=universe, period=period, walk_p=walk_p, walk_q=walk_q,
        dimensions=dimensions, graph_mode="knn", knn_k=knn_k,
    )
    specs = [
        RunSpec(method="corr", **common),
        RunSpec(method="demean", cross_sectional_demean=True, **common),
        RunSpec(method="market_neutral", market_neutral=True, **common),
        RunSpec(method="pca_market", spectral_clean="market", **common),
        RunSpec(method="rmt", spectral_clean="rmt", spectral_k=1, **common),
    ]
    if include_lasso:
        specs.append(RunSpec(method="lasso", use_lasso=True, lasso_alpha=lasso_alpha, **common))
    return specs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_seeds(s: str) -> list[int]:
    if "-" in s and "," not in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description="StockScore embedding benchmark")
    ap.add_argument("--universe", default="r3000", choices=list(_UNIVERSE_PREFIX))
    ap.add_argument("--period", default="2y")
    ap.add_argument("--seeds", default="0-4", help="e.g. '0-9' or '0,1,2'")
    ap.add_argument("--methods", default="all",
                    help="'all', 'ours', or comma list of method names")
    ap.add_argument("--walk_p", type=float, default=0.8)
    ap.add_argument("--walk_q", type=float, default=7.2)
    ap.add_argument("--dimensions", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--out", default=os.path.join("results", "benchmark_summary.csv"))
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    labels = load_all_levels()
    returns_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def get_returns(spec: RunSpec) -> pd.DataFrame:
        key = (spec.universe, spec.period)
        if key not in returns_cache:
            returns_cache[key] = fetch_returns(spec)
        return returns_cache[key]

    all_specs = competitor_specs(
        args.universe, args.period, walk_p=args.walk_p, walk_q=args.walk_q,
        dimensions=args.dimensions, threshold=args.threshold,
    )
    if args.methods == "ours":
        specs = [s for s in all_specs if s.method == "forced_connectivity"]
    elif args.methods != "all":
        wanted = set(args.methods.split(","))
        specs = [s for s in all_specs if s.method in wanted]
    else:
        specs = all_specs

    rows = []

    # Raw-correlation baseline (seed-independent), reported once.
    base_spec = specs[0] if specs else RunSpec(universe=args.universe, period=args.period)
    print("🔬 Raw-correlation top-k baseline...")
    raw = run_raw_correlation(base_spec, labels, returns=get_returns(base_spec))
    rows.append({**raw.spec.tag(), "n_seeds": 0, "n_tickers": raw.n_tickers,
                 **{f"{c}.mean": v for c, v in raw.summary_row().items() if "@" in c}})

    for spec in specs:
        print(f"\n🧪 {spec.method} ({len(seeds)} seeds)...")
        results = run_seeds(spec, seeds, labels, returns=get_returns(spec))
        rows.append(aggregate_over_seeds(results))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    # Console summary: sub-industry & sector hit-rate@3 with std.
    print("\n" + "=" * 72)
    print(f"{'method':22} {'sub-ind hit@3':>16} {'sector hit@3':>16}")
    print("-" * 72)
    for _, r in df.iterrows():
        si = r.get("sub_industry.hit_rate@3.mean", r.get("sub_industry.hit_rate@3"))
        sistd = r.get("sub_industry.hit_rate@3.std", 0.0) or 0.0
        sec = r.get("sector.hit_rate@3.mean", r.get("sector.hit_rate@3"))
        secstd = r.get("sector.hit_rate@3.std", 0.0) or 0.0
        print(f"{r['method']:22} {si:>10.4f}±{sistd:<4.3f} {sec:>10.4f}±{secstd:<4.3f}")
    print("=" * 72)
    print(f"📝 Saved {args.out}")


if __name__ == "__main__":
    main()
