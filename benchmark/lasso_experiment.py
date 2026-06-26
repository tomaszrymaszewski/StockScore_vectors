"""
Careful test of the Graphical Lasso (partial-correlation) graph vs the plain correlation graph.

Graphical Lasso is O(n^3) and slow, so this runs on a SMALL, well-conditioned subset (default
200 stocks of the S&P 500 5y panel, T/n ~ 6). It exercises the real `train_vectors` code path
(`GraphBuilder._compute_matrix` with `use_precision_matrix=True`, now normalized to partial
correlation) and answers two questions:

  Part 1 — DENSITY CONTROL: how the lasso `alpha` sparsifies the graph (higher alpha = fewer
           edges). This is the principled way to keep the partial-correlation graph from being
           over-connected, instead of an arbitrary threshold.

  Part 2 — QUALITY AT MATCHED DENSITY: correlation vs partial-correlation graphs thresholded to
           the SAME average degree, then repaired (Forced Connectivity) and embedded. Compares
           hit-rate@3 and V-measure at sector + sub-industry, averaged over seeds. Matched
           density isolates "which matrix gives a better graph", not "which is denser".

Run:  python -m benchmark.lasso_experiment            # n=200, sp500 5y
      python -m benchmark.lasso_experiment --n 150 --period 2y --seeds 0-2
"""
from __future__ import annotations

import argparse
import time

import networkx as nx
import numpy as np

from train_vectors import Config, EmbeddingTrainer, GraphBuilder
from benchmark import evaluation as ev
from benchmark.clustering_metrics import clustering_metrics
from benchmark.labels import load_all_levels
from benchmark.run_benchmark import RunSpec, fetch_returns


def subsample(returns, n, seed=0):
    rng = np.random.default_rng(seed)
    cols = sorted(map(str, rng.choice(list(returns.columns), size=min(n, returns.shape[1]),
                                      replace=False)))
    return returns[cols].dropna()


def threshold_for_degree(M, target_deg):
    """Pick |value| threshold so the graph has ~target average degree."""
    n = M.shape[0]
    A = np.abs(M).copy()
    np.fill_diagonal(A, 0.0)
    vals = A[np.triu_indices(n, 1)]
    vals = vals[vals > 1e-9]
    target_edges = int(target_deg * n / 2)
    if target_edges >= len(vals) or target_edges <= 0:
        return 0.0
    return float(np.partition(vals, len(vals) - target_edges)[len(vals) - target_edges])


def build_graph(M, cols, returns, cfg, target_deg):
    """Threshold M to the target degree, then apply the real Forced-Connectivity repair."""
    t = threshold_for_degree(M, target_deg)
    G = nx.Graph()
    G.add_nodes_from(cols)
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            w = abs(M[i, j])
            if w > t and np.isfinite(w):
                G.add_edge(cols[i], cols[j], weight=float(w))
    GraphBuilder(cfg)._rescue_islands(G, M, list(cols), returns)  # real repair code
    return G


def evaluate(emb, labels):
    out = {}
    for lv in ["sector", "sub_industry"]:
        fr = ev.neighbor_match_frame(emb, labels[lv], max_k=3)
        out[f"hit@3_{lv}"] = ev.hit_rate(fr, 3)
        out[f"V_{lv}"] = clustering_metrics(emb, labels[lv], seed=0).get("v_measure", float("nan"))
    return out


def base_cfg(dims, seed, **over):
    cfg = Config(skip_pinecone=True, seed=seed)
    cfg.dimensions = dims
    cfg.min_degree = 3
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def matrix_for(name, returns, gb, alpha, dims):
    """Build each co-movement-removal matrix via the real train_vectors code paths."""
    t0 = time.time()
    if name == "correlation":
        M = np.nan_to_num(returns.corr().values)
    elif name == "demean":
        M = np.nan_to_num(gb._cross_sectional_demean(returns).corr().values)
    elif name == "market_neutral":
        M = np.nan_to_num(gb._residualize_market(returns).corr().values)
    elif name == "pca_market":
        M = GraphBuilder(base_cfg(dims, 0, spectral_clean="market"))._compute_matrix(returns)
    elif name == "rmt":
        M = GraphBuilder(base_cfg(dims, 0, spectral_clean="rmt", spectral_k=1))._compute_matrix(returns)
    elif name == "lasso_pcorr":
        M = GraphBuilder(base_cfg(dims, 0, use_precision_matrix=True,
                                  lasso_alpha=alpha))._compute_matrix(returns)
    else:
        raise ValueError(name)
    return M, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="Graphical Lasso vs correlation graph (small-n)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--dims", type=int, default=64)
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--alpha", type=float, default=0.0, help="0=CV for the quality run")
    args = ap.parse_args()
    lo, hi = (args.seeds.split("-") + [args.seeds])[:2]
    seeds = list(range(int(lo), int(hi) + 1)) if "-" in args.seeds else [int(x) for x in args.seeds.split(",")]

    full = fetch_returns(RunSpec(universe=args.universe, period=args.period))
    returns = subsample(full, args.n)
    cols = list(returns.columns)
    labels = load_all_levels()
    print(f"\nsubset: {returns.shape[0]} days x {len(cols)} stocks "
          f"(T/n={returns.shape[0]/len(cols):.1f})")

    # --- Part 1: alpha controls density ---
    print("\n=== Part 1: lasso alpha -> graph density (fixed-alpha GraphicalLasso) ===")
    print(f"{'alpha':>7} {'fit_s':>7} {'nonzero_edges':>14} {'avg_degree':>11}")
    for alpha in [0.2, 0.4, 0.6]:
        cfg = base_cfg(args.dims, 0, use_precision_matrix=True, lasso_alpha=alpha)
        t0 = time.time()
        M = GraphBuilder(cfg)._compute_matrix(returns)
        dt = time.time() - t0
        A = np.abs(M).copy(); np.fill_diagonal(A, 0)
        e = int((A > 1e-9).sum() // 2)
        print(f"{alpha:>7.2f} {dt:>7.1f} {e:>14} {2*e/len(cols):>11.1f}")

    # --- Part 2: all co-movement-removal methods at MATCHED density ---
    print("\n=== Part 2: co-movement removal at MATCHED density ===")
    gb = GraphBuilder(base_cfg(args.dims, 0))
    method_names = ["correlation", "demean", "market_neutral", "pca_market", "rmt", "lasso_pcorr"]
    mats = {}
    print("matrix build times:")
    for name in method_names:
        M, dt = matrix_for(name, returns, gb, args.alpha, args.dims)
        mats[name] = M
        print(f"   {name:16} {dt:>7.2f}s")

    print(f"\n{'method':16} {'avgdeg':>7} {'hit@3_sec':>10} {'hit@3_sub':>10} "
          f"{'V_sec':>8} {'V_sub':>8}")
    print("-" * 66)
    for target_deg in [5, 10, 20]:
        print(f"-- target avg degree ~{target_deg} --")
        for name in method_names:
            M = mats[name]
            accs, real_deg = {}, None
            for s in seeds:
                cfg = base_cfg(args.dims, s)  # matrix already built; cfg only drives train+repair
                G = build_graph(M, cols, returns, cfg, target_deg)
                real_deg = 2 * G.number_of_edges() / G.number_of_nodes()
                emb = EmbeddingTrainer(cfg).train(G)
                for k, v in evaluate(emb, labels).items():
                    accs.setdefault(k, []).append(v)
            m = {k: float(np.mean(v)) for k, v in accs.items()}
            print(f"{name:16} {real_deg:>7.1f} {m['hit@3_sector']:>10.3f} "
                  f"{m['hit@3_sub_industry']:>10.3f} {m['V_sector']:>8.3f} {m['V_sub_industry']:>8.3f}")


if __name__ == "__main__":
    main()
