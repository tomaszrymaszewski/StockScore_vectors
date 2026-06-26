"""
Apply all our graph-construction + co-movement techniques to Sarmah's EXACT network and see which
beat their MST baseline.

Fixed (= Sarmah's setup): S&P 500 504 stocks, 2017 daily log returns, Node2Vec optimum
(l=100, r=50, p=2, q=0.5, w=5, dim=16). Only the graph/matrix construction varies. Evaluated with
Sarmah's V-measure at the 3 aligned GICS levels (avg compared to their 0.63) plus our hit-rate@3.

Methods: mst (Sarmah baseline) | threshold | fc (threshold+repair) | knn | demean | market_neutral
| pca_market | rmt | lasso (Graphical Lasso). Threshold-based graphs use a data-appropriate
correlation_threshold (daily return correlations rarely exceed ~0.6); edge counts are reported so
densities are transparent.

Run:  python -m benchmark.sarmah_improvements --seeds 0-4
"""
from __future__ import annotations

import argparse

import numpy as np

from train_vectors import Config, EmbeddingTrainer, GraphBuilder
from benchmark import evaluation as ev
from benchmark.clustering_metrics import clustering_metrics
from benchmark.labels import load_all_levels
from benchmark.reproduce_sarmah import load_kaggle_year_log_returns

ALIGNED = ["sector", "industry", "sub_industry"]   # Sarmah's 3 levels, aligned by K
THRESH = 0.40                                       # data-appropriate for daily-return correlations

METHODS = {
    "mst (Sarmah)":   dict(graph_filtering="mst", mst_distance_transform=True, mst_extra_edges_pct=0.0),
    "threshold":      dict(graph_filtering="threshold", correlation_threshold=THRESH, repair_islands=False),
    "fc":             dict(graph_filtering="threshold", correlation_threshold=THRESH, repair_islands=True),
    "knn":            dict(graph_filtering="knn", knn_k=10),
    "demean":         dict(graph_filtering="knn", knn_k=10, cross_sectional_demean=True),
    "market_neutral": dict(graph_filtering="knn", knn_k=10, market_neutral=True),
    "pca_market":     dict(graph_filtering="knn", knn_k=10, spectral_clean="market"),
    "rmt":            dict(graph_filtering="knn", knn_k=10, spectral_clean="rmt", spectral_k=1),
    "lasso (a=0.5)":  dict(graph_filtering="knn", knn_k=10, use_precision_matrix=True,
                           lasso_alpha=0.5),
}


def base_config(seed: int) -> Config:
    c = Config(skip_pinecone=True, seed=seed)
    c.dimensions = 16
    c.walk_p, c.walk_q = 2.0, 0.5
    c.walk_length, c.num_walks, c.window = 100, 50, 5
    return c


def _parse_seeds(s):
    if "-" in s and "," not in s:
        lo, hi = s.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description="Our techniques on Sarmah's network")
    ap.add_argument("--seeds", default="0-4")
    args = ap.parse_args()
    seeds = _parse_seeds(args.seeds)

    ret = load_kaggle_year_log_returns(year=2017, n=504)
    labels = load_all_levels()
    print(f"📂 Sarmah network: {ret.shape[1]} stocks x {ret.shape[0]} days "
          f"(log returns, Node2Vec optimum)\n")

    rows = []
    for name, flags in METHODS.items():
        accV = {lv: [] for lv in ALIGNED}
        hits, edges = [], []
        for s in seeds:
            c = base_config(s)
            for k, v in flags.items():
                setattr(c, k, v)
            G = GraphBuilder(c).build(ret)
            edges.append(G.number_of_edges())
            emb = EmbeddingTrainer(c).train(G)
            for lv in ALIGNED:
                accV[lv].append(clustering_metrics(emb, labels[lv], seed=s)["v_measure"])
            hits.append(ev.hit_rate(ev.neighbor_match_frame(emb, labels["sub_industry"], max_k=3), 3))
        rows.append({
            "method": name, "edges": int(np.mean(edges)),
            "V_sec": np.mean(accV["sector"]), "V_ind": np.mean(accV["industry"]),
            "V_sub": np.mean(accV["sub_industry"]),
            "V_avg": np.mean([np.mean(accV[lv]) for lv in ALIGNED]),
            "hit3_sub": np.mean(hits),
        })

    print(f"{'method':16}{'edges':>7}{'V_sec':>7}{'V_ind':>7}{'V_sub':>7}{'V_avg':>7}{'hit@3sub':>9}")
    print("-" * 69)
    base_avg = next(r["V_avg"] for r in rows if r["method"].startswith("mst"))
    for r in rows:
        d = r["V_avg"] - base_avg
        flag = f"  ({d:+.3f} vs MST)" if not r["method"].startswith("mst") else "  (baseline)"
        print(f"{r['method']:16}{r['edges']:>7}{r['V_sec']:>7.3f}{r['V_ind']:>7.3f}"
              f"{r['V_sub']:>7.3f}{r['V_avg']:>7.3f}{r['hit3_sub']:>9.3f}{flag}")
    print("-" * 69)
    print("Sarmah (paper): V_sec 0.35  V_ind 0.70  V_sub 0.84  V_avg 0.63")


if __name__ == "__main__":
    main()
