"""
Direct comparison to Dolphin et al. (arXiv:2407.18645) on their public benchmark dataset.

Runs OUR methods through their exact evaluation protocol -- SVM + 5-fold CV sector classification
(weighted F1 / accuracy) -- on the Dolphin/Kaggle universe, so each method can be read straight
off their Table 1 leaderboard (611 US stocks, 2000-2018, 11 NASDAQ-taxonomy sectors):

    Dolphin et al. [9]  (2202.08968)        F1 0.60*   acc 60*
    Sarmah et al. [25]  (node2vec on corr)  F1 0.61*   acc 61*
    Dolphin et al. [10] (2305.00245)        F1 0.66*   acc 65*
    Contrastive (Sigmoid-Softmax, best)     F1 0.69    acc 69%
(* = best-effort reimplementation by Dolphin; only the contrastive row is a method's own number.)

Methods compared (all kNN graphs except `fc`, isolating the co-movement matrix):
    fc | corr | demean | market_neutral | pca_market | rmt | lasso

DATA: data/dolphin/{returns_df_611.csv, historical_stocks.csv}, reconstructed from Kaggle via
`jobs.prep_kaggle_data --dolphin`. Our full-period-survivor set is ~1,113 stocks (Dolphin's exact
611 filter is unknown), i.e. ~2x their universe and therefore a harder, conservative comparison.
Runs locally (data/dolphin is excluded from the Modal image).

Run:  python -m benchmark.dolphin_benchmark --seeds 0-4 --dimensions 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from train_vectors import Config
from benchmark import evaluation as ev
from benchmark.classification_metrics import svm_classification
from benchmark.clustering_metrics import clustering_metrics
from benchmark.embedding_methods import build_embeddings

DATA_DIR = os.path.join("data", "dolphin")
RETURNS_PATH = os.path.join(DATA_DIR, "returns_df_611.csv")
STOCKS_PATH = os.path.join(DATA_DIR, "historical_stocks.csv")
LEVELS = ["sector", "industry"]          # the NASDAQ-taxonomy levels in historical_stocks.csv
BAD_SECTORS = {None, "MISCELLANEOUS", "SECTOR"}

METHOD_FLAGS = {
    # Node2Vec on various correlation-graph constructions (embedder defaults to node2vec).
    "fc":             dict(graph_filtering="threshold", correlation_threshold=0.65, repair_islands=True),
    "corr":           dict(graph_filtering="knn", knn_k=10),
    "demean":         dict(graph_filtering="knn", knn_k=10, cross_sectional_demean=True),
    "market_neutral": dict(graph_filtering="knn", knn_k=10, market_neutral=True),
    "pca_market":     dict(graph_filtering="knn", knn_k=10, spectral_clean="market"),
    "rmt":            dict(graph_filtering="knn", knn_k=10, spectral_clean="rmt", spectral_k=1),
    "lasso":          dict(graph_filtering="knn", knn_k=10, use_precision_matrix=True,
                           lasso_alpha=0.5),
    # Other embedding families (no correlation graph): Dolphin's own contrastive method, and
    # the no-embedding raw-feature baselines that are the honest yardstick on THIS dataset.
    "contrastive":    dict(embedder="contrastive"),
    "spectral":       dict(graph_filtering="knn", knn_k=10, embedder="spectral"),
    "netmf":          dict(graph_filtering="knn", knn_k=10, embedder="netmf"),
    "raw_corr":       dict(embedder="raw_corr"),
    "raw_returns":    dict(embedder="raw_returns"),
}


def load_dolphin_data():
    if not (os.path.exists(RETURNS_PATH) and os.path.exists(STOCKS_PATH)):
        sys.exit(f"❌ Dolphin data not found. Build it with:\n"
                 f"   python -m jobs.prep_kaggle_data --start 2000-01-01 --end 2018-12-31 "
                 f"--min-coverage 1.0 --dolphin")
    returns = pd.read_csv(RETURNS_PATH, index_col=0)
    returns.index = pd.to_datetime(returns.index)
    stocks = pd.read_csv(STOCKS_PATH)
    lvl_maps = {lv: dict(zip(stocks["ticker"], stocks[lv])) for lv in LEVELS}

    keep = [t for t in returns.columns if lvl_maps["sector"].get(t) not in BAD_SECTORS]
    returns = returns[keep].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    labels = {lv: {t: lvl_maps[lv][t] for t in returns.columns if t in lvl_maps[lv]}
              for lv in LEVELS}
    print(f"📂 Dolphin set: {returns.shape[1]} stocks x {returns.shape[0]} days, "
          f"{len(set(labels['sector'].values()))} sectors  (Dolphin used 611)")
    return returns, labels


def make_config(method: str, seed: int, dims: int) -> Config:
    cfg = Config(skip_pinecone=True, seed=seed)
    cfg.dimensions = int(dims)
    cfg.include_metadata = False
    for k, v in METHOD_FLAGS[method].items():
        setattr(cfg, k, v)
    return cfg


def evaluate(emb: dict, labels: dict, seed: int) -> dict:
    out = {}
    for lv in LEVELS:
        fr = ev.neighbor_match_frame(emb, labels[lv], max_k=3)
        sv = svm_classification(emb, labels[lv], seed=seed)
        out[f"svm_f1_{lv}"] = sv.get("f1_weighted")
        out[f"svm_acc_{lv}"] = sv.get("accuracy")
        out[f"hit3_{lv}"] = ev.hit_rate(fr, 3)
    out["v_sector"] = clustering_metrics(emb, labels["sector"], seed=seed).get("v_measure")
    return out


def _parse_seeds(s: str) -> list[int]:
    if "-" in s and "," not in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description="Direct Dolphin-protocol comparison of our methods")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--dimensions", type=int, default=16, help="match Dolphin's d=16")
    ap.add_argument("--methods",
                    default="raw_corr,raw_returns,contrastive,corr,pca_market,spectral,netmf",
                    help="any of: raw_corr raw_returns contrastive corr pca_market spectral netmf "
                         "fc demean market_neutral rmt lasso")
    ap.add_argument("--out", default=os.path.join("results", "dolphin_comparison.csv"))
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    methods = args.methods.split(",")
    returns, labels = load_dolphin_data()
    vec_dir = os.path.join("results", "vectors")
    os.makedirs(vec_dir, exist_ok=True)

    rows = []
    for method in methods:
        accs, seeds_map = defaultdict(list), {}
        for s in seeds:
            print(f"   ▶ {method} seed={s}", file=sys.stderr)
            cfg = make_config(method, s, args.dimensions)
            emb = build_embeddings(cfg, returns, s)
            seeds_map[str(s)] = {t: [float(x) for x in v] for t, v in emb.items()}
            for k, v in evaluate(emb, labels, s).items():
                if v is not None:
                    accs[k].append(v)
        # Save the vectorspace (build_vectors format). Skip the raw-feature baselines: their
        # vectors are N-dimensional (tens of MB) and instant to recompute from the returns.
        if not method.startswith("raw"):
            with open(os.path.join(vec_dir, f"{method}_dolphin.json"), "w") as f:
                json.dump({"schema": "stockscore.vectors.v1",
                           "spec": {"method": method, "universe": "dolphin", "period": "2000-2018"},
                           "universe": "dolphin", "period": "2000-2018", "seeds": seeds_map}, f)
        row = {"method": method, "n_seeds": len(seeds)}
        for k, vals in accs.items():
            row[f"{k}_mean"] = float(np.mean(vals))
            row[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 78)
    print(f"DIRECT DOLPHIN COMPARISON  ({returns.shape[1]} stocks, SVM 5-fold, dim={args.dimensions}, "
          f"{len(seeds)} seeds)")
    print("=" * 78)
    print(f"{'method':16} {'SVM-F1 sector':>16} {'SVM-acc sector':>16} {'hit@3 sec':>11} {'SVM-F1 ind':>11}")
    print("-" * 78)
    for r in rows:
        print(f"{r['method']:16} "
              f"{r.get('svm_f1_sector_mean',0):.3f}±{r.get('svm_f1_sector_std',0):.3f}    "
              f"{r.get('svm_acc_sector_mean',0):.3f}±{r.get('svm_acc_sector_std',0):.3f}    "
              f"{r.get('hit3_sector_mean',0):>11.3f} {r.get('svm_f1_industry_mean',0):>11.3f}")
    print("-" * 78)
    print("Dolphin Table 1 (their 611-stock data): Dolphin[9] 0.60 | Sarmah 0.61 | "
          "Dolphin[10] 0.66 | Contrastive 0.69 (sector F1)")
    print(f"📝 Saved {args.out}")


if __name__ == "__main__":
    main()
