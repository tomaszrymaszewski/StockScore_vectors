"""
Does ANY embedding beat raw features? -- the point-2 generalization beyond Node2Vec.

The headline limitation in RESULTS.md ("the Node2Vec embedding does not clearly beat raw
correlation/returns") is, as stated, a claim about ONE embedding method. This script puts five
representations on the SAME graph substrate (kNN k=10 on the raw correlation matrix) and the
SAME three metric families, so the claim can be made about embeddings as a CLASS:

    node2vec     local random walks + skip-gram            (the original)
    spectral     Laplacian Eigenmaps (global spectral)     (different family)
    netmf        DeepWalk-as-matrix-factorization          (closed-form skip-gram)
    raw_corr     a stock = its correlation row             (no embedding)
    raw_returns  a stock = its standardized return series  (no embedding)

Metrics: retrieval hit-rate@k (sector + sub-industry), supervised SVM weighted-F1 (Dolphin's
protocol), and KMeans V-measure (Sarmah's). Seed-averaged (spectral/netmf/raw are deterministic
-> std 0; only node2vec carries walk noise). Each embedding is then paired-Wilcoxon tested
against the raw-feature nulls on per-stock hit@3, so "beats raw features" is significance-tested,
not eyeballed.

CLI:
    python -m benchmark.embedding_families --universe sp500 --period 5y --seeds 0-4
    python -m benchmark.embedding_families --universe sp500 --period 5y --save-vectors
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from benchmark import evaluation as ev
from benchmark import significance as sig
from benchmark.classification_metrics import svm_classification
from benchmark.clustering_metrics import clustering_metrics
from benchmark.embedding_methods import is_deterministic
from benchmark.labels import load_all_levels
from benchmark.run_benchmark import RunSpec, fetch_returns, train_embedding, _parse_seeds

# All families share one graph substrate so only the EMBEDDING differs. raw_*/contrastive ignore it.
FAMILIES = ["node2vec", "spectral", "netmf", "contrastive", "raw_corr", "raw_returns"]
HIT_LEVELS = ["sector", "sub_industry"]
SVM_LEVELS = ["sector", "sub_industry"]


def spec_for(embedder: str, universe: str, period: str, dimensions: int) -> RunSpec:
    """A kNN-k10/raw-correlation substrate spec with the chosen embedder."""
    return RunSpec(
        method=embedder, universe=universe, period=period, embedder=embedder,
        graph_mode="knn", knn_k=10, dimensions=dimensions,
    )


def score_embedding(emb: dict, labels: dict, seed: int) -> tuple[dict, pd.DataFrame]:
    """Full metric row for one `{ticker: vector}`: hit@{1,3,5} (sector+sub), SVM-F1
    (sector+sub), V-measure (sector+sub). Returns (metrics, sub_industry_match_frame).

    The single shared scorer for both the local sweep and the Modal fan-out, so a remote
    container and a local run produce identical numbers.
    """
    metrics, sub_frame = {}, None
    for lv in HIT_LEVELS:
        fr = ev.neighbor_match_frame(emb, labels[lv], max_k=max(ev.DEFAULT_KS))
        if lv == "sub_industry":
            sub_frame = fr
        for k in (1, 3, 5):
            metrics[f"hit@{k}_{lv}"] = ev.hit_rate(fr, k)
    for lv in SVM_LEVELS:
        metrics[f"svm_f1_{lv}"] = svm_classification(emb, labels[lv], seed=seed).get("f1_weighted")
    metrics["v_sector"] = clustering_metrics(emb, labels["sector"], seed=seed).get("v_measure")
    metrics["v_sub"] = clustering_metrics(emb, labels["sub_industry"], seed=seed).get("v_measure")
    return metrics, sub_frame


def run_family(
    embedder: str, universe: str, period: str, seeds: list[int], dimensions: int,
    returns: pd.DataFrame, labels: dict,
) -> tuple[dict, dict, dict]:
    """Return (aggregate_row, per_seed_subindustry_frames, seed0_embedding).

    Deterministic families are evaluated at a single seed (their output is seed-independent),
    so the expensive metrics are not recomputed five times for an identical vector space.
    """
    spec = spec_for(embedder, universe, period, dimensions)
    use_seeds = [seeds[0]] if is_deterministic(embedder) else seeds

    acc: dict[str, list[float]] = {}
    frames: list[pd.DataFrame] = []
    emb0 = None
    for s in use_seeds:
        emb, _ = train_embedding(spec, s, returns=returns)
        if emb0 is None:
            emb0 = emb
        m, sub_frame = score_embedding(emb, labels, s)
        frames.append(sub_frame)
        for key, val in m.items():
            acc.setdefault(key, []).append(val)

    return aggregate_row(embedder, acc, len(use_seeds), emb0), frames, emb0


def aggregate_row(embedder: str, acc: dict[str, list[float]], n_seeds: int, emb0: dict) -> dict:
    """Collapse per-seed metric lists into a {key_mean, key_std} row (shared local/Modal)."""
    row = {"method": embedder, "n_seeds": n_seeds,
           "dim": len(next(iter(emb0.values()))), "n_stocks": len(emb0)}
    for key, vals in acc.items():
        vv = [v for v in vals if v is not None]
        row[f"{key}_mean"] = float(np.mean(vv)) if vv else float("nan")
        row[f"{key}_std"] = float(np.std(vv, ddof=1)) if len(vv) > 1 else 0.0
    return row


def summarize(universe: str, period: str, dimensions: int, methods: list[str],
              rows: list[dict], frames_by_method: dict, embs: dict,
              out: str | None = None, save_vectors: bool = False) -> pd.DataFrame:
    """Print the family table + paired-Wilcoxon-vs-raw significance, save CSV (+ optional vectors).

    Shared by the local sweep and the Modal entrypoint so both emit identical output.
    `frames_by_method[m]` = list of per-seed sub-industry match frames; `embs[m]` = a seed-0
    embedding for optional persistence.
    """
    df = pd.DataFrame(rows)
    out = out or os.path.join("results", f"embedding_families_{universe}_{period}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)

    print("\n" + "=" * 92)
    print(f"EMBEDDING FAMILIES  |  {universe}/{period}  dim={dimensions}  "
          f"(graph: kNN k=10 on raw corr; raw_* use no graph)")
    print("=" * 92)
    print(f"{'method':12} {'sec hit@3':>11} {'sub hit@3':>11} "
          f"{'sec SVM-F1':>11} {'sub SVM-F1':>11} {'sec V':>8}")
    print("-" * 92)
    by_method = {r["method"]: r for r in rows}
    for m in methods:
        r = by_method.get(m)
        if r is None:
            continue
        print(f"{r['method']:12} "
              f"{r.get('hit@3_sector_mean',float('nan')):>7.3f}±{r.get('hit@3_sector_std',0):.3f} "
              f"{r.get('hit@3_sub_industry_mean',float('nan')):>7.3f}±{r.get('hit@3_sub_industry_std',0):.3f} "
              f"{r.get('svm_f1_sector_mean',float('nan')):>7.3f}±{r.get('svm_f1_sector_std',0):.3f} "
              f"{r.get('svm_f1_sub_industry_mean',float('nan')):>7.3f}±{r.get('svm_f1_sub_industry_std',0):.3f} "
              f"{r.get('v_sector_mean',float('nan')):>8.3f}")
    print("-" * 92)

    embed_methods = [m for m in methods if m not in ("raw_corr", "raw_returns")]
    for null in ("raw_returns", "raw_corr"):
        if null not in frames_by_method or not embed_methods:
            continue
        subset = {m: frames_by_method[m] for m in embed_methods + [null] if m in frames_by_method}
        res = sig.compare_methods(subset, reference=null, level="sub_industry", k=3)
        if res is None or res.empty:
            continue
        print(f"\nPaired Wilcoxon vs {null} (sub-industry hit@3; +delta = embedding beats raw):")
        cols = ["method_a", "delta", "p_value", "p_bh", "effect_size", "ci_low", "ci_high"]
        print(res[[c for c in cols if c in res.columns]].round(4).to_string(index=False))

    if save_vectors:
        vec_dir = os.path.join("results", "vectors")
        os.makedirs(vec_dir, exist_ok=True)
        for m, emb in embs.items():
            if emb is None:
                continue
            dim = len(next(iter(emb.values())))
            if dim > 512 and m.startswith("raw"):
                continue
            path = os.path.join(vec_dir, f"{m}_fam_{universe}_{period}.json")
            with open(path, "w") as f:
                json.dump({"schema": "stockscore.vectors.v1",
                           "spec": {"method": m, "universe": universe, "period": period,
                                    "embedder": m, "dimensions": dim},
                           "universe": universe, "period": period, "seeds": {"0": emb}}, f)
        print(f"   saved vector spaces -> {vec_dir}")

    print(f"\n📝 Saved {out}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Embedding-family comparison (does any beat raw features?)")
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--dimensions", type=int, default=64)
    ap.add_argument("--methods", default=",".join(FAMILIES))
    ap.add_argument("--save-vectors", action="store_true",
                    help="also persist each vector space to results/vectors/<m>_fam_<uni>_<per>.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    methods = [m for m in args.methods.split(",") if m]
    labels = load_all_levels()
    returns = fetch_returns(spec_for("node2vec", args.universe, args.period, args.dimensions))

    rows, frames_by_method, embs = [], {}, {}
    for m in methods:
        print(f"\n🧪 {m} ...")
        row, frames, emb0 = run_family(
            m, args.universe, args.period, seeds, args.dimensions, returns, labels)
        rows.append(row)
        frames_by_method[m] = frames
        embs[m] = emb0

    summarize(args.universe, args.period, args.dimensions, methods,
              rows, frames_by_method, embs, out=args.out, save_vectors=args.save_vectors)
    print("Reads: a positive, significant delta vs raw_returns/raw_corr means the embedding adds "
          "value; ~0 or negative means it is a (lossy) re-encoding of the raw features.")


if __name__ == "__main__":
    main()
