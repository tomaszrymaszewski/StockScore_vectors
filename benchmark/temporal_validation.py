"""
Out-of-sample / temporal validation -- does past structure predict the FUTURE, and does an
embedding (a denoised, low-rank view) generalize where raw correlation overfits sample noise?

Every other result in the repo is single-period, in-sample label recovery. The reviewer's
sharpest question is temporal: raw correlation fits the sample (and wins in-sample), but it also
fits the sample's NOISE. A 64-dim embedding cannot represent that noise, so its real shot at
beating raw features is out of sample. We test that directly, alignment-free (so independent
embeddings never have to be rotated into a common frame):

  * split each universe's history into two DISJOINT windows A (past) and B (future);
  * build every representation on A only;
  * **future co-movement** -- for each stock take its top-k A-neighbors, then measure their
    REALIZED correlation in window B. A representation whose A-neighbors still co-move in B
    captured durable structure; one that fit A's noise regresses to the mean. `corr_direct`
    (rank by raw corr_A) is the null; `oracle_B` (rank by corr_B) is the unreachable ceiling.
  * **label-recovery stability** -- GICS hit@3 built on A vs built on B (labels are static), to
    show whether the raw-vs-embedding ordering is a one-window fluke or holds across time.

CLI:
    python -m benchmark.temporal_validation --universe sp500 --period 5y --seeds 0-2
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from benchmark import evaluation as ev
from benchmark.embedding_methods import is_deterministic
from benchmark.labels import load_all_levels
from benchmark.run_benchmark import RunSpec, cache_file_for, train_embedding, _parse_seeds

GRAPH_FAMILIES = ["node2vec", "spectral", "netmf"]   # built on A's graph
RAW_FAMILIES = ["raw_corr", "raw_returns"]            # feature vectors from A


def load_prices(universe: str, period: str) -> pd.DataFrame:
    path = cache_file_for(universe, period)
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index).normalize()
    df = df.groupby(df.index).last().sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def clean_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """pct-change returns with the same hygiene as DataFetcher (drop flat / non-finite)."""
    sub = prices.dropna(how="all").ffill().dropna(axis=1, thresh=int(len(prices) * 0.6))
    rets = sub.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return rets.loc[:, rets.std() > 1e-8]


def split_windows(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Disjoint past/future halves at the median date."""
    dates = prices.index
    mid = dates[len(dates) // 2]
    A = clean_returns(prices.loc[:mid])
    B = clean_returns(prices.loc[mid:])
    return A, B, str(mid.date())


def _ordered_matrix(emb: dict[str, list[float]], tickers: list[str]) -> np.ndarray:
    return np.asarray([emb[t] for t in tickers], dtype=np.float64)


def _cosine_topk(X: np.ndarray, k: int) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    sim = (X / norms) @ (X / norms).T
    return _topk_from_sim(sim, k)


def _topk_from_sim(sim: np.ndarray, k: int) -> np.ndarray:
    # A stock is never its own neighbor: mask the diagonal so self-similarity (1.0 on a
    # correlation/cosine matrix) cannot be selected -- otherwise every stock trivially
    # "matches its own label" and both hit-rate and future-corr are inflated.
    np.fill_diagonal(sim, -np.inf)
    k = min(k, sim.shape[1] - 1)
    part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    order = np.argsort(-sim[rows, part], axis=1)
    return part[rows, order]


def mean_future_corr(topk_idx: np.ndarray, corr_future: np.ndarray) -> float:
    """Mean realized future-window correlation of each stock's top-k (past) neighbors."""
    rows = np.arange(topk_idx.shape[0])[:, None]
    vals = corr_future[rows, topk_idx]          # (n, k) future corr of past neighbors
    return float(np.nanmean(vals))


def gics_hit3_from_topk(topk_idx: np.ndarray, label_arr: np.ndarray) -> float:
    nbr = label_arr[topk_idx]
    return float((nbr == label_arr[:, None]).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Temporal / out-of-sample validation")
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--dimensions", type=int, default=64)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    k = args.k
    prices = load_prices(args.universe, args.period)
    retA, retB, mid = split_windows(prices)
    labels_all = load_all_levels()
    sub_labels = labels_all["sub_industry"]

    # Common tickers present in both windows and labeled, in a fixed order.
    tickers = [t for t in retA.columns if t in retB.columns and t in sub_labels]
    retA, retB = retA[tickers], retB[tickers]
    label_arr = np.asarray([sub_labels[t] for t in tickers], dtype=object)
    corrA = np.nan_to_num(retA.corr().values)
    corrB = np.nan_to_num(retB.corr().values)
    print(f"📂 {args.universe}/{args.period}: {len(tickers)} common stocks | "
          f"A={retA.shape[0]}d (≤{mid})  B={retB.shape[0]}d (>{mid})")

    def spec(embedder: str) -> RunSpec:
        return RunSpec(method=embedder, universe=args.universe, period=args.period,
                       embedder=embedder, graph_mode="knn", knn_k=10, dimensions=args.dimensions)

    rows = []

    # --- Reference null + oracle (rank directly by the correlation matrix) ---
    topk_corrA = _topk_from_sim(corrA.copy(), k)
    topk_corrB = _topk_from_sim(corrB.copy(), k)
    rows.append({"method": "corr_direct (null)", "future_corr": mean_future_corr(topk_corrA, corrB),
                 "insample_corr": mean_future_corr(topk_corrA, corrA),
                 "gics_hit3_A": gics_hit3_from_topk(topk_corrA, label_arr),
                 "gics_hit3_B": gics_hit3_from_topk(topk_corrB, label_arr)})
    rows.append({"method": "oracle_B (ceiling)", "future_corr": mean_future_corr(topk_corrB, corrB),
                 "insample_corr": float("nan"), "gics_hit3_A": float("nan"),
                 "gics_hit3_B": float("nan")})

    # --- Each representation: build on A (and on B for the stability column) ---
    for m in GRAPH_FAMILIES + RAW_FAMILIES:
        print(f"   ▶ {m}")
        use_seeds = [seeds[0]] if is_deterministic(m) else seeds
        fut, ins, hitA, hitB = [], [], [], []
        for s in use_seeds:
            embA, _ = train_embedding(spec(m), s, returns=retA)
            embB, _ = train_embedding(spec(m), s, returns=retB)
            XA = _ordered_matrix(embA, tickers)
            XB = _ordered_matrix(embB, tickers)
            tkA = _cosine_topk(XA, k)
            tkB = _cosine_topk(XB, k)
            fut.append(mean_future_corr(tkA, corrB))
            ins.append(mean_future_corr(tkA, corrA))
            hitA.append(gics_hit3_from_topk(tkA, label_arr))
            hitB.append(gics_hit3_from_topk(tkB, label_arr))
        rows.append({"method": m, "future_corr": float(np.mean(fut)),
                     "insample_corr": float(np.mean(ins)),
                     "gics_hit3_A": float(np.mean(hitA)), "gics_hit3_B": float(np.mean(hitB))})

    df = pd.DataFrame(rows)
    df["retention"] = df["future_corr"] / df["insample_corr"]
    out = args.out or os.path.join("results", f"temporal_{args.universe}_{args.period}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)

    print("\n" + "=" * 88)
    print(f"TEMPORAL / OUT-OF-SAMPLE  |  {args.universe}/{args.period}  (split {mid}, k={k})")
    print("=" * 88)
    print(f"{'method':20} {'GICS hit@3 A':>12} {'GICS hit@3 B':>12} "
          f"{'insample ρ':>11} {'future ρ':>10} {'retention':>10}")
    print("-" * 88)
    for _, r in df.iterrows():
        def f(x, w=12, p=3):
            return f"{x:>{w}.{p}f}" if pd.notna(x) else f"{'—':>{w}}"
        print(f"{r['method']:20} {f(r['gics_hit3_A'])} {f(r['gics_hit3_B'])} "
              f"{f(r['insample_corr'],11)} {f(r['future_corr'],10)} {f(r['retention'],10)}")
    print("-" * 88)
    print("future ρ = realized window-B correlation of each stock's top-k window-A neighbors.")
    print("Higher future ρ / retention = the representation captured DURABLE structure, not "
          "in-sample noise. If an embedding's future ρ > corr_direct's, it generalizes better.")
    print(f"\n📝 Saved {out}")


if __name__ == "__main__":
    main()
