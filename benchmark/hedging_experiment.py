"""
Hedging / risk experiment — do embeddings improve a real portfolio decision out-of-sample?

A finance referee will ask whether the learned vectors have any economic value beyond
GICS retrieval.  This module implements a simple, transparent, OOS hedging test:

    For each stock i, find its top-k peers in the *train* window using a given representation
    (Node2Vec, raw correlation, etc.).  Short an equal-weighted basket of those k peers.
    Measure the annualised realised volatility of the residual r_i - r_basket in the
    disjoint *test* window.

The experiment is deliberately simple: it isolates whether the peer set selected by the
embedding is better at explaining future risk than a raw-correlation peer set.  If the
embedding does not beat the Pearson baseline, it is honest evidence that the paper's value
is representation/compression, not risk reduction.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from benchmark.embedding_methods import is_deterministic
from benchmark.labels import load_all_levels
from benchmark.run_benchmark import RunSpec, train_embedding, _parse_seeds
from benchmark.significance import paired_bootstrap_ci, correct_pvalues
from benchmark.temporal_validation import (
    load_prices, clean_returns, generate_windows, split_windows,
)

HEDGE_METHODS = ["node2vec", "spectral", "netmf", "raw_corr", "raw_returns", "corr_direct"]
_DETERMINISTIC = {"spectral", "netmf", "raw_corr", "raw_returns", "corr_direct"}


def _ordered_similarity(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix from a stacked representation."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    unit = vectors / norms
    return unit @ unit.T


def _topk_from_sim(sim: np.ndarray, k: int) -> np.ndarray:
    """Top-k neighbor indices per row (excluding self)."""
    sim = sim.copy()
    np.fill_diagonal(sim, -np.inf)
    k = min(k, sim.shape[1] - 1)
    part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    order = np.argsort(-sim[rows, part], axis=1)
    return part[rows, order]


def _train_similarity(
    method: str,
    spec_factory,
    ret_train: pd.DataFrame,
    seed: int,
    tickers: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_similarity, train_returns_aligned) for a method on the train window.

    For ``corr_direct`` the similarity is the raw Pearson correlation matrix.  For all other
    methods it is the cosine similarity of the representation vectors trained on the train
    window.
    """
    ret_train = ret_train[tickers]
    if method == "corr_direct":
        sim = np.nan_to_num(ret_train.corr().values)
        return sim, ret_train.values

    spec = spec_factory(method)
    emb, _ = train_embedding(spec, seed, returns=ret_train)
    X = np.asarray([emb[t] for t in tickers], dtype=np.float64)
    sim = _ordered_similarity(X)
    return sim, ret_train.values


def _hedge_residual_volatility(
    ret_test: pd.DataFrame,
    topk_idx: np.ndarray,
    tickers: list[str],
    min_obs: int = 30,
) -> dict[str, np.ndarray]:
    """Compute per-stock OOS residual volatility for an equal-weighted k-peer hedge.

    Returns arrays keyed by ``vol_unhedged``, ``vol_hedged``, ``vol_reduction``.
    Each entry is the annualised realised volatility (sqrt(252) * std) across the test
    window for one stock.  Stocks with insufficient data are dropped.
    """
    ret_test = ret_test[tickers]
    n = len(tickers)
    unhedged = np.full(n, np.nan)
    hedged = np.full(n, np.nan)
    reduction = np.full(n, np.nan)

    for i in range(n):
        stock = ret_test.iloc[:, i]
        peers = ret_test.iloc[:, topk_idx[i]]
        frame = pd.concat([stock, peers], axis=1).dropna()
        if len(frame) < min_obs:
            continue
        r_stock = frame.iloc[:, 0].to_numpy(dtype=float)
        r_peers = frame.iloc[:, 1:].to_numpy(dtype=float)
        r_hedge = r_peers.mean(axis=1)
        residual = r_stock - r_hedge
        unhedged[i] = np.sqrt(252) * float(r_stock.std(ddof=1))
        hedged[i] = np.sqrt(252) * float(residual.std(ddof=1))
        reduction[i] = 1.0 - hedged[i] / unhedged[i] if unhedged[i] > 1e-12 else np.nan

    return {"vol_unhedged": unhedged, "vol_hedged": hedged, "vol_reduction": reduction}


def _run_window(
    ret_train: pd.DataFrame,
    ret_test: pd.DataFrame,
    window_label: str,
    train_end: str,
    test_end: str,
    spec_factory,
    seeds: list[int],
    k: int,
    tickers: list[str],
    universe: str,
) -> tuple[list[dict], list[dict]]:
    """Run every hedging method on a single train/test window and return per-method rows.

    Also returns a list of per-stock records so that paired significance tests can be run
    on the same unit of analysis (individual stocks) as the rest of the paper.
    """
    rows = []
    per_stock_records = []

    # Unhedged baseline (same for every method, reported once per window)
    unhedged_vols = np.full(len(tickers), np.nan)
    for i, t in enumerate(tickers):
        s = ret_test[t].dropna()
        if len(s) >= 30:
            unhedged_vols[i] = np.sqrt(252) * float(s.std(ddof=1))
    rows.append({
        "method": "unhedged",
        "vol_hedged_mean": float(np.nanmean(unhedged_vols)),
        "vol_hedged_std": float(np.nanstd(unhedged_vols, ddof=1)),
        "vol_hedged_median": float(np.nanmedian(unhedged_vols)),
        "vol_reduction_mean": 0.0,
        "vol_reduction_median": 0.0,
        "n_stocks": int(np.sum(~np.isnan(unhedged_vols))),
    })
    for i, t in enumerate(tickers):
        if not np.isnan(unhedged_vols[i]):
            per_stock_records.append({
                "universe": universe, "window": window_label, "train_end": train_end,
                "test_end": test_end, "k": k, "method": "unhedged", "seed": 0,
                "ticker": t, "vol_unhedged": float(unhedged_vols[i]),
                "vol_hedged": float(unhedged_vols[i]), "vol_reduction": 0.0,
            })

    for method in HEDGE_METHODS:
        use_seeds = [seeds[0]] if method in _DETERMINISTIC else seeds
        vol_h, vol_u, red = [], [], []
        for s in use_seeds:
            sim, _ = _train_similarity(method, spec_factory, ret_train, s, tickers)
            topk = _topk_from_sim(sim, k)
            v = _hedge_residual_volatility(ret_test, topk, tickers)
            vol_h.append(np.nanmean(v["vol_hedged"]))
            vol_u.append(np.nanmean(v["vol_unhedged"]))
            red.append(np.nanmean(v["vol_reduction"]))
            for i, t in enumerate(tickers):
                if not np.isnan(v["vol_hedged"][i]):
                    per_stock_records.append({
                        "universe": universe, "window": window_label, "train_end": train_end,
                        "test_end": test_end, "k": k, "method": method, "seed": s,
                        "ticker": t, "vol_unhedged": float(v["vol_unhedged"][i]),
                        "vol_hedged": float(v["vol_hedged"][i]),
                        "vol_reduction": float(v["vol_reduction"][i]),
                    })
        rows.append({
            "method": method,
            "vol_hedged_mean": float(np.mean(vol_h)),
            "vol_hedged_std": float(np.std(vol_h, ddof=1)) if len(vol_h) > 1 else 0.0,
            "vol_hedged_median": float(np.nanmedian(np.asarray(vol_h))),
            "vol_reduction_mean": float(np.mean(red)),
            "vol_reduction_std": float(np.std(red, ddof=1)) if len(red) > 1 else 0.0,
            "vol_reduction_median": float(np.nanmedian(np.asarray(red))),
            "n_stocks": len(tickers),
        })

    for r in rows:
        r["window"] = window_label
        r["train_end"] = train_end
        r["test_end"] = test_end
        r["universe"] = universe
        r["k"] = k
    return rows, per_stock_records


def _aggregate(rows: list[dict]) -> pd.DataFrame:
    """Aggregate per-window rows to method-level mean ± std."""
    records = defaultdict(list)
    for r in rows:
        records[r["method"]].append(r)
    agg_rows = []
    for method, recs in records.items():
        base = {"method": method}
        for metric in ["vol_hedged_mean", "vol_hedged_median", "vol_reduction_mean", "vol_reduction_median"]:
            vals = [r[metric] for r in recs if not np.isnan(r[metric])]
            if vals:
                base[f"{metric}_mean"] = float(np.mean(vals))
                base[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            else:
                base[f"{metric}_mean"] = float("nan")
                base[f"{metric}_std"] = float("nan")
        agg_rows.append(base)
    return pd.DataFrame(agg_rows)


def compute_significance(per_stock_df: pd.DataFrame, baseline: str = "raw_returns") -> pd.DataFrame:
    """Paired Wilcoxon signed-rank + bootstrap CI for each method vs ``baseline``.

    The unit of analysis is the individual stock, paired across methods on the same
    (window, seed, ticker).  This mirrors the rest of the paper's significance machinery.
    """
    if per_stock_df.empty:
        return pd.DataFrame()
    rows = []
    for universe in per_stock_df["universe"].unique():
        sub = per_stock_df[per_stock_df["universe"] == universe]
        base = sub[sub["method"] == baseline]
        if base.empty:
            continue
        methods = [m for m in sub["method"].unique() if m != baseline]
        for method in methods:
            a = sub[sub["method"] == method][["window", "seed", "ticker", "vol_hedged"]]
            b = base[["window", "seed", "ticker", "vol_hedged"]]
            merged = pd.merge(a, b, on=["window", "seed", "ticker"], suffixes=("_m", "_b"))
            if merged.empty:
                continue
            diff = merged["vol_hedged_m"].to_numpy() - merged["vol_hedged_b"].to_numpy()
            diff = diff[np.isfinite(diff)]
            if len(diff) == 0:
                continue
            mean_m = float(merged["vol_hedged_m"].mean())
            mean_b = float(merged["vol_hedged_b"].mean())
            delta = float(diff.mean())
            if np.allclose(diff, 0, atol=1e-12):
                statv, p = 0.0, 1.0
            else:
                statv, p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            ci_low, ci_high = paired_bootstrap_ci(
                merged["vol_hedged_m"].to_numpy(),
                merged["vol_hedged_b"].to_numpy(),
                n_boot=2000, ci=0.95, seed=0,
            )
            rows.append({
                "universe": universe,
                "method": method,
                "baseline": baseline,
                "n": len(diff),
                "mean_method": mean_m,
                "mean_baseline": mean_b,
                "delta_mean": delta,
                "wilcoxon_stat": float(statv),
                "p_value": float(p),
                "ci_low": ci_low,
                "ci_high": ci_high,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    corrected = correct_pvalues(df["p_value"])
    df["p_holm"] = corrected["p_holm"].to_numpy()
    df["p_bh"] = corrected["p_bh"].to_numpy()
    df["significant_bh"] = df["p_bh"] < 0.05
    df["practically_meaningful"] = df["delta_mean"].abs() >= 0.01
    return df.sort_values(["universe", "delta_mean"]).reset_index(drop=True)


def run_universe(
    universe: str,
    period: str,
    train_period: str | None,
    test_horizon: str | None,
    num_windows: int,
    k: int,
    dimensions: int,
    seeds: list[int],
) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """Run the hedging experiment on one universe across all windows."""
    prices = load_prices(universe, period)
    labels_all = load_all_levels()
    sub_labels = labels_all["sub_industry"]

    def spec_factory(embedder: str) -> RunSpec:
        return RunSpec(
            method=embedder, universe=universe, period=period,
            embedder=embedder, graph_mode="knn", knn_k=10,
            dimensions=dimensions,
        )

    all_rows = []
    all_per_stock = []

    if train_period and test_horizon:
        windows = generate_windows(prices, train_period, test_horizon, num_windows)
    else:
        ret_train, ret_test, mid = split_windows(prices)
        windows = [(ret_train, ret_test, f"split_{mid}", mid, "")]

    for ret_train, ret_test, label, train_end, test_end in windows:
        tickers = [t for t in ret_train.columns if t in ret_test.columns and t in sub_labels]
        if len(tickers) < 50:
            print(f"  SKIP {label}: only {len(tickers)} common tickers")
            continue
        print(f"  ▶ {label}: {len(tickers)} stocks, train={len(ret_train)}d, test={len(ret_test)}d")
        rows, per_stock = _run_window(
            ret_train, ret_test, label, train_end, test_end,
            spec_factory, seeds, k, tickers, universe,
        )
        all_rows.extend(rows)
        all_per_stock.extend(per_stock)

    sig_df = compute_significance(pd.DataFrame(all_per_stock), baseline="raw_returns")
    return all_rows, all_per_stock, sig_df


def evaluate_hedging_window(payload: dict) -> dict:
    """Modal entrypoint: run one rolling train/test window through every hedging method.

    The payload mirrors the arguments used by ``main`` but targets a single window so the
    sweep can be fanned out across containers.
    """
    try:
        universe = payload["universe"]
        period = payload["period"]
        train_period = payload["train_period"]
        test_horizon = payload["test_horizon"]
        num_windows = payload["num_windows"]
        window_index = payload["window_index"]
        k = payload["k"]
        dimensions = payload["dimensions"]
        if "seed" in payload:
            seeds = [int(payload["seed"])]
        else:
            seeds = _parse_seeds(payload["seeds"])

        prices = load_prices(universe, period)
        labels_all = load_all_levels()
        sub_labels = labels_all["sub_industry"]

        windows = generate_windows(prices, train_period, test_horizon, num_windows)
        ret_train, ret_test, label, train_end, test_end = windows[window_index]

        tickers = [t for t in ret_train.columns if t in ret_test.columns and t in sub_labels]
        if len(tickers) < 50:
            return {"rows": [], "skipped": True, "reason": f"only {len(tickers)} common tickers"}

        def spec_factory(embedder: str) -> RunSpec:
            return RunSpec(
                method=embedder, universe=universe, period=period,
                embedder=embedder, graph_mode="knn", knn_k=10,
                dimensions=dimensions,
            )

        rows, per_stock = _run_window(
            ret_train, ret_test, label, train_end, test_end,
            spec_factory, seeds, k, tickers, universe,
        )
        return {"rows": rows, "per_stock": per_stock, "skipped": False, "universe": universe}
    except Exception as e:
        return {"rows": [], "per_stock": [], "skipped": True, "reason": f"{type(e).__name__}: {e}", "universe": payload.get("universe", "")}


def evaluate_hedging_universe(payload: dict) -> dict:
    """Modal entrypoint: run one full universe and return aggregate + significance results."""
    try:
        universe = payload["universe"]
        period = payload["period"]
        train_period = payload["train_period"]
        test_horizon = payload["test_horizon"]
        num_windows = payload["num_windows"]
        k = payload["k"]
        dimensions = payload["dimensions"]
        seeds = _parse_seeds(payload["seeds"])

        rows, per_stock, sig_df = run_universe(
            universe, period, train_period, test_horizon,
            num_windows, k, dimensions, seeds,
        )
        per_stock_csv = pd.DataFrame(per_stock).to_csv(index=False)
        return {
            "rows": rows,
            "per_stock_csv": per_stock_csv,
            "significance": sig_df.to_dict("records"),
            "universe": universe,
            "skipped": False,
        }
    except Exception as e:
        return {
            "rows": [], "per_stock_csv": "", "significance": [],
            "universe": payload.get("universe", ""), "skipped": True,
            "reason": f"{type(e).__name__}: {e}",
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Hedging / OOS risk experiment")
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--seeds", default="0-2", help="e.g. '0-4' or '0,1,2'")
    ap.add_argument("--dimensions", type=int, default=64)
    ap.add_argument("--k", type=int, default=10, help="Number of peer stocks in the hedge basket")
    ap.add_argument("--train-period", default="1y", help="Train-window length (e.g. 1y, 2y)")
    ap.add_argument("--test-horizon", default="1y", help="Test-window length (e.g. 1y)")
    ap.add_argument("--num-windows", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    k = args.k
    universe = args.universe
    period = args.period

    rows, per_stock, sig_df = run_universe(
        universe, period, args.train_period, args.test_horizon,
        args.num_windows, k, args.dimensions, seeds,
    )

    if not rows:
        print("No valid windows generated.")
        return

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(args.out) if args.out else "results"
    os.makedirs(out_dir or ".", exist_ok=True)
    base = args.out or os.path.join("results", f"hedging_{universe}_{period}_k{k}")
    per_window_path = f"{base}_per_window.csv"
    aggregate_path = f"{base}_aggregate.csv"
    per_stock_path = f"{base}_per_stock.csv"
    significance_path = f"{base}_significance.csv"

    df.to_csv(per_window_path, index=False)
    agg = _aggregate(rows)
    agg.to_csv(aggregate_path, index=False)
    pd.DataFrame(per_stock).to_csv(per_stock_path, index=False)
    sig_df.to_csv(significance_path, index=False)

    print("\n" + "=" * 100)
    print(f"HEDGING / OOS RISK  |  {universe}/{period}  k={k}  train={args.train_period}  test={args.test_horizon}")
    print("=" * 100)
    print(f"{'method':20} {'hedged vol mean':>16} {'hedged vol median':>18} {'risk reduction':>16}")
    print("-" * 100)
    for _, r in agg.iterrows():
        def fmt(x, w=12, p=2):
            return f"{x:>{w}.{p}f}" if pd.notna(x) else f"{'—':>{w}}"
        print(f"{r['method']:20} "
              f"{fmt(r['vol_hedged_mean_mean'], 16)} "
              f"{fmt(r['vol_hedged_median_mean'], 18)} "
              f"{fmt(r['vol_reduction_mean_mean'], 16)}")
    print("-" * 100)
    if not sig_df.empty:
        print("\nPaired Wilcoxon vs raw_returns baseline (per-stock hedged vol):")
        print(f"{'method':16} {'delta':>10} {'p (BH)':>10} {'CI95':>24} {'n':>10}")
        print("-" * 80)
        for _, r in sig_df.iterrows():
            ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
            print(f"{r['method']:16} {r['delta_mean']:>10.4f} {r['p_bh']:>10.4f} {ci:>24} {r['n']:>10}")
        print("-" * 80)
    print(f"Per-window rows:   {per_window_path}")
    print(f"Aggregate rows:    {aggregate_path}")
    print(f"Per-stock rows:    {per_stock_path}")
    print(f"Significance rows: {significance_path}")


if __name__ == "__main__":
    main()
