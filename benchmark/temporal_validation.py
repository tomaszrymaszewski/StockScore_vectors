"""
Out-of-sample / temporal validation -- multiple rolling windows, multiple train/test
periods, and aggregate statistics.

Design
------
Uses the full history loaded by `--period` (e.g. 5y) as a single panel, then generates
rolling train/test windows of requested lengths.  For each window we:

  * build every representation on the train window only;
  * measure its nearest-neighbour GICS hit@3 in the train window (in-sample) and
    again in the test window (out-of-sample, labels are static);
  * measure the realized future correlation of each stock's top-k train neighbours
    inside the test window (`future_corr`).

The old single-split behaviour is preserved when no `--train-periods` or
`--test-horizons` are given.

CLI examples
------------
    # Old default: single median split
    python -m benchmark.temporal_validation --universe sp500 --period 5y --seeds 0-2

    # Rolling windows: 5 train/test pairs, 1y train, 1y test
    python -m benchmark.temporal_validation --universe sp500 --period 5y \
            --train-periods 1y --test-horizons 1y --num-windows 5 --seeds 0-4

    # Sweep over several train/test lengths and aggregate
    python -m benchmark.temporal_validation --universe sp500 --period 5y \
            --train-periods 1y,2y --test-horizons 1y,2y --num-windows 5 --seeds 0-4
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from benchmark import evaluation as ev
from benchmark.embedding_methods import is_deterministic
from benchmark.labels import load_all_levels
from benchmark.run_benchmark import RunSpec, cache_file_for, train_embedding, _parse_seeds

GRAPH_FAMILIES = ["node2vec", "spectral", "netmf"]
RAW_FAMILIES = ["raw_corr", "raw_returns"]
ALL_FAMILIES = GRAPH_FAMILIES + RAW_FAMILIES

# Approximate trading days per period.  Used only to convert period strings into a
# number of rows; the actual windows are anchored to real dates in the index.
_TRADING_DAYS = {
    "0.5y": 126,
    "1y": 252,
    "2y": 504,
    "5y": 1260,
}


def load_prices(universe: str, period: str) -> pd.DataFrame:
    path = cache_file_for(universe, period)
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index).normalize()
    df = df.groupby(df.index).last().sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def clean_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """pct-change returns with the same hygiene as DataFetcher."""
    sub = prices.dropna(how="all").ffill().dropna(axis=1, thresh=int(len(prices) * 0.6))
    rets = sub.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return rets.loc[:, rets.std() > 1e-8]


def _period_to_days(period: str) -> int:
    """Convert a period token like '1y', '2y', '0.5y' into approximate trading days."""
    p = period.lower().strip()
    if p in _TRADING_DAYS:
        return _TRADING_DAYS[p]
    # Try to parse simple numeric strings, e.g. "252d"
    if p.endswith("d") and p[:-1].isdigit():
        return int(p[:-1])
    raise ValueError(f"Unknown period {period!r}; expected one of {list(_TRADING_DAYS)} or '<int>d'")


def generate_windows(
    prices: pd.DataFrame,
    train_period: str,
    test_horizon: str,
    num_windows: int,
    min_test_days: int = 60,
) -> list[tuple[pd.DataFrame, pd.DataFrame, str, str, str]]:
    """
    Return a list of (train_returns, test_returns, split_label, train_end, test_end).
    Windows are rolling and evenly spaced across the available history.  The test set
    always follows the train set without overlap.
    """
    returns = clean_returns(prices)
    total_days = len(returns)
    train_days = _period_to_days(train_period)
    test_days = _period_to_days(test_horizon)

    if train_days + test_days > total_days:
        raise ValueError(
            f"Not enough history for train={train_period} + test={test_horizon} "
            f"({train_days + test_days} days needed, {total_days} available)"
        )
    if test_days < min_test_days:
        raise ValueError(f"Test horizon too short ({test_days}d < {min_test_days}d minimum)")

    # Rolling step: evenly space the first train day so that the last window ends near the end.
    usable = total_days - train_days - test_days
    if num_windows == 1:
        offsets = [0]
    else:
        offsets = [int(round(i * usable / (num_windows - 1))) for i in range(num_windows)]

    windows = []
    for i, offset in enumerate(offsets, 1):
        train_start = offset
        train_end = train_start + train_days
        test_start = train_end
        test_end = test_start + test_days

        ret_train = returns.iloc[train_start:train_end]
        ret_test = returns.iloc[test_start:test_end]

        train_start_date = str(ret_train.index[0].date())
        train_end_date = str(ret_train.index[-1].date())
        test_end_date = str(ret_test.index[-1].date())
        label = (
            f"w{i:02d}_train[{train_start_date}:{train_end_date}]_test[{test_end_date}]"
        )
        windows.append((ret_train, ret_test, label, train_end_date, test_end_date))

    return windows


def split_windows(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Disjoint past/future halves at the median date (legacy single-split behaviour)."""
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
    np.fill_diagonal(sim, -np.inf)
    k = min(k, sim.shape[1] - 1)
    part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    order = np.argsort(-sim[rows, part], axis=1)
    return part[rows, order]


def mean_future_corr(topk_idx: np.ndarray, corr_future: np.ndarray) -> float:
    """Mean realized future-window correlation of each stock's top-k (past) neighbors."""
    rows = np.arange(topk_idx.shape[0])[:, None]
    vals = corr_future[rows, topk_idx]
    return float(np.nanmean(vals))


def gics_hit3_from_topk(topk_idx: np.ndarray, label_arr: np.ndarray) -> float:
    nbr = label_arr[topk_idx]
    return float((nbr == label_arr[:, None]).mean())


def _run_window(
    ret_train: pd.DataFrame,
    ret_test: pd.DataFrame,
    window_label: str,
    train_end: str,
    test_end: str,
    spec: callable,
    seeds: list[int],
    k: int,
    label_arr: np.ndarray,
    tickers: list[str],
) -> list[dict]:
    """Run every method on a single train/test window and return per-method rows."""
    corr_train = np.nan_to_num(ret_train.corr().values)
    corr_test = np.nan_to_num(ret_test.corr().values)

    rows = []

    # Reference null + oracle (rank directly by the correlation matrix)
    topk_corr_train = _topk_from_sim(corr_train.copy(), k)
    topk_corr_test = _topk_from_sim(corr_test.copy(), k)
    rows.append({
        "method": "corr_direct (null)",
        "future_corr": mean_future_corr(topk_corr_train, corr_test),
        "insample_corr": mean_future_corr(topk_corr_train, corr_train),
        "gics_hit3_train": gics_hit3_from_topk(topk_corr_train, label_arr),
        "gics_hit3_test": gics_hit3_from_topk(topk_corr_test, label_arr),
    })
    rows.append({
        "method": "oracle_B (ceiling)",
        "future_corr": mean_future_corr(topk_corr_test, corr_test),
        "insample_corr": float("nan"),
        "gics_hit3_train": float("nan"),
        "gics_hit3_test": float("nan"),
    })

    for m in ALL_FAMILIES:
        use_seeds = [seeds[0]] if is_deterministic(m) else seeds
        fut, ins, hit_train, hit_test = [], [], [], []
        for s in use_seeds:
            emb_train, _ = train_embedding(spec(m), s, returns=ret_train)
            emb_test, _ = train_embedding(spec(m), s, returns=ret_test)
            X_train = _ordered_matrix(emb_train, tickers)
            X_test = _ordered_matrix(emb_test, tickers)
            tk_train = _cosine_topk(X_train, k)
            tk_test = _cosine_topk(X_test, k)
            fut.append(mean_future_corr(tk_train, corr_test))
            ins.append(mean_future_corr(tk_train, corr_train))
            hit_train.append(gics_hit3_from_topk(tk_train, label_arr))
            hit_test.append(gics_hit3_from_topk(tk_test, label_arr))
        rows.append({
            "method": m,
            "future_corr": float(np.mean(fut)),
            "insample_corr": float(np.mean(ins)),
            "gics_hit3_train": float(np.mean(hit_train)),
            "gics_hit3_test": float(np.mean(hit_test)),
        })

    for r in rows:
        r["window"] = window_label
        r["train_end"] = train_end
        r["test_end"] = test_end

    return rows


def _aggregate(rows: list[dict]) -> pd.DataFrame:
    """Aggregate per-window rows to method-level mean ± std."""
    records = defaultdict(list)
    for r in rows:
        key = r["method"]
        records[key].append(r)

    agg_rows = []
    for method, recs in records.items():
        base = {"method": method}
        for metric in ["future_corr", "insample_corr", "gics_hit3_train", "gics_hit3_test"]:
            vals = [r[metric] for r in recs if not np.isnan(r[metric])]
            if vals:
                base[f"{metric}_mean"] = float(np.mean(vals))
                base[f"{metric}_std"] = float(np.std(vals))
                base[f"{metric}_n"] = len(vals)
            else:
                base[f"{metric}_mean"] = float("nan")
                base[f"{metric}_std"] = float("nan")
                base[f"{metric}_n"] = 0
        agg_rows.append(base)
    return pd.DataFrame(agg_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Temporal / out-of-sample validation")
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--dimensions", type=int, default=64)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--train-periods", default=None,
                    help="Comma-separated train-window lengths, e.g. 1y,2y")
    ap.add_argument("--test-horizons", default=None,
                    help="Comma-separated test-window lengths, e.g. 1y,2y")
    ap.add_argument("--num-windows", type=int, default=5,
                    help="Number of rolling train/test windows to generate")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    k = args.k
    prices = load_prices(args.universe, args.period)
    labels_all = load_all_levels()
    sub_labels = labels_all["sub_industry"]

    def spec(embedder: str) -> RunSpec:
        return RunSpec(
            method=embedder,
            universe=args.universe,
            period=args.period,
            embedder=embedder,
            graph_mode="knn",
            knn_k=10,
            dimensions=args.dimensions,
        )

    all_rows = []

    if args.train_periods and args.test_horizons:
        train_periods = [p.strip() for p in args.train_periods.split(",")]
        test_horizons = [h.strip() for h in args.test_horizons.split(",")]

        for train_period in train_periods:
            for test_horizon in test_horizons:
                print(f"\n{'='*80}")
                print(f"Configuration: train={train_period}, test={test_horizon}, k={k}")
                print(f"{'='*80}")
                try:
                    windows = generate_windows(
                        prices, train_period, test_horizon, args.num_windows
                    )
                except ValueError as e:
                    print(f"SKIP: {e}")
                    continue

                for ret_train, ret_test, label, train_end, test_end in windows:
                    tickers = [t for t in ret_train.columns if t in ret_test.columns and t in sub_labels]
                    if len(tickers) < 50:
                        print(f"  SKIP {label}: only {len(tickers)} common tickers")
                        continue

                    ret_train_s = ret_train[tickers]
                    ret_test_s = ret_test[tickers]
                    label_arr = np.asarray([sub_labels[t] for t in tickers], dtype=object)
                    print(f"  ▶ {label}: {len(tickers)} stocks, train={len(ret_train_s)}d, test={len(ret_test_s)}d")

                    rows = _run_window(
                        ret_train_s, ret_test_s, label, train_end, test_end,
                        spec, seeds, k, label_arr, tickers,
                    )
                    for r in rows:
                        r["train_period"] = train_period
                        r["test_horizon"] = test_horizon
                    all_rows.extend(rows)

    else:
        # Legacy single-split behaviour
        ret_train, ret_test, mid = split_windows(prices)
        tickers = [t for t in ret_train.columns if t in ret_test.columns and t in sub_labels]
        ret_train, ret_test = ret_train[tickers], ret_test[tickers]
        label_arr = np.asarray([sub_labels[t] for t in tickers], dtype=object)
        print(f"📂 {args.universe}/{args.period}: {len(tickers)} common stocks | "
              f"train={ret_train.shape[0]}d (≤{mid})  test={ret_test.shape[0]}d (>{mid})")

        rows = _run_window(
            ret_train, ret_test, f"split_{mid}", mid, "",
            spec, seeds, k, label_arr, tickers,
        )
        for r in rows:
            r["train_period"] = "half"
            r["test_horizon"] = "half"
        all_rows.extend(rows)

    if not all_rows:
        print("No valid windows generated.")
        return

    df = pd.DataFrame(all_rows)
    df["retention"] = df["future_corr"] / df["insample_corr"]

    # Output paths
    out_dir = os.path.dirname(args.out) if args.out else "results"
    os.makedirs(out_dir or ".", exist_ok=True)
    base = args.out or os.path.join("results", f"temporal_{args.universe}_{args.period}")
    per_window_path = f"{base}_per_window.csv"
    aggregate_path = f"{base}_aggregate.csv"

    df.to_csv(per_window_path, index=False)
    agg = _aggregate(all_rows)
    agg.to_csv(aggregate_path, index=False)

    # Print aggregate table
    print("\n" + "=" * 100)
    print(f"TEMPORAL / OUT-OF-SAMPLE AGGREGATE | {args.universe}/{args.period} | k={k}")
    print("=" * 100)
    print(f"{'method':20} {'hit@3 train':>12} {'hit@3 test':>12} "
          f"{'insample ρ':>11} {'future ρ':>10} {'retention':>10}")
    print("-" * 100)
    for _, r in agg.iterrows():
        def f(x, w=12, p=3):
            return f"{x:>{w}.{p}f}" if pd.notna(x) else f"{'—':>{w}}"
        print(f"{r['method']:20} "
              f"{f(r['gics_hit3_train_mean'])} ±{f(r['gics_hit3_train_std'], w=5, p=3)} "
              f"{f(r['gics_hit3_test_mean'])} ±{f(r['gics_hit3_test_std'], w=5, p=3)} "
              f"{f(r['insample_corr_mean'], 11)} {f(r['future_corr_mean'], 10)} "
              f"{f(r['future_corr_mean'] / r['insample_corr_mean'], 10)}")
    print("-" * 100)
    print(f"Per-window rows saved to {per_window_path}")
    print(f"Aggregate rows saved to {aggregate_path}")


if __name__ == "__main__":
    main()
