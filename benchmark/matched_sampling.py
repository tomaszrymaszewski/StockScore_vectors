"""
Universe / covariance confound -> a controlled experiment.

The paper observes that the Russell 3000 produces better embeddings than the S&P 500 and
worries this is confounded: R3000 has both (a) more stocks and (b) a different composition
(it includes mid-caps that may "bridge" sector clusters). This module converts that
confession into an experiment by **matching sample size** so only composition varies.

Two matched comparisons:

* **Node-count matched** -- randomly subsample the larger universe down to the smaller
  universe's node count and recompute the hit-rate over many subsamples. If the larger
  universe still wins at equal node count, the advantage is compositional (the mid-cap
  "bridge" hypothesis), not merely "more nodes".

* **Composition vs size** -- compare a random R3000 subsample (size control, mixed caps)
  against the R3000-restricted-to-S&P-500 tickers (the large-cap core at the same node
  count). The gap isolates the mid-cap contribution.

To stay tractable locally the default representation is the **raw correlation** baseline
(no Node2Vec); pass `representation="embedding"` to run the full pipeline (better suited to
a Modal sweep). Either way every subsample flows through the identical hit-rate scoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from benchmark.labels import load_all_levels, load_labels
from benchmark.run_benchmark import RunSpec, fetch_returns, run_raw_correlation, run_single
from benchmark import evaluation as ev


def subsample_columns(returns: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Random column (stock) subsample without replacement."""
    rng = np.random.default_rng(seed)
    n = min(n, returns.shape[1])
    cols = rng.choice(returns.columns.to_numpy(), size=n, replace=False)
    return returns[list(cols)]


def _hit_rate_for_returns(
    returns: pd.DataFrame,
    labels_by_level,
    *,
    representation: str,
    level: str,
    k: int,
    seed: int,
    spec: RunSpec,
) -> float:
    if returns.shape[1] < k + 2:
        return float("nan")
    if representation == "raw_corr":
        res = run_raw_correlation(spec, labels_by_level, returns=returns)
    elif representation == "embedding":
        res = run_single(spec, seed, labels_by_level, returns=returns)
    else:
        raise ValueError("representation must be 'raw_corr' or 'embedding'")
    return res.metrics[level].hit_rate[k]


@dataclass
class MatchedResult:
    summary: pd.DataFrame      # one row per (group, subsample) with hit-rate
    table: pd.DataFrame        # mean +/- std per group

    def __str__(self) -> str:
        return self.table.to_string()


def run_matched_node_count(
    large: str = "r3000",
    small: str = "sp500",
    period: str = "2y",
    *,
    representation: str = "raw_corr",
    level: str = "sub_industry",
    k: int = 3,
    n_subsamples: int = 20,
    seeds: list[int] | None = None,
    spec_kwargs: dict | None = None,
) -> MatchedResult:
    """Compare universes at MATCHED node count.

    Groups reported:
      * small_full            -- the small universe, all of its stocks
      * large_subsample       -- large universe randomly subsampled to |small| stocks
      * large_core_of_small   -- large universe restricted to the small universe's tickers
                                 (large-cap core, same node count) -- composition control
      * large_full            -- large universe, all stocks (for reference)
    """
    seeds = seeds or [0]
    spec_kwargs = spec_kwargs or {}
    labels_by_level = load_all_levels()

    spec_large = RunSpec(universe=large, period=period, **spec_kwargs)
    spec_small = RunSpec(universe=small, period=period, **spec_kwargs)
    ret_large = fetch_returns(spec_large)
    ret_small = fetch_returns(spec_small)
    n_small = ret_small.shape[1]
    small_tickers = [t for t in ret_small.columns if t in ret_large.columns]

    rows = []

    def record(group, returns, sub_id, seed):
        hr = _hit_rate_for_returns(
            returns, labels_by_level, representation=representation,
            level=level, k=k, seed=seed, spec=spec_large,
        )
        rows.append({"group": group, "subsample": sub_id, "seed": seed,
                     "n_stocks": returns.shape[1], f"hit_rate@{k}": hr})

    # Reference groups (single evaluation; seed only matters for embeddings).
    for seed in (seeds if representation == "embedding" else seeds[:1]):
        record("small_full", ret_small, 0, seed)
        record("large_full", ret_large, 0, seed)
        record("large_core_of_small", ret_large[small_tickers], 0, seed)

    # Random subsamples of the large universe at matched node count.
    for j in range(n_subsamples):
        sub = subsample_columns(ret_large, n_small, seed=1000 + j)
        seed = seeds[j % len(seeds)]
        record("large_subsample", sub, j, seed)

    summary = pd.DataFrame(rows)
    table = (
        summary.groupby("group")[f"hit_rate@{k}"]
        .agg(["mean", "std", "count"])
        .reindex(["small_full", "large_subsample", "large_core_of_small", "large_full"])
    )
    return MatchedResult(summary=summary, table=table)


if __name__ == "__main__":
    res = run_matched_node_count(
        period="0.5y", representation="raw_corr", level="sub_industry",
        k=3, n_subsamples=10,
    )
    print(res.table)
