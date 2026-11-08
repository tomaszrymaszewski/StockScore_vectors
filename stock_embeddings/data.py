"""Convert user-provided price data into an aligned returns DataFrame."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence, Union

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)

PriceSeries = Union[pd.Series, Sequence[float], np.ndarray]


def prices_to_returns(
    data: Mapping[str, PriceSeries],
    config: Config | None = None,
) -> pd.DataFrame:
    """Build a returns DataFrame from a dict of {ticker: price series}.

    Accepts pandas Series (optionally with a DatetimeIndex), lists, or numpy
    arrays of prices. Series are outer-joined on their index (positional if no
    index), forward-filled, converted to simple returns, and cleaned:
    non-trading days dropped, tickers with too much missing history dropped,
    flat-lined stocks removed.
    """
    config = config or Config()
    if not data:
        raise ValueError("data is empty: provide {ticker: price_series}")

    df = _to_dataframe(data)

    # Drop non-trading days, then tickers with too much missing data.
    df = df.dropna(how="all")
    df = df.dropna(axis=1, thresh=int(len(df) * config.min_history_frac))
    if df.empty:
        raise ValueError("No ticker survived the missing-data filter "
                         f"(min_history_frac={config.min_history_frac})")

    # Forward-fill holes, then drop remaining rows with NaNs.
    df = df.ffill().dropna(axis=0)
    if len(df) < config.min_rows + 1:
        raise ValueError(f"Insufficient data: only {len(df)} price rows "
                         f"(need at least {config.min_rows + 1})")

    returns = df.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    # Drop stocks that didn't move (flat lines).
    returns = returns.loc[:, returns.std() > 1e-8]
    if returns.shape[1] < 2:
        raise ValueError("Fewer than 2 tickers with valid, moving price data")

    logger.info("Returns matrix: %d days x %d tickers",
                returns.shape[0], returns.shape[1])
    return returns


def _to_dataframe(data: Mapping[str, PriceSeries]) -> pd.DataFrame:
    """Align heterogeneous price inputs into a single price DataFrame."""
    series = {}
    for ticker, prices in data.items():
        if isinstance(prices, pd.Series):
            s = pd.Series(prices.values, index=prices.index)
            if not isinstance(s.index, pd.DatetimeIndex):
                s.index = pd.RangeIndex(len(s))
        else:
            arr = np.asarray(prices, dtype=float)
            if arr.ndim != 1:
                raise ValueError(f"Price data for {ticker!r} must be 1-dimensional")
            s = pd.Series(arr, index=pd.RangeIndex(len(arr)))
        s = s.apply(pd.to_numeric, errors="coerce")
        series[str(ticker)] = s

    df = pd.DataFrame(series)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.groupby(df.index).last().sort_index()
    else:
        df = df.sort_index()
    return df
