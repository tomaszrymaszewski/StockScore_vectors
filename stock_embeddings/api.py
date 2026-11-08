"""Public API: train stock embeddings from a dict of price series."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Mapping, Sequence, Union

import networkx as nx
import numpy as np
import pandas as pd

from .config import Config
from .data import PriceSeries, prices_to_returns
from .embedders import RAW_EMBEDDERS, build_embeddings
from .graph import GraphBuilder

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Outcome of a training run.

    Attributes:
        embeddings: {ticker: vector} -- the primary output.
        returns:    the aligned returns DataFrame actually used.
        graph:      the stock similarity graph (None for raw_* embedders).
        config:     the resolved Config used for this run.
    """

    embeddings: dict[str, list[float]]
    returns: pd.DataFrame
    graph: nx.Graph | None
    config: Config

    def to_dataframe(self) -> pd.DataFrame:
        """Embeddings as a DataFrame indexed by ticker (one column per dimension)."""
        return pd.DataFrame.from_dict(self.embeddings, orient="index")

    def most_similar(self, ticker: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k most cosine-similar tickers to `ticker` (excluding itself)."""
        tickers = list(self.embeddings.keys())
        if ticker not in tickers:
            raise KeyError(f"{ticker!r} not in trained embeddings")
        M = np.array([self.embeddings[t] for t in tickers], dtype=float)
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        M = M / norms
        sims = M @ M[tickers.index(ticker)]
        order = np.argsort(sims)[::-1]
        out = []
        for i in order:
            if tickers[i] == ticker:
                continue
            out.append((tickers[i], float(sims[i])))
            if len(out) >= k:
                break
        return out


def train_embeddings(
    data: Mapping[str, PriceSeries],
    config: Config | None = None,
    **params,
) -> dict[str, list[float]]:
    """Train stock embeddings from a dict of {ticker: price series}.

    Args:
        data:   mapping of ticker -> prices (pandas Series, list, or numpy
                array). A DatetimeIndex is used for alignment if present.
        config: optional pre-built Config; keyword arguments override it.
        **params: any Config field, e.g. ``dimensions=32``,
            ``graph_filtering="knn"``, ``embedder="spectral"``.

    Returns:
        {ticker: embedding vector} dict.

    Example:
        >>> embeddings = train_embeddings({"AAPL": aapl_prices, "MSFT": msft_prices})
        >>> embeddings = train_embeddings(data, dimensions=32, embedder="spectral")
    """
    return fit(data, config=config, **params).embeddings


def fit(
    data: Mapping[str, PriceSeries],
    config: Config | None = None,
    **params,
) -> TrainResult:
    """Like :func:`train_embeddings` but returns the full :class:`TrainResult`
    (embeddings + returns + graph + resolved config)."""
    config = _resolve_config(config, params)
    _setup_logging(config.verbose)

    returns = prices_to_returns(data, config)

    graph = None
    if config.embedder not in RAW_EMBEDDERS:
        graph = GraphBuilder(config).build(returns)

    embeddings = build_embeddings(config, returns, graph=graph)
    logger.info("Trained %d embeddings (embedder=%s)", len(embeddings),
                config.embedder)
    return TrainResult(embeddings=embeddings, returns=returns, graph=graph,
                       config=config)


def _resolve_config(config: Config | None, params: dict) -> Config:
    if config is None:
        return Config(**params)
    if not params:
        return config
    known = {f.name for f in fields(Config)}
    unknown = set(params) - known
    if unknown:
        raise TypeError(f"Unknown Config parameter(s): {sorted(unknown)}")
    overrides = {f.name: getattr(config, f.name) for f in fields(Config)}
    overrides.update(params)
    return Config(**overrides)


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(level)
    pkg_logger = logging.getLogger("stock_embeddings")
    pkg_logger.setLevel(level)
    if verbose and not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        pkg_logger.addHandler(handler)
        pkg_logger.propagate = False
