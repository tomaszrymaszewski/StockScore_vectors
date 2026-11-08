"""Smoke tests for the stock_embeddings library using synthetic price data."""

import numpy as np
import pandas as pd
import pytest

from stock_embeddings import Config, fit, train_embeddings


def make_prices(n_stocks: int = 12, n_days: int = 300, seed: int = 0) -> dict:
    """Synthetic prices with 3 correlated 'sector' blocks."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    factors = rng.normal(0, 0.01, size=(n_days, 3))
    data = {}
    for i in range(n_stocks):
        sector = i % 3
        noise = rng.normal(0, 0.005, size=n_days)
        rets = factors[:, sector] + noise
        prices = 100 * np.exp(np.cumsum(rets))
        data[f"STK{i:02d}"] = pd.Series(prices, index=idx)
    return data


def test_defaults_node2vec():
    emb = train_embeddings(make_prices(), verbose=False, num_walks=5,
                           walk_length=5, dimensions=8)
    assert len(emb) == 12
    assert all(len(v) == 8 for v in emb.values())


def test_all_embedders():
    data = make_prices()
    for embedder in ["node2vec", "spectral", "netmf", "raw_corr", "raw_returns"]:
        emb = train_embeddings(data, embedder=embedder, dimensions=8,
                               num_walks=5, walk_length=5, verbose=False)
        assert len(emb) == 12, embedder
        assert all(np.all(np.isfinite(v)) for v in emb.values()), embedder


def test_graph_filters():
    data = make_prices()
    for filt in ["threshold", "mst", "knn", "signed_knn"]:
        result = fit(data, graph_filtering=filt, embedder="spectral",
                     dimensions=4, verbose=False)
        assert result.graph is not None and result.graph.number_of_edges() > 0, filt


def test_preprocessing_options():
    data = make_prices()
    emb = train_embeddings(data, embedder="spectral", dimensions=4,
                           cross_sectional_demean=True, market_neutral=True,
                           spectral_clean="market", verbose=False)
    assert len(emb) == 12


def test_lists_and_arrays_accepted():
    rng = np.random.default_rng(1)
    data = {f"S{i}": list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
            for i in range(6)}
    emb = train_embeddings(data, embedder="raw_corr", verbose=False)
    assert len(emb) == 6


def test_result_helpers():
    result = fit(make_prices(), embedder="spectral", dimensions=4, verbose=False)
    df = result.to_dataframe()
    assert df.shape == (12, 4)
    sims = result.most_similar("STK00", k=3)
    assert len(sims) == 3 and all(t != "STK00" for t, _ in sims)


def test_unknown_param_raises():
    with pytest.raises(TypeError):
        train_embeddings(make_prices(), not_a_param=1)


def test_empty_data_raises():
    with pytest.raises(ValueError):
        train_embeddings({})


def test_determinism():
    data = make_prices()
    kwargs = dict(num_walks=5, walk_length=5, dimensions=8, verbose=False)
    a = train_embeddings(data, seed=7, **kwargs)
    b = train_embeddings(data, seed=7, **kwargs)
    assert a.keys() == b.keys()
    assert all(np.allclose(a[t], b[t]) for t in a)


def test_config_object_with_overrides():
    cfg = Config(dimensions=4, verbose=False)
    emb = train_embeddings(make_prices(), config=cfg, embedder="spectral")
    assert all(len(v) == 4 for v in emb.values())
