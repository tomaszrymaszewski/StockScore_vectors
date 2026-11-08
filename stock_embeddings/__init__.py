"""stock_embeddings -- train stock embeddings from price data.

Basic usage:
    from stock_embeddings import train_embeddings

    data = {"AAPL": aapl_price_series, "MSFT": msft_price_series, ...}
    embeddings = train_embeddings(data)                          # all defaults
    embeddings = train_embeddings(data, dimensions=32,           # any Config field
                                  graph_filtering="knn",
                                  embedder="spectral")

Use `fit()` instead of `train_embeddings()` to also get the returns matrix,
the similarity graph, and helper methods (to_dataframe, most_similar).
"""

from .api import TrainResult, fit, train_embeddings
from .config import Config

__version__ = "0.1.0"

__all__ = ["train_embeddings", "fit", "Config", "TrainResult", "__version__"]
