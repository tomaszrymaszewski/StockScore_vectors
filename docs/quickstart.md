# Quickstart

## Setup

Requires Python 3.10+.

```bash
pip install stock-embeddings
```

Or from source:

```bash
git clone https://github.com/tomaszrymaszewski/StockScore_vectors.git
cd StockScore_vectors
pip install .
```

## Train your first embeddings

```python
import pandas as pd
from stock_embeddings import train_embeddings

# Any dict of ticker -> prices works: pandas Series (optionally with a
# DatetimeIndex), lists, or numpy arrays.
prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
data = {ticker: prices[ticker].dropna() for ticker in prices.columns}

embeddings = train_embeddings(data)
# -> {"AAPL": [64 floats], "MSFT": [64 floats], ...}
```

That's it — every parameter has a research-backed default (Node2Vec, 64 dimensions,
correlation-threshold graph with Forced Connectivity repair).

## Inspect the result

```python
from stock_embeddings import fit

result = fit(data)                      # full TrainResult object
result.embeddings                       # {ticker: vector}
result.to_dataframe()                   # embeddings as a DataFrame
result.most_similar("AAPL", k=10)       # [("MSFT", 0.98), ("AMZN", 0.97), ...]
result.graph                            # the stock similarity network (networkx)
result.returns                          # the aligned returns matrix actually used
```

## Change a parameter

```python
embeddings = train_embeddings(data, dimensions=32, embedder="spectral")
```

See [Configuration](configuration.md) for every knob and
[Embedding families](embedders.md) for choosing a representation.
