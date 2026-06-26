"""
Distill the Kaggle "Daily Historical Stock Prices (1970-2018)" dataset into the compact
artifacts this project uses -- a free, reproducible alternative to the Polygon price pulls.

Kaggle dataset: ehallmar/daily-historical-stock-prices-1970-2018
  historical_stocks.csv        (small)  ticker,exchange,name,sector,industry
  historical_stock_prices.csv  (~21M rows, big)  ticker,open,close,adj_close,low,high,volume,date

The big prices file cannot be sliced on Kaggle, so download it once; this job reads it in
CHUNKS (low memory), filters to a date window + survivor tickers, and writes small outputs.
You can delete the big raw file afterward.

Outputs (paths configurable):
  data/kaggle_prices.csv    adj_close panel, dates x tickers  (drop-in for the main pipeline)
  data/kaggle_sectors.csv   ticker,sector,industry,name,exchange  (survivors only, no MISC)
With --dolphin it additionally writes the benchmark format consumed by
benchmark.dolphin_benchmark:
  data/dolphin/returns_df_611.csv      daily returns, dates x tickers
  data/dolphin/historical_stocks.csv   the label file

Examples:
  # Reconstruct the Dolphin 2000-2018 survivor universe + benchmark files:
  python -m jobs.prep_kaggle_data --start 2000-01-01 --end 2018-12-31 \
      --min-coverage 0.9 --dolphin

  # Just build a price panel for a custom window:
  python -m jobs.prep_kaggle_data --start 2010-01-01 --end 2018-12-31 --out data/kaggle_2010_2018.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data", "kaggle_raw")
DEFAULT_STOCKS = os.path.join(RAW_DIR, "historical_stocks.csv")
DEFAULT_PRICES = os.path.join(RAW_DIR, "historical_stock_prices.csv")
PRICE_COL = "adj_close"  # fall back to close if absent


def load_stocks(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"❌ Missing {path}. Download historical_stocks.csv from Kaggle into {RAW_DIR}/.")
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].astype(str)
    print(f"📂 historical_stocks.csv: {len(df)} tickers, "
          f"{df['sector'].nunique()} sectors")
    return df


def stream_prices(
    path: str, start: str, end: str, tickers: set[str] | None, chunksize: int
) -> pd.DataFrame:
    """Chunked read of the big prices file -> long [ticker, date, price] within the window."""
    if not os.path.exists(path):
        sys.exit(f"❌ Missing {path}. Download historical_stock_prices.csv from Kaggle into {RAW_DIR}/.")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    keep_parts = []
    seen = 0
    # Peek at the header to pick the price column.
    header = pd.read_csv(path, nrows=0).columns.tolist()
    price_col = PRICE_COL if PRICE_COL in header else "close"
    usecols = ["ticker", "date", price_col]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize,
                             dtype={"ticker": "category"}):
        seen += len(chunk)
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk[(chunk["date"] >= start_ts) & (chunk["date"] <= end_ts)]
        if tickers is not None:
            chunk = chunk[chunk["ticker"].astype(str).isin(tickers)]
        if len(chunk):
            chunk = chunk.rename(columns={price_col: "price"})
            chunk["price"] = chunk["price"].astype("float32")
            keep_parts.append(chunk[["ticker", "date", "price"]])
        print(f"   ...scanned {seen:,} rows, kept {sum(len(p) for p in keep_parts):,}",
              end="\r", file=sys.stderr)
    print(file=sys.stderr)
    if not keep_parts:
        sys.exit("❌ No rows in the requested window. Check --start/--end.")
    return pd.concat(keep_parts, ignore_index=True)


def build_panel(long_df: pd.DataFrame, min_coverage: float) -> pd.DataFrame:
    """Pivot long -> wide adj_close panel; keep survivor tickers by coverage."""
    long_df["ticker"] = long_df["ticker"].astype(str)
    panel = long_df.pivot_table(index="date", columns="ticker", values="price", aggfunc="last")
    panel = panel.sort_index()
    n_days = len(panel)
    coverage = panel.notna().sum(axis=0) / n_days
    survivors = coverage[coverage >= min_coverage].index.tolist()
    panel = panel[survivors].ffill().dropna(axis=1, how="any")
    print(f"🧮 Panel: {panel.shape[0]} days x {panel.shape[1]} survivor tickers "
          f"(coverage >= {min_coverage:.0%})")
    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description="Distill Kaggle stock data into project artifacts")
    ap.add_argument("--stocks", default=DEFAULT_STOCKS)
    ap.add_argument("--prices", default=DEFAULT_PRICES)
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2018-12-31")
    ap.add_argument("--min-coverage", type=float, default=0.9,
                    help="keep tickers present on >= this fraction of trading days")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--out", default=os.path.join("data", "kaggle_prices.csv"))
    ap.add_argument("--sectors-out", default=os.path.join("data", "kaggle_sectors.csv"))
    ap.add_argument("--dolphin", action="store_true",
                    help="also write data/dolphin/{returns_df_611.csv,historical_stocks.csv}")
    args = ap.parse_args()

    stocks = load_stocks(args.stocks)
    # Drop unlabeled, MISCELLANEOUS (as the authors do), and a stray junk "SECTOR" row
    # present in this Kaggle file. Leaves the 11 real NASDAQ-taxonomy sectors.
    bad_sectors = {"MISCELLANEOUS", "SECTOR"}
    valid = set(stocks.loc[stocks["sector"].notna() & ~stocks["sector"].isin(bad_sectors),
                           "ticker"])

    long_df = stream_prices(args.prices, args.start, args.end, valid, args.chunksize)
    panel = build_panel(long_df, args.min_coverage)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    panel.to_csv(args.out)
    print(f"📝 Prices -> {args.out}")

    survivors = list(panel.columns)
    sec = stocks[stocks["ticker"].isin(survivors)][
        ["ticker", "sector", "industry", "name", "exchange"]
    ]
    sec.to_csv(args.sectors_out, index=False)
    print(f"📝 Sectors -> {args.sectors_out} ({sec['sector'].nunique()} sectors)")

    if args.dolphin:
        ddir = os.path.join("data", "dolphin")
        os.makedirs(ddir, exist_ok=True)
        returns = panel.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
        returns.to_csv(os.path.join(ddir, "returns_df_611.csv"))
        stocks[stocks["ticker"].isin(survivors)].to_csv(
            os.path.join(ddir, "historical_stocks.csv"), index=False)
        print(f"📝 Dolphin benchmark files -> {ddir}/  "
              f"({len(survivors)} stocks; their paper used 611)")


if __name__ == "__main__":
    main()
