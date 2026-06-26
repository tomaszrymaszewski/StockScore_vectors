"""
Regenerate the price-history CSVs that the benchmark trains on.

The large `data/<universe>_<period>.csv` files are gitignored. This script rebuilds them
from Polygon.io so the repo is reproducible without committing market data. Output format
matches what `train_vectors.DataFetcher` expects: a wide CSV with a date index and one
adjusted-close column per ticker.

Universe tickers come from:
  * r3000  -> the iShares IWV holdings file (data/Holdings_details_Russell_3000_ETF.csv)
  * sp500 / r1000 / custom -> tickers.py, or --tickers AAPL,MSFT,...

Examples:
    python -m jobs.fetch_prices --universe r3000 --period 2y
    python -m jobs.fetch_prices --universe sp500 --period 5y --end 2025-11-30

Notes:
  * Needs POLYGON_API_KEY in the environment / .env.
  * Polygon's free tier is rate-limited (~5 req/min); the script sleeps + retries. For
    thousands of tickers use a paid tier or run overnight.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY", "")
DATA_DIR = os.environ.get("STOCKSCORE_DATA_DIR", "data")
HOLDINGS_FILE = os.path.join(DATA_DIR, "Holdings_details_Russell_3000_ETF.csv")

_UNIVERSE_PREFIX = {"sp500": "sp500", "r1000": "russell_1000", "r3000": "russell_3000"}


def parse_period_days(period: str) -> int:
    period = period.lower().strip()
    if period.endswith("y"):
        return int(float(period[:-1]) * 365)
    if period.endswith("m"):
        return int(float(period[:-1]) * 30)
    if period.endswith("d"):
        return int(period[:-1])
    raise ValueError(f"Unrecognized period {period!r}")


def universe_tickers(universe: str) -> list[str]:
    if universe == "r3000":
        header_idx = None
        with open(HOLDINGS_FILE) as f:
            for i, line in enumerate(f):
                if "TICKER" in line and "SUB-INDUSTRY" in line:
                    header_idx = i
                    break
        df = pd.read_csv(HOLDINGS_FILE, skiprows=header_idx)
        df.columns = [c.strip() for c in df.columns]
        return [str(t).strip().upper() for t in df["TICKER"].dropna().unique()]
    # sp500 / r1000 / anything else: fall back to the static list in tickers.py.
    from tickers import TICKERS

    return list(TICKERS)


def fetch_one(ticker: str, start: str, end: str, *, max_retries: int = 5) -> pd.Series | None:
    """Daily adjusted close for one ticker as a date-indexed Series."""
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"
    )
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 429:  # rate limited
                time.sleep(15)
                continue
            data = resp.json()
            results = data.get("results")
            if not results:
                return None
            idx = pd.to_datetime([r["t"] for r in results], unit="ms").normalize()
            return pd.Series([r["c"] for r in results], index=idx, name=ticker)
        except Exception:
            time.sleep(5)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch price CSVs from Polygon")
    ap.add_argument("--universe", default="r3000", choices=list(_UNIVERSE_PREFIX))
    ap.add_argument("--period", default="2y")
    ap.add_argument("--end", default="2025-11-30", help="End date YYYY-MM-DD")
    ap.add_argument("--tickers", default=None, help="Comma list overriding the universe")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds between requests")
    args = ap.parse_args()

    if not API_KEY:
        print("❌ POLYGON_API_KEY not set (put it in .env). See .env.example.")
        return 1

    end = datetime.strptime(args.end, "%Y-%m-%d")
    start = end - timedelta(days=parse_period_days(args.period))
    start_s, end_s = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",")]
        if args.tickers
        else universe_tickers(args.universe)
    )
    print(f"📥 {len(tickers)} tickers | {start_s} -> {end_s}")

    series = {}
    for i, t in enumerate(tickers, 1):
        s = fetch_one(t, start_s, end_s)
        if s is not None and not s.empty:
            series[t] = s
        if i % 50 == 0:
            print(f"   ... {i}/{len(tickers)} ({len(series)} with data)")
        if args.sleep:
            time.sleep(args.sleep)

    if not series:
        print("❌ No data fetched.")
        return 1

    wide = pd.DataFrame(series).sort_index()
    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, f"{_UNIVERSE_PREFIX[args.universe]}_{args.period}.csv")
    wide.to_csv(out)
    print(f"✅ Wrote {out}  ({wide.shape[0]} days x {wide.shape[1]} tickers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
