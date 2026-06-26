import requests
import pandas as pd
import datetime
import numpy as np
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
YEARS = 5  # Number of years of data to fetch
API_KEY = os.getenv('POLYGON_API_KEY')
OUTPUT_FILE = f"data/russell_1000_{YEARS}y.csv"


# ==========================================
# DATA ACQUISITION
# ==========================================

def get_sp500_tickers():
    """Scrapes the current S&P 500 ticker list from Wikipedia."""
    print("Fetching S&P 500 tickers from Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/Russell_1000_Index'
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        df = tables[3]
        return df['Symbol'].tolist()
    except Exception as e:
        print(f"Error fetching ticker list: {e}")
        return []


import concurrent.futures


def _fetch_ticker_data(ticker, start_str, end_str, api_key):
    """Helper to fetch data for a single ticker."""
    url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
    params = {"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": api_key}

    try:
        # Create a new session per thread or use a simple request
        # Using a fresh session or requests directly is fine for thread ease,
        # but passing session is better for connection pooling.
        # For simplicity and thread safety, let's use direct requests here or a thread-local session if needed.
        # Given the volume, direct requests is simplest, but less efficient on handshake.
        # Let's start a session inside since this is a one-off fetch per task or use a shared session if we are careful.
        # Actually, requests.get is thread-safe.

        resp = requests.get(url, params=params)

        if resp.status_code == 200:
            data = resp.json()
            if 'results' in data:
                dates = [datetime.datetime.fromtimestamp(b['t'] / 1000).strftime('%Y-%m-%d') for b in data['results']]
                closes = [b['c'] for b in data['results']]
                return ticker, pd.Series(closes, index=dates)
        elif resp.status_code == 429:
            print(f"Rate limit hit for {ticker}, sleeping...")
            time.sleep(1)  # Default to 1s if we hit it, cleaner than 0.01 in parallel race
            # Retry once
            resp = requests.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                if 'results' in data:
                    dates = [datetime.datetime.fromtimestamp(b['t'] / 1000).strftime('%Y-%m-%d') for b in
                             data['results']]
                    closes = [b['c'] for b in data['results']]
                    return ticker, pd.Series(closes, index=dates)

    except Exception as e:
        print(f"Error {ticker}: {e}")

    return ticker, None


def fetch_data(tickers, api_key, years):
    """Fetches years of daily close data from Massive (Polygon.io) using parallel requests."""
    end_date = datetime.datetime.strptime("2025-11-30", '%Y-%m-%d').date()
    start_date = end_date - datetime.timedelta(days=int(years * 365))
    start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

    all_data = {}
    print(f"Fetching data for {len(tickers)} tickers from {start_str} to {end_str} with parallel requests...")

    # Use ThreadPoolExecutor for parallel IO operations
    # 50 workers is a good balance for 100 req/s limit to avoid instant 429s but keep it fast
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        future_to_ticker = {
            executor.submit(_fetch_ticker_data, ticker, start_str, end_str, api_key): ticker
            for ticker in tickers
        }

        completed_count = 0
        total_count = len(tickers)

        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                t, series = future.result()
                if series is not None:
                    all_data[t] = series
            except Exception as exc:
                print(f'{ticker} generated an exception: {exc}')

            completed_count += 1
            if completed_count % 50 == 0:
                print(f"Progress: {completed_count}/{total_count}")

    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def process_data(df):
    """Handles ticker renames and removes incomplete data."""
    print("Processing data...")

    # Ticker Synchronization (Renames and Merges)
    renames = {
        'FB': 'META', 'FISV': 'FI', 'ANTM': 'ELV', 'ABC': 'COR', 'PKI': 'RVTY',
        'NLOK': 'GEN', 'RE': 'EG', 'DISCA': 'WBD', 'DISCK': 'WBD', 'FLT': 'CPAY',
        'PEAK': 'DOC', 'WRK': 'SW', 'CDAY': 'DAY', 'KSU': 'CP'
    }

    for old, new in renames.items():
        if old in df.columns:
            if new in df.columns:
                # Fill missing new values with old values (handling the rename transition)
                df[new] = df[new].fillna(df[old])
                df.drop(columns=[old], inplace=True)
            else:
                df.rename(columns={old: new}, inplace=True)

    # Remove assets without data for the full period (any NaNs)
    # Note: df is an index-aligned dataframe. If a column has ANY NaN, it means
    # it is missing data for some days where other stocks have data.
    original_count = len(df.columns)
    df_clean = df.dropna(axis=1, how='any')
    final_count = len(df_clean.columns)

    print(f"Removed {original_count - final_count} assets due to missing data.")

    return df_clean


if __name__ == "__main__":
    tickers = get_sp500_tickers()
    if tickers:
        raw_df = fetch_data(tickers, API_KEY, YEARS)
        if not raw_df.empty:
            final_df = process_data(raw_df)
            final_df.to_csv(OUTPUT_FILE)
            print(f"Process complete. data saved to {OUTPUT_FILE}")
        else:
            print("No data fetched.")
