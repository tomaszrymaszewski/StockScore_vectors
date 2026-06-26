"""
GICS label loading at all four granularities.

The iShares Russell 3000 holdings file gives one GICS *sub-industry* label per ticker.
`data/gics_hierarchy.csv` (built by `jobs/build_gics_hierarchy.py`) rolls each
sub-industry up to its industry, industry group, and sector. Together they let the
benchmark score the embedding's hit-rate along the full GICS granularity ladder:

    sector (11)  ->  industry_group (25)  ->  industry (74)  ->  sub_industry (159)

Usage:
    from benchmark.labels import load_labels, LEVELS
    labels = load_labels("sub_industry")     # {ticker: "Semiconductors", ...}
    sector_labels = load_labels("sector")    # {ticker: "Information Technology", ...}
"""
from __future__ import annotations

import os

import pandas as pd

LEVELS = ("sector", "industry_group", "industry", "sub_industry")

_DATA_DIR = os.environ.get("STOCKSCORE_DATA_DIR", "data")
HOLDINGS_FILE = os.path.join(_DATA_DIR, "Holdings_details_Russell_3000_ETF.csv")
HIERARCHY_FILE = os.path.join(_DATA_DIR, "gics_hierarchy.csv")

_JUNK = {"---", "MARKET VALUE*", "SHARES", "", "nan"}


def _normalize_ticker(t: str) -> str:
    """Normalize ticker formatting so holdings labels match price-CSV columns.

    Price CSVs and holdings occasionally differ on class-share punctuation
    (BRK.B vs BRK-B). We standardize on the dash form used elsewhere in the repo.
    """
    return str(t).strip().upper().replace(".", "-")


def load_ticker_subindustry() -> dict[str, str]:
    """Map ticker -> GICS sub-industry from the Russell 3000 holdings file."""
    header_idx = None
    with open(HOLDINGS_FILE) as f:
        for i, line in enumerate(f):
            if "TICKER" in line and "SUB-INDUSTRY" in line:
                header_idx = i
                break
    if header_idx is None:
        raise RuntimeError(f"Could not find header row in {HOLDINGS_FILE}")

    df = pd.read_csv(HOLDINGS_FILE, skiprows=header_idx)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["TICKER", "SUB-INDUSTRY"])

    mapping: dict[str, str] = {}
    for ticker, sub in zip(df["TICKER"], df["SUB-INDUSTRY"]):
        sub = str(sub).strip()
        if sub in _JUNK or sub.startswith(("$", "-$")):
            continue
        mapping[_normalize_ticker(ticker)] = sub
    return mapping


def load_hierarchy() -> pd.DataFrame:
    """Load the sub_industry -> industry -> industry_group -> sector table."""
    if not os.path.exists(HIERARCHY_FILE):
        raise FileNotFoundError(
            f"{HIERARCHY_FILE} not found. Run: python -m jobs.build_gics_hierarchy"
        )
    return pd.read_csv(HIERARCHY_FILE)


def load_labels(level: str = "sub_industry") -> dict[str, str]:
    """Return {ticker: label} at the requested GICS granularity.

    `level` must be one of LEVELS. Tickers whose sub-industry is not present in the
    hierarchy map are dropped (they cannot be rolled up).
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")

    ticker_sub = load_ticker_subindustry()
    if level == "sub_industry":
        return ticker_sub

    hierarchy = load_hierarchy().set_index("sub_industry")[level].to_dict()
    out: dict[str, str] = {}
    for ticker, sub in ticker_sub.items():
        parent = hierarchy.get(sub)
        if parent is not None:
            out[ticker] = parent
    return out


def load_all_levels() -> dict[str, dict[str, str]]:
    """Return {level: {ticker: label}} for every GICS granularity."""
    return {level: load_labels(level) for level in LEVELS}


if __name__ == "__main__":
    for lvl in LEVELS:
        labels = load_labels(lvl)
        n_classes = len(set(labels.values()))
        print(f"{lvl:16} {len(labels):5d} tickers across {n_classes:4d} classes")
