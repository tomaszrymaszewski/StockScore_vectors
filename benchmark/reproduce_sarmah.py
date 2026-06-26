"""
Faithful reproduction of Sarmah et al. (arXiv:2207.07183).

Their exact setup (from the paper):
  * Data:   S&P 500 constituents, calendar year 2021, daily adjusted close, LOG returns (504 stocks).
  * Graph:  MST on the Mantegna distance d_ij = sqrt(2(1-rho_ij)), edges relabeled with rho.
  * Embed:  Node2Vec optimum -- walk_length=100, num_walks=50, p=2, q=0.5, window=5, dim=16.
  * Eval:   KMeans (K = #GICS classes) -> V-measure, at 3 levels.
            Sarmah K: Sector 9, Industry-Group 59, Sub-group 169.
            Reported V-measure: 0.35 / 0.70 / 0.84  (average 0.63).

Our reproduction uses Polygon S&P 500 adjusted prices sliced to 2021 (~471 stocks, 246 days --
slightly fewer than their 504/252) with GICS labels. We align Sarmah's three granularities to GICS
levels by cluster count: Sector->sector, Group(K=59)->industry, Sub-group(K=169)->sub_industry.

Run:  python -m benchmark.reproduce_sarmah --seeds 0-4
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from train_vectors import Config, EmbeddingTrainer, GraphBuilder
from benchmark.clustering_metrics import clustering_metrics
from benchmark.labels import load_all_levels

# Sarmah's reported V-measure and K. Their K (9/59/169) do NOT match standard GICS counts and 169
# exceeds the GICS sub-industry maximum (163) -- their labels are non-standard (Yahoo-sourced). We
# align their 3 levels to our canonical GICS levels by closest K: Sector->sector, "Group"(59)->our
# industry(74), "Subgroup"(169)->our sub_industry(163). We report all 4 GICS levels for transparency.
SARMAH = {
    "sector":       {"v": 0.35, "K": 9},    # their "Sector"
    "industry":     {"v": 0.70, "K": 59},   # their "Group"  (K=59 ~ GICS Industry, not Group)
    "sub_industry": {"v": 0.84, "K": 169},  # their "Subgroup" (K=169 > GICS max 163)
}
LEVELS = ["sector", "industry_group", "industry", "sub_industry"]


def load_2021_log_returns(path="data/sp500_5y.csv", start="2021-01-01", end="2021-12-31"):
    """S&P 500 2021 daily log returns from Polygon prices (Sarmah's exact year)."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.loc[start:end].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, thresh=int(len(df) * 0.99)).ffill().dropna(axis=1)
    logret = np.log(df).diff().dropna()
    return logret.replace([np.inf, -np.inf], np.nan).dropna(axis=1)


def _holdings_ranked_by_market_value():
    """GICS-labeled tickers from the Holdings file, ordered by market value (largest first)."""
    path = "data/Holdings_details_Russell_3000_ETF.csv"
    with open(path) as f:
        lines = f.readlines()
    hidx = next(i for i, l in enumerate(lines) if "TICKER" in l and "SUB-INDUSTRY" in l)
    h = pd.read_csv(path, skiprows=hidx).dropna(subset=["TICKER", "SUB-INDUSTRY", "MARKET VALUE*"])
    h["mv"] = pd.to_numeric(
        h["MARKET VALUE*"].astype(str).str.replace(r"[\$,]", "", regex=True), errors="coerce")
    return h.dropna(subset=["mv"]).sort_values("mv", ascending=False)["TICKER"].astype(str).tolist()


def load_kaggle_year_log_returns(year=2017, n=504, chunksize=3_000_000):
    """Daily log returns for the top-`n` large-cap GICS-labeled stocks in a full pre-2018 year.

    Kaggle ends mid-2018 (no 2019+), so it cannot supply Sarmah's 2021 — this replicates their
    METHOD on a full, well-covered earlier year, MATCHING THEIR N (=504) by taking the largest-cap
    GICS-labeled names (ranked by Holdings market value) with full `year` coverage. That both hits
    the sample size and approximates the S&P 500 (≈ the 500 largest US stocks). GICS labels come
    from the Holdings file.
    """
    ranked = _holdings_ranked_by_market_value()
    rankpos = {t: i for i, t in enumerate(ranked)}
    cand = ranked[: max(3 * n, 1500)]
    wanted = set(cand) | {t.replace(".", "-") for t in cand}
    start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    path = "data/kaggle_raw/historical_stock_prices.csv"
    pcol = "adj_close" if "adj_close" in pd.read_csv(path, nrows=0).columns else "close"
    parts = []
    for chunk in pd.read_csv(path, usecols=["ticker", "date", pcol], chunksize=chunksize):
        chunk = chunk[chunk["ticker"].isin(wanted)]
        if len(chunk):
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
            chunk = chunk[(chunk["date"] >= start) & (chunk["date"] <= end)]
            if len(chunk):
                parts.append(chunk.rename(columns={pcol: "price"}))
    panel = pd.concat(parts).pivot_table(index="date", columns="ticker",
                                         values="price", aggfunc="last").sort_index()
    panel = panel.dropna(axis=1, thresh=int(len(panel) * 0.99)).ffill().dropna(axis=1)
    # Map Kaggle tickers back to Holdings format (so they match GICS labels), then take top-n by cap.
    def holdings_name(c):
        return c if c in rankpos else (c.replace("-", ".") if c.replace("-", ".") in rankpos else c)
    panel = panel.rename(columns={c: holdings_name(c) for c in panel.columns})
    panel = panel.loc[:, ~panel.columns.duplicated()]
    survivors = sorted([c for c in panel.columns if c in rankpos], key=lambda c: rankpos[c])[:n]
    panel = panel[survivors]
    logret = np.log(panel).diff().dropna()
    return logret.replace([np.inf, -np.inf], np.nan).dropna(axis=1)


def sarmah_config(seed: int) -> Config:
    cfg = Config(skip_pinecone=True, seed=seed)
    cfg.graph_filtering = "mst"
    cfg.mst_distance_transform = True       # d = sqrt(2(1-rho)); MST; relabel with rho
    cfg.mst_extra_edges_pct = 0.0           # pure tree
    cfg.dimensions = 16
    cfg.walk_p, cfg.walk_q = 2.0, 0.5
    cfg.walk_length, cfg.num_walks, cfg.window = 100, 50, 5
    return cfg


def _parse_seeds(s):
    if "-" in s and "," not in s:
        lo, hi = s.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description="Reproduce Sarmah et al. (2207.07183)")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--source", default="polygon2021", choices=["polygon2021", "kaggle2017"],
                    help="polygon2021 = Sarmah's exact year; kaggle2017 = method on a full earlier year")
    ap.add_argument("--year", type=int, default=2017, help="year for --source kaggle2017")
    ap.add_argument("--n", type=int, default=504, help="#stocks for --source kaggle2017 (match Sarmah)")
    args = ap.parse_args()
    seeds = _parse_seeds(args.seeds)

    if args.source == "kaggle2017":
        ret = load_kaggle_year_log_returns(year=args.year, n=args.n)
        src = f"Kaggle S&P 500 {args.year}"
    else:
        ret = load_2021_log_returns()
        src = "Polygon S&P 500 2021"
    labels = load_all_levels()
    print(f"📂 {src}: {ret.shape[1]} stocks x {ret.shape[0]} days, LOG returns "
          f"(Sarmah: 504 stocks / 252 days, year 2021)")

    acc = {lv: [] for lv in LEVELS}
    nclu = {lv: None for lv in LEVELS}
    graph_info = None
    for s in seeds:
        cfg = sarmah_config(s)
        G = GraphBuilder(cfg).build(ret)
        graph_info = (G.number_of_nodes(), G.number_of_edges())
        emb = EmbeddingTrainer(cfg).train(G)
        for lv in LEVELS:
            cm = clustering_metrics(emb, labels[lv], seed=s)
            acc[lv].append(cm["v_measure"])
            nclu[lv] = cm["n_clusters"]

    print(f"🌲 MST graph: {graph_info[0]} nodes, {graph_info[1]} edges (tree => edges=nodes-1)")
    print("\n" + "=" * 76)
    print(f"REPRODUCTION OF SARMAH et al. -- V-measure ({len(seeds)} seeds)")
    print("=" * 76)
    print(f"{'GICS level':16} {'K ours':>7} {'K Sarmah':>9} {'V ours':>16} {'V Sarmah':>9}")
    print("-" * 78)
    for lv in LEVELS:
        v = np.array(acc[lv])
        s = SARMAH.get(lv)
        ks = str(s["K"]) if s else "—"
        vs = f"{s['v']:.2f}" if s else "—"
        print(f"{lv:16} {nclu[lv]:>7} {ks:>9} {v.mean():.3f} ± {v.std(ddof=1):.3f}{'':4} {vs:>9}")
    aligned = [lv for lv in LEVELS if lv in SARMAH]
    ours_avg = np.mean([np.mean(acc[lv]) for lv in aligned])
    print("-" * 78)
    print(f"{'avg (3 aligned)':16} {'':>7} {'':>9} {ours_avg:>13.3f}{'':4} {0.63:>9}")


if __name__ == "__main__":
    main()
