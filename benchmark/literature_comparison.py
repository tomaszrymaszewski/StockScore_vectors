"""
Compute the literature-comparison metrics on every saved vectorspace, so each method/universe
can be placed next to Sarmah et al. (V-measure) and Dolphin et al. (SVM sector classification).

For each vectorspace JSON (from build_vectors / modal_app) it computes, per seed then averaged:
  * CLUSTERING (Sarmah's language): V-measure / NMI / ARI via KMeans, at sector, industry-group,
    industry, sub-industry. n_clusters reported so granularities can be aligned to Sarmah's
    K = 9 / 59 / 169 (their "group" K=59 ~ our `industry`, their sub-group ~ our `sub_industry`).
  * CLASSIFICATION (Dolphin's protocol): SVM 5-fold weighted-F1 / accuracy, at sector (+ industry,
    sub-industry for completeness; rare classes are dropped, count reported).

Writes a tidy long CSV and prints two pivot tables (V-measure and SVM-F1 by universe x method).

Reference points (NOT head-to-head -- different data/universe; cite in passing):
  Sarmah (S&P 500, MST):  V-measure 0.35 / 0.70 / 0.84  (sector / group K59 / sub-group K169)
  Dolphin (611 stocks):   SVM sector F1  0.60 / 0.61 / 0.66 / 0.69  (reimpl. Dolphin/Sarmah/Dolphin/contrastive)

Run:  python -m benchmark.literature_comparison            # all results/vectors/*_5y.json
      python -m benchmark.literature_comparison results/vectors/pca_market_r3000_5y.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from benchmark.classification_metrics import svm_classification
from benchmark.clustering_metrics import clustering_metrics
from benchmark.labels import load_all_levels

CLUSTER_LEVELS = ["sector", "industry_group", "industry", "sub_industry"]
SVM_LEVELS = ["sector", "industry", "sub_industry"]


def parse_name(path: str) -> tuple[str, str, str]:
    """corr_sp500_5y.json -> ('corr','sp500','5y'); handles methods with underscores."""
    base = os.path.basename(path)
    if base.endswith(".json"):
        base = base[:-5]
    parts = base.split("_")
    period, universe, method = parts[-1], parts[-2], "_".join(parts[:-2])
    return method, universe, period


def metrics_for_file(path: str, labels) -> list[dict]:
    method, universe, period = parse_name(path)
    seeds = json.load(open(path)).get("seeds", {})
    acc: dict[tuple, list] = {}
    for s, emb in seeds.items():
        for lv in CLUSTER_LEVELS:
            cm = clustering_metrics(emb, labels[lv], seed=int(s))
            for key in ("v_measure", "nmi", "ari", "n_clusters"):
                acc.setdefault((lv, key), []).append(cm.get(key))
        for lv in SVM_LEVELS:
            sv = svm_classification(emb, labels[lv], seed=int(s))
            acc.setdefault((lv, "svm_f1"), []).append(sv.get("f1_weighted"))
            acc.setdefault((lv, "svm_acc"), []).append(sv.get("accuracy"))
    rows = []
    for (lv, metric), vals in acc.items():
        vv = [x for x in vals if x is not None]
        if not vv:
            continue
        rows.append({
            "universe": universe, "method": method, "level": lv, "metric": metric,
            "mean": float(np.mean(vv)),
            "std": float(np.std(vv, ddof=1)) if len(vv) > 1 else 0.0,
            "n_seeds": len(vv),
        })
    return rows


def _pivot(df, metric, level):
    sub = df[(df["metric"] == metric) & (df["level"] == level)]
    if sub.empty:
        return None
    return sub.pivot_table(index="method", columns="universe", values="mean").round(4)


def main():
    ap = argparse.ArgumentParser(description="Literature-comparison metrics on all vectorspaces")
    ap.add_argument("paths", nargs="*", help="vectorspace JSONs (default results/vectors/*_5y.json)")
    ap.add_argument("--out", default="results/literature_metrics.csv")
    args = ap.parse_args()

    paths = args.paths or sorted(glob.glob(os.path.join("results", "vectors", "*_5y.json")))
    if not paths:
        raise SystemExit("No vectorspace files found.")
    labels = load_all_levels()

    rows = []
    for p in paths:
        print(f"… {os.path.basename(p)}")
        rows.extend(metrics_for_file(p, labels))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    method_order = ["corr", "demean", "market_neutral", "pca_market", "rmt"]
    uni_order = [u for u in ["sp500", "r1000", "r3000"] if u in df["universe"].unique()]

    def show(title, metric, level):
        piv = _pivot(df, metric, level)
        if piv is None:
            return
        piv = piv.reindex([m for m in method_order if m in piv.index])
        piv = piv[[u for u in uni_order if u in piv.columns]]
        print(f"\n### {title} ({metric} @ {level}) ###")
        print(piv.to_string())

    print("\n" + "=" * 70)
    print("SARMAH anchor — V-measure (ref S&P500 MST: 0.35 sector / 0.70 grp / 0.84 sub)")
    print("=" * 70)
    show("V-measure", "v_measure", "sector")
    show("V-measure", "v_measure", "industry")       # ~ Sarmah 'group' (K~59-73)
    show("V-measure", "v_measure", "sub_industry")   # ~ Sarmah 'sub-group'
    show("ARI (does not inflate with K)", "ari", "sub_industry")

    print("\n" + "=" * 70)
    print("DOLPHIN anchor — SVM sector F1 (ref 611-stock: 0.60/0.61/0.66/0.69)")
    print("=" * 70)
    show("SVM weighted-F1", "svm_f1", "sector")
    show("SVM accuracy", "svm_acc", "sector")

    print(f"\n📝 Full tidy metrics -> {args.out}")


if __name__ == "__main__":
    main()
