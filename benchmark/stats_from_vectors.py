"""
Step 2 of the decoupled workflow: read a saved vector space JSON and compute ALL stats.

No retraining happens here -- the script consumes the `{seed: {ticker: vector}}` payload that
`benchmark.build_vectors` wrote and reports, per GICS granularity level and averaged over the
saved seeds (mean +/- std):

  * retrieval : hit-rate@k, success@k, MRR@k   (k in 1, 3, 5, 10)
  * clustering: V-measure, NMI, ARI, homogeneity, completeness  (KMeans, K = #classes)

With `--compare other.json` it additionally runs the paired significance machinery between the
two saved vector spaces on their matched seeds (Wilcoxon signed-rank + paired bootstrap CI on
per-stock hit-fraction, and McNemar on success@1) at the chosen level/k.

CLI:
    python -m benchmark.stats_from_vectors results/vectors/fc_r3000_2y.json
    python -m benchmark.stats_from_vectors a.json --compare b.json --level sub_industry --k 3
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from benchmark import evaluation as ev
from benchmark import significance as sig
from benchmark.clustering_metrics import clustering_metrics
from benchmark.labels import LEVELS, load_all_levels

CLUSTER_KEYS = ("v_measure", "nmi", "ari", "homogeneity", "completeness")


def load_vectors(path: str) -> dict:
    with open(path) as f:
        payload = json.load(f)
    if "seeds" not in payload:
        raise ValueError(f"{path} is not a vectors payload (missing 'seeds').")
    return payload


def seed_embeddings(payload: dict) -> dict[str, dict[str, list[float]]]:
    return payload["seeds"]


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def stats_table(
    payload: dict,
    labels_by_level: dict[str, dict[str, str]],
    ks: tuple[int, ...] = ev.DEFAULT_KS,
) -> pd.DataFrame:
    """Aggregate retrieval + clustering metrics over seeds for each level -> tidy DataFrame."""
    seeds = seed_embeddings(payload)
    rows = []
    for level in LEVELS:
        labels = labels_by_level[level]
        # Collect per-seed values for each metric.
        acc: dict[str, list[float]] = {}
        n_stocks = 0
        for seed_str, emb in seeds.items():
            frame = ev.neighbor_match_frame(emb, labels, max_k=max(ks))
            bundle = ev.metric_bundle(frame, ks)
            n_stocks = bundle.n_stocks
            for k in ks:
                acc.setdefault(f"hit_rate@{k}", []).append(bundle.hit_rate[k])
                acc.setdefault(f"success@{k}", []).append(bundle.success_at_k[k])
                acc.setdefault(f"mrr@{k}", []).append(bundle.mrr[k])
            cm = clustering_metrics(emb, labels, seed=int(seed_str))
            for key in CLUSTER_KEYS:
                acc.setdefault(key, []).append(cm.get(key))
        for metric, vals in acc.items():
            mean, std = _mean_std(vals)
            rows.append({"level": level, "metric": metric, "mean": mean,
                         "std": std, "n_seeds": len(seeds), "n_stocks": n_stocks})
    return pd.DataFrame(rows)


def print_stats(df: pd.DataFrame, payload: dict) -> None:
    spec = payload.get("spec", {})
    print("\n" + "=" * 74)
    print(f"VECTOR-SPACE STATS  |  {spec.get('method','?')}  "
          f"{payload.get('universe','?')}/{payload.get('period','?')}  "
          f"dim={spec.get('dimensions','?')}  seeds={int(df['n_seeds'].iloc[0])}")
    print("=" * 74)
    for level in df["level"].drop_duplicates():
        sub = df[df["level"] == level].set_index("metric")
        n = int(sub["n_stocks"].iloc[0])
        print(f"\n[{level}]  (n={n} stocks)")
        for metric in ["hit_rate@1", "hit_rate@3", "hit_rate@5", "hit_rate@10",
                       "mrr@3", "success@3", "v_measure", "ari", "nmi"]:
            if metric in sub.index:
                r = sub.loc[metric]
                print(f"   {metric:14} {r['mean']:.4f} ± {r['std']:.4f}")


def compare(
    payload_a: dict,
    payload_b: dict,
    labels_by_level: dict[str, dict[str, str]],
    level: str,
    k: int,
) -> pd.DataFrame:
    """Paired Wilcoxon (+bootstrap CI) and McNemar between two saved vector spaces."""
    labels = labels_by_level[level]
    name_a = payload_a.get("spec", {}).get("method", "A")
    name_b = payload_b.get("spec", {}).get("method", "B")
    if name_a == name_b:
        name_a, name_b = f"{name_a}#a", f"{name_b}#b"

    def frames(payload):
        return [ev.neighbor_match_frame(emb, labels, max_k=max(k, 1))
                for emb in seed_embeddings(payload).values()]

    frames_by_method = {name_a: frames(payload_a), name_b: frames(payload_b)}
    wilcox = sig.compare_methods(frames_by_method, reference=name_b, level=level, k=k)

    # McNemar on success@1 (seed-averaged -> majority vote per stock for a binary call).
    succ_a = pd.concat([ev.per_stock_success(f, 1) for f in frames_by_method[name_a]], axis=1).mean(axis=1) >= 0.5
    succ_b = pd.concat([ev.per_stock_success(f, 1) for f in frames_by_method[name_b]], axis=1).mean(axis=1) >= 0.5
    mc = sig.mcnemar_compare(succ_a, succ_b, method_a=name_a, method_b=name_b, level=level)
    print("\nMcNemar success@1:",
          f"delta={mc.delta:+.4f}  p={mc.p_value:.3g}  effect={mc.effect_size:+.3f}")
    return wilcox


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute stats from a saved vector space (step 2)")
    ap.add_argument("vectors", help="path to a vectors JSON from build_vectors")
    ap.add_argument("--compare", default=None, help="second vectors JSON for paired significance")
    ap.add_argument("--level", default="sub_industry", choices=LEVELS)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=None, help="optional CSV path for the stats table")
    args = ap.parse_args()

    labels = load_all_levels()
    payload = load_vectors(args.vectors)

    df = stats_table(payload, labels)
    print_stats(df, payload)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\n📝 Saved stats table -> {args.out}")

    if args.compare:
        other = load_vectors(args.compare)
        print("\n" + "=" * 74)
        print(f"PAIRED COMPARISON @ {args.level}, k={args.k}  "
              f"({payload['spec'].get('method')} vs {other['spec'].get('method')})")
        print("=" * 74)
        res = compare(payload, other, labels, args.level, args.k)
        cols = ["method_a", "mean_a", "mean_b", "delta", "p_value", "p_bh",
                "effect_size", "ci_low", "ci_high", "beats_reference"]
        with pd.option_context("display.width", 200):
            print(res[[c for c in cols if c in res.columns]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
