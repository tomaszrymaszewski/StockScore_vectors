"""
Step 1 of the decoupled workflow: GENERATE a vector space and SAVE it to JSON.

Trains a Node2Vec embedding for one method across one or more seeds and persists the raw
`{ticker: vector}` map (per seed) to a single JSON file, together with the spec metadata.
Step 2 (`benchmark.stats_from_vectors`) reads that file and computes every metric without
retraining -- so the expensive R3000 training is paid once and you can iterate on
metrics/plots/significance freely.

This replaces the old implicit coupling where training and scoring happened in one in-memory
pass (and the only on-disk vector space, `quarterly_vectors.json`, existed solely for the
now-removed vector-shift analysis).

CLI:
    python -m benchmark.build_vectors --universe r3000 --period 2y --seeds 0-9 \
        --method forced_connectivity --out results/vectors/fc_r3000_2y.json
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

from benchmark.run_benchmark import (
    RunSpec,
    _parse_seeds,
    competitor_specs,
    fetch_returns,
    train_embedding,
)

DEFAULT_OUT_DIR = os.path.join("results", "vectors")


def build_spec(args) -> RunSpec:
    """Resolve a method name to a RunSpec, falling back to a plain threshold spec."""
    registry = {
        s.method: s
        for s in competitor_specs(
            args.universe, args.period,
            walk_p=args.walk_p, walk_q=args.walk_q,
            dimensions=args.dimensions, threshold=args.threshold,
        )
    }
    if args.method in registry:
        spec = registry[args.method]
    else:
        spec = RunSpec(
            method=args.method, universe=args.universe, period=args.period,
            walk_p=args.walk_p, walk_q=args.walk_q,
            dimensions=args.dimensions, threshold=args.threshold,
        )
    if args.min_degree is not None:
        spec = replace(spec, min_degree=int(args.min_degree))
    return spec


def build_payload(spec: RunSpec, seeds: list[int]) -> dict:
    """Train `spec` at each seed (returns fetched once) and assemble the JSON payload."""
    returns = fetch_returns(spec)
    meta = dict(spec.tag())
    meta.update(
        dimensions=spec.dimensions, walk_length=spec.walk_length,
        num_walks=spec.num_walks, window=spec.window,
    )
    payload = {
        "schema": "stockscore.vectors.v1",
        "spec": meta,
        "universe": spec.universe,
        "period": spec.period,
        "seeds": {},
    }
    for s in seeds:
        print(f"   ▶ training {spec.method} seed={s} ...")
        embeddings, _ = train_embedding(spec, s, returns=returns)
        # Plain lists, JSON-serializable.
        payload["seeds"][str(s)] = {t: list(map(float, v)) for t, v in embeddings.items()}
        print(f"     ✓ {len(embeddings)} vectors")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate + save a vector space to JSON (step 1)")
    ap.add_argument("--universe", default="r3000")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--seeds", default="0-4", help="e.g. '0-9' or '0,1,2'")
    ap.add_argument("--method", default="forced_connectivity",
                    help="forced_connectivity | mst | pmfg_approx | knn | threshold_norepair | <custom>")
    ap.add_argument("--walk_p", type=float, default=0.8)
    ap.add_argument("--walk_q", type=float, default=7.2)
    ap.add_argument("--dimensions", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--min_degree", type=int, default=None,
                    help="Forced-Connectivity target minimum degree (default 3)")
    ap.add_argument("--out", default=None,
                    help="output JSON path (default results/vectors/<method>_<universe>_<period>.json)")
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    spec = build_spec(args)
    out = args.out or os.path.join(
        DEFAULT_OUT_DIR, f"{spec.method}_{spec.universe}_{spec.period}.json"
    )

    payload = build_payload(spec, seeds)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f)
    print(f"\n📝 Saved {len(seeds)} seed(s) of {spec.method} ({spec.universe}/{spec.period}) -> {out}")
    print(f"   Next: python -m benchmark.stats_from_vectors {out}")


if __name__ == "__main__":
    main()
