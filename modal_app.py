"""
Modal wrapper: fan the seed x config benchmark sweep across cloud containers.

Why Modal: the paper needs each configuration evaluated over many seeds (and the
matched-sampling experiment over many subsamples). That is embarrassingly parallel, so
running it as a `.map` over Modal containers turns an overnight local sweep into minutes.

Compute note: Node2Vec here is gensim Word2Vec, which is **CPU-bound** (single-process,
`workers=1` for determinism). The win on Modal is therefore *many CPU containers in
parallel*, not a GPU. A GPU only helps if you swap in a GPU Node2Vec (e.g. PyTorch
Geometric's `Node2Vec`); that is left as a documented future option -- to use it, change
`GPU` below and add `torch`/`torch-geometric` to the image.

Prerequisites:
    pip install modal
    modal token new

Run a sweep (results written locally to results/benchmark_modal.csv):
    modal run modal_app.py --universe r3000 --period 2y --seeds 0-19 --methods all

This mounts the local `data/` directory into the container, so no data is committed to the
repo and nothing is downloaded at run time. For very large data, switch the Mount to a
modal.Volume populated once via `modal volume put`.
"""
from __future__ import annotations

import os
import pathlib

import modal

# Directory names excluded from the image anywhere in the tree, plus large distilled data files
# that must NOT be uploaded (the raw Kaggle prices file alone is ~2 GB). Uses a callable so the
# match is robust regardless of glob semantics.
_EXCLUDE_DIR_NAMES = {
    "venv", ".venv", "env", ".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
    "reports", "results", "trash", ".claude", ".qodo", ".idea",
    "kaggle_raw",  # the 2GB raw prices file stays out; data/dolphin (~64MB) now ships
}
_EXCLUDE_FILE_NAMES = {"kaggle_prices.csv", "kaggle_sectors.csv"}


def _ignore(path: pathlib.Path) -> bool:
    if set(path.parts) & _EXCLUDE_DIR_NAMES:
        return True
    if path.name in _EXCLUDE_FILE_NAMES:
        return True
    return path.name.endswith((".pyc", ".pdf"))

REQUIREMENTS = [
    "numpy==1.26.4",
    "pandas==2.0.3",
    "networkx==3.1",
    "scikit-learn==1.7.1",
    "scipy==1.13.0",
    "gensim==4.4.0",
    "node2vec==0.5.0",
    "python-louvain==0.16",
    "tqdm==4.67.1",
]

# CPU-bound; bump `cpu`/`memory` rather than reaching for a GPU. To try the GPU Node2Vec
# path, set GPU = "A10G" and add torch/torch-geometric to the image + a GPU trainer.
GPU = None
CONTAINER_CPU = 2.0
CONTAINER_MEMORY = 4096

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(*REQUIREMENTS)
    .env({"STOCKSCORE_DATA_DIR": "/root/app/data", "PYTHONPATH": "/root/app"})
    # Ship the repo source + data into the container LAST (Modal requires add_local_* to be the
    # final build step). `data/` is gitignored locally but present on the launching machine, so it
    # travels with the build context; `_ignore` keeps the 2GB Kaggle raw + venvs out.
    .add_local_dir(
        PROJECT_ROOT,
        remote_path="/root/app",
        ignore=_ignore,
    )
)

app = modal.App("stockscore-benchmark", image=image)


@app.function(cpu=CONTAINER_CPU, memory=CONTAINER_MEMORY, gpu=GPU, timeout=60 * 60)
def evaluate_spec(payload: dict) -> dict:
    """Train + score one (spec, seed) in a container.

    Returns the flat summary row AND the trained embedding ({ticker: vector}) so every
    vectorspace is persisted by the caller. The raw-correlation baseline has no embedding.
    """
    import sys

    sys.path.insert(0, "/root/app")
    os.chdir("/root/app")

    from benchmark.run_benchmark import (
        RunSpec, train_embedding, evaluate_embedding, run_raw_correlation, RunResult,
    )
    from benchmark.labels import load_all_levels

    labels = load_all_levels()
    spec = RunSpec(**payload["spec"])

    if spec.method == "raw_correlation" or payload.get("raw_correlation"):
        result = run_raw_correlation(spec, labels)
        return {"row": result.summary_row(), "embeddings": None}

    seed = payload["seed"]
    emb, _ = train_embedding(spec, seed)
    metrics, _ = evaluate_embedding(emb, labels)
    row = RunResult(spec=spec, seed=seed,
                    n_tickers=metrics["sub_industry"].n_stocks, metrics=metrics).summary_row()
    return {
        "row": row,
        "embeddings": {t: [float(x) for x in v] for t, v in emb.items()},
        "method": spec.method, "universe": spec.universe, "period": spec.period, "seed": seed,
    }


@app.function(cpu=CONTAINER_CPU, memory=CONTAINER_MEMORY, gpu=GPU, timeout=60 * 60)
def evaluate_dolphin(payload: dict) -> dict:
    """Train + score one (method, seed) on the Dolphin 611-stock benchmark, return embeddings."""
    import sys
    sys.path.insert(0, "/root/app")
    os.chdir("/root/app")
    from benchmark.dolphin_benchmark import load_dolphin_data, make_config, evaluate
    from benchmark.embedding_methods import build_embeddings

    returns, labels = load_dolphin_data()
    method, seed, dims = payload["method"], payload["seed"], payload["dimensions"]
    cfg = make_config(method, seed, dims)
    emb = build_embeddings(cfg, returns, seed)
    row = {"method": method, "seed": seed, **evaluate(emb, labels, seed)}
    # Don't ship the huge raw-feature vectorspaces (N-dim per stock); they're instant to rebuild.
    ship = None if method.startswith("raw") else {t: [float(x) for x in v] for t, v in emb.items()}
    return {"row": row, "method": method, "seed": seed, "embeddings": ship}


@app.local_entrypoint()
def dolphin(
    methods: str = "rmt,lasso",
    seeds: str = "0-4",
    dimensions: int = 16,
    out: str = "results/dolphin_comparison_modal.csv",
):
    """Fan the Dolphin-protocol comparison across containers (for the slow methods like lasso)."""
    import json
    from collections import defaultdict

    import pandas as pd

    from benchmark.run_benchmark import _parse_seeds

    payloads = [{"method": m, "seed": s, "dimensions": dimensions}
                for m in methods.split(",") for s in _parse_seeds(seeds)]
    print(f"🚀 Dispatching {len(payloads)} Dolphin tasks to Modal...")
    results = list(evaluate_dolphin.map(payloads))

    by_method, rows = defaultdict(dict), []
    for r in results:
        rows.append(r["row"])
        by_method[r["method"]][str(r["seed"])] = r["embeddings"]
    vec_dir = os.path.join("results", "vectors")
    os.makedirs(vec_dir, exist_ok=True)
    for method, seeds_map in by_method.items():
        # raw-feature baselines ship no embeddings (None) -- nothing to persist.
        if method.startswith("raw") or not any(seeds_map.values()):
            continue
        with open(os.path.join(vec_dir, f"{method}_dolphin.json"), "w") as f:
            json.dump({"schema": "stockscore.vectors.v1",
                       "spec": {"method": method, "universe": "dolphin", "period": "2000-2018"},
                       "universe": "dolphin", "period": "2000-2018", "seeds": seeds_map}, f)
        print(f"💾 Saved vectorspace {method}_dolphin.json ({len(seeds_map)} seeds)")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"📝 Saved {len(df)} rows to {out}")


@app.function(cpu=CONTAINER_CPU, memory=CONTAINER_MEMORY, gpu=GPU, timeout=60 * 60)
def evaluate_density(payload: dict) -> dict:
    """One (graph-density, seed) on Sarmah's saved network -> V-measure + hit@3."""
    import sys
    sys.path.insert(0, "/root/app")
    os.chdir("/root/app")
    import numpy as np
    import pandas as pd
    from train_vectors import Config, EmbeddingTrainer, GraphBuilder
    from benchmark import evaluation as ev
    from benchmark.clustering_metrics import clustering_metrics
    from benchmark.labels import load_all_levels

    ret = pd.read_csv("data/sarmah_2017_returns.csv", index_col=0)
    labels = load_all_levels()
    c = Config(skip_pinecone=True, seed=payload["seed"])
    c.dimensions = 16
    c.walk_p, c.walk_q = 2.0, 0.5
    c.walk_length, c.num_walks, c.window = 100, 50, 5
    if payload["method"] == "knn":
        c.graph_filtering = "knn"
        c.knn_k = int(payload["param"])
    else:  # fc = threshold + repair; lower threshold => denser
        c.graph_filtering = "threshold"
        c.correlation_threshold = float(payload["param"])
        c.repair_islands = True
    G = GraphBuilder(c).build(ret)
    emb = EmbeddingTrainer(c).train(G)
    aligned = ["sector", "industry", "sub_industry"]
    V = {lv: clustering_metrics(emb, labels[lv], seed=payload["seed"])["v_measure"] for lv in aligned}
    hit = ev.hit_rate(ev.neighbor_match_frame(emb, labels["sub_industry"], max_k=3), 3)
    return {"method": payload["method"], "param": payload["param"], "seed": payload["seed"],
            "edges": G.number_of_edges(), "n_nodes": G.number_of_nodes(),
            "V_sec": V["sector"], "V_ind": V["industry"], "V_sub": V["sub_industry"],
            "V_avg": float(np.mean(list(V.values()))), "hit3_sub": hit}


@app.local_entrypoint()
def sarmah_density(
    seeds: str = "0-4",
    knn_ks: str = "10,20,40,80,160,320",
    fc_thresholds: str = "0.40,0.30,0.20,0.10,0.05",
    out: str = "results/sarmah_density_modal.csv",
):
    """Sweep denser kNN and FC graphs on Sarmah's network across Modal containers."""
    import pandas as pd

    from benchmark.run_benchmark import _parse_seeds

    sl = _parse_seeds(seeds)
    payloads = [{"method": "knn", "param": int(k), "seed": s}
                for k in knn_ks.split(",") for s in sl]
    payloads += [{"method": "fc", "param": float(t), "seed": s}
                 for t in fc_thresholds.split(",") for s in sl]
    print(f"🚀 Dispatching {len(payloads)} density tasks to Modal...")
    rows = list(evaluate_density.map(payloads))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    agg = (df.groupby(["method", "param"])
             .agg(edges=("edges", "mean"), n_nodes=("n_nodes", "mean"),
                  V_avg=("V_avg", "mean"), V_sub=("V_sub", "mean"), hit3=("hit3_sub", "mean"))
             .reset_index().sort_values(["method", "edges"]))
    print(f"\n{'method':6}{'param':>7}{'edges':>8}{'nodes':>7}{'V_avg':>8}{'V_sub':>8}{'hit@3':>8}")
    for _, r in agg.iterrows():
        print(f"{r['method']:6}{r['param']:>7}{r['edges']:>8.0f}{r['n_nodes']:>7.0f}"
              f"{r['V_avg']:>8.3f}{r['V_sub']:>8.3f}{r['hit3']:>8.3f}")
    print("MST baseline (Sarmah): V_avg 0.638 | best earlier: kNN k=10 0.696")
    print(f"📝 Saved {out}")


@app.local_entrypoint()
def main(
    suite: str = "competitor",     # "competitor" (graph methods) | "comovement" (matrix methods)
    universe: str = "sp500",
    period: str = "5y",
    seeds: str = "0-9",
    methods: str = "all",
    walk_p: float = 0.8,
    walk_q: float = 7.2,
    dimensions: int = 64,
    threshold: float = 0.65,
    knn_k: int = 10,
    include_lasso: bool = False,
    lasso_alpha: float = 0.5,      # fixed-alpha GraphicalLasso (CV is too slow at scale)
    out: str = "results/benchmark_modal.csv",
):
    """Build the (method x seed) grid locally, fan out to containers, collect rows."""
    import pandas as pd

    from benchmark.run_benchmark import (
        competitor_specs, comovement_specs, _parse_seeds, RunSpec,
    )

    seed_list = _parse_seeds(seeds)
    if suite == "comovement":
        specs = comovement_specs(
            universe, period, walk_p=walk_p, walk_q=walk_q,
            dimensions=dimensions, knn_k=knn_k, include_lasso=include_lasso,
            lasso_alpha=lasso_alpha,
        )
    else:
        specs = competitor_specs(
            universe, period, walk_p=walk_p, walk_q=walk_q,
            dimensions=dimensions, threshold=threshold,
        )
    if methods == "ours":
        specs = [s for s in specs if s.method == "forced_connectivity"]
    elif methods != "all":
        wanted = set(methods.split(","))
        specs = [s for s in specs if s.method in wanted]

    payloads = []
    # Raw-correlation baseline once (seed-independent).
    base = specs[0] if specs else RunSpec(universe=universe, period=period)
    payloads.append({"spec": vars(base), "seed": 0, "raw_correlation": True})
    # Each method across all seeds.
    for spec in specs:
        for s in seed_list:
            payloads.append({"spec": vars(spec), "seed": s})

    print(f"🚀 Dispatching {len(payloads)} container tasks to Modal...")
    results = list(evaluate_spec.map(payloads))

    # Persist EVERY vectorspace, grouped by method into the build_vectors payload format
    # (so benchmark.stats_from_vectors can read them directly).
    import json
    from collections import defaultdict

    by_method: dict[str, dict] = defaultdict(dict)
    meta: dict[str, tuple] = {}
    rows = []
    for r in results:
        rows.append(r["row"])
        if r.get("embeddings"):
            by_method[r["method"]][str(r["seed"])] = r["embeddings"]
            meta[r["method"]] = (r["universe"], r["period"])

    vec_dir = os.path.join("results", "vectors")
    os.makedirs(vec_dir, exist_ok=True)
    for method, seeds_map in by_method.items():
        uni, per = meta[method]
        payload_out = {
            "schema": "stockscore.vectors.v1",
            "spec": {"method": method, "universe": uni, "period": per},
            "universe": uni, "period": per, "seeds": seeds_map,
        }
        path = os.path.join(vec_dir, f"{method}_{uni}_{per}.json")
        with open(path, "w") as f:
            json.dump(payload_out, f)
        print(f"💾 Saved vectorspace {path} ({len(seeds_map)} seeds)")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"📝 Saved {len(df)} metric rows to {out}")


# ---------------------------------------------------------------------------
# Embedding-family comparison (node2vec vs spectral vs netmf vs raw features)
# ---------------------------------------------------------------------------


@app.function(cpu=CONTAINER_CPU, memory=CONTAINER_MEMORY, gpu=GPU, timeout=60 * 60)
def evaluate_family(payload: dict) -> dict:
    """Train one (universe, embedder, seed) and return its full metric row + embedding.

    The expensive bit is node2vec; spectral/netmf are cheap but ride along so the whole grid
    fans out uniformly. Raw-feature baselines are NOT computed here (instant + huge-dim locally).
    """
    import sys
    sys.path.insert(0, "/root/app")
    os.chdir("/root/app")
    from benchmark.run_benchmark import train_embedding
    from benchmark.embedding_families import score_embedding, spec_for
    from benchmark.labels import load_all_levels

    labels = load_all_levels()
    spec = spec_for(payload["embedder"], payload["universe"], payload["period"],
                    payload["dimensions"])
    emb, _ = train_embedding(spec, payload["seed"])
    metrics, _ = score_embedding(emb, labels, payload["seed"])
    return {"universe": payload["universe"], "embedder": payload["embedder"],
            "seed": payload["seed"], "metrics": metrics,
            "embedding": {t: [float(x) for x in v] for t, v in emb.items()}}


@app.local_entrypoint()
def families(
    universes: str = "sp500,r1000,r3000",
    embedders: str = "node2vec,spectral,netmf",
    seeds: str = "0-2",
    dimensions: int = 64,
):
    """Fan the embedding-family comparison across containers; score raw baselines locally.

    modal run modal_app.py::families --universes r3000 --seeds 0-4
    """
    from collections import defaultdict

    from benchmark import evaluation as ev
    from benchmark.run_benchmark import _parse_seeds, fetch_returns
    from benchmark.embedding_families import spec_for, score_embedding, aggregate_row, summarize
    from benchmark.embedding_methods import (
        is_deterministic, raw_correlation_vectors, raw_return_vectors,
    )
    from benchmark.labels import load_all_levels

    uni_list = [u for u in universes.split(",") if u]
    emb_list = [e for e in embedders.split(",") if e]
    seed_list = _parse_seeds(seeds)
    labels = load_all_levels()

    payloads = []
    for u in uni_list:
        for e in emb_list:
            sds = [seed_list[0]] if is_deterministic(e) else seed_list
            for s in sds:
                payloads.append({"universe": u, "embedder": e, "period": "5y",
                                 "seed": s, "dimensions": dimensions})
    print(f"🚀 Dispatching {len(payloads)} family tasks to Modal...")
    results = list(evaluate_family.map(payloads))

    # Group remote results by (universe, embedder); recompute only the cheap sub-industry
    # match frame locally (for significance) -- the metrics themselves come from the container.
    acc: dict = defaultdict(lambda: defaultdict(list))
    emb0: dict = {}
    n_seeds: dict = defaultdict(int)
    frames_local: dict = defaultdict(list)
    for r in results:
        key = (r["universe"], r["embedder"])
        n_seeds[key] += 1
        for mk, mv in r["metrics"].items():
            acc[key][mk].append(mv)
        emb0.setdefault(key, r["embedding"])
        frames_local[key].append(
            ev.neighbor_match_frame(r["embedding"], labels["sub_industry"], max_k=max(ev.DEFAULT_KS)))

    methods = emb_list + ["raw_corr", "raw_returns"]
    for u in uni_list:
        rows, frames_by_method, embs = [], {}, {}
        for e in emb_list:
            key = (u, e)
            if key not in emb0:
                continue
            rows.append(aggregate_row(e, acc[key], n_seeds[key], emb0[key]))
            frames_by_method[e] = frames_local[key]
            embs[e] = emb0[key]
        # Raw-feature baselines locally (instant; too high-dim to ship from a container).
        returns = fetch_returns(spec_for("node2vec", u, "5y", dimensions))
        for e, fn in (("raw_corr", raw_correlation_vectors), ("raw_returns", raw_return_vectors)):
            emb = fn(returns)
            m, fr = score_embedding(emb, labels, seed_list[0])
            rows.append(aggregate_row(e, {k: [v] for k, v in m.items()}, 1, emb))
            frames_by_method[e] = [fr]
            embs[e] = emb
        summarize(u, "5y", dimensions, methods, rows, frames_by_method, embs,
                  out=os.path.join("results", f"embedding_families_{u}_5y.csv"),
                  save_vectors=True)


# ---------------------------------------------------------------------------
# Ablations: embedding-dimension sweep + Node2Vec p/q provenance/sensitivity
# ---------------------------------------------------------------------------


@app.function(cpu=CONTAINER_CPU, memory=CONTAINER_MEMORY, gpu=GPU, timeout=60 * 60)
def evaluate_runspec_full(payload: dict) -> dict:
    """Train + full-score (hit/SVM/V) one arbitrary RunSpec at one seed. Returns tag + metrics."""
    import sys
    sys.path.insert(0, "/root/app")
    os.chdir("/root/app")
    from benchmark.run_benchmark import RunSpec, train_embedding
    from benchmark.embedding_families import score_embedding
    from benchmark.labels import load_all_levels

    labels = load_all_levels()
    spec = RunSpec(**payload["spec"])
    emb, _ = train_embedding(spec, payload["seed"])
    metrics, _ = score_embedding(emb, labels, payload["seed"])
    return {**payload.get("tag", {}), "seed": payload["seed"], "metrics": metrics}


def _agg_rows(results: list, key_fields: list) -> "list[dict]":
    """Aggregate per-seed metric dicts into mean/std rows, grouped by key_fields."""
    import numpy as np
    from collections import defaultdict
    acc: dict = defaultdict(lambda: defaultdict(list))
    keymeta: dict = {}
    for r in results:
        key = tuple(r[k] for k in key_fields)
        keymeta[key] = {k: r[k] for k in key_fields}
        for mk, mv in r["metrics"].items():
            acc[key][mk].append(mv)
    rows = []
    for key, metrics in acc.items():
        row = dict(keymeta[key])
        for mk, vals in metrics.items():
            vv = [x for x in vals if x is not None]
            row[f"{mk}_mean"] = float(np.mean(vv)) if vv else float("nan")
            row[f"{mk}_std"] = float(np.std(vv, ddof=1)) if len(vv) > 1 else 0.0
        rows.append(row)
    return rows


@app.local_entrypoint()
def dim_sweep(
    universe: str = "sp500",
    period: str = "5y",
    embedders: str = "node2vec,spectral,netmf",
    dims: str = "8,16,32,64,128",
    seeds: str = "0-2",
    out: str = "results/dim_sweep.csv",
):
    """Sweep embedding dimension to show the family verdict is not a dim=64 artifact."""
    import pandas as pd
    import numpy as np

    from benchmark.run_benchmark import _parse_seeds, fetch_returns
    from benchmark.embedding_families import score_embedding, spec_for
    from benchmark.embedding_methods import (
        is_deterministic, raw_correlation_vectors, raw_return_vectors,
    )
    from benchmark.labels import load_all_levels

    dim_list = [int(d) for d in dims.split(",")]
    emb_list = [e for e in embedders.split(",") if e]
    seed_list = _parse_seeds(seeds)

    payloads = []
    for e in emb_list:
        for d in dim_list:
            sds = [seed_list[0]] if is_deterministic(e) else seed_list
            for s in sds:
                spec = spec_for(e, universe, period, d)
                payloads.append({"spec": vars(spec), "seed": s, "tag": {"embedder": e, "dim": d}})
    print(f"🚀 Dispatching {len(payloads)} dim-sweep tasks to Modal...")
    results = list(evaluate_runspec_full.map(payloads))
    rows = _agg_rows(results, ["embedder", "dim"])

    # Raw baselines are dim-free: one reference value each (dim = -1 sentinel).
    labels = load_all_levels()
    returns = fetch_returns(spec_for("node2vec", universe, period, 64))
    for e, fn in (("raw_corr", raw_correlation_vectors), ("raw_returns", raw_return_vectors)):
        m, _ = score_embedding(fn(returns), labels, seed_list[0])
        rows.append({"embedder": e, "dim": -1, **{f"{k}_mean": v for k, v in m.items()},
                     **{f"{k}_std": 0.0 for k in m}})

    df = pd.DataFrame(rows).sort_values(["embedder", "dim"])
    outp = out.replace(".csv", f"_{universe}.csv")
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    df.to_csv(outp, index=False)

    print("\n" + "=" * 84)
    print(f"DIMENSION SWEEP  |  {universe}/{period}  (raw baselines are dim-free, shown as 'raw')")
    print("=" * 84)
    print(f"{'embedder':12} " + "".join(f"{('d='+str(d)):>11}" for d in dim_list))
    for metric, lab in [("hit@3_sub_industry", "sub hit@3"), ("svm_f1_sector", "sec SVM-F1")]:
        print(f"\n[{lab}]")
        for e in emb_list:
            cells = []
            for d in dim_list:
                v = df[(df.embedder == e) & (df.dim == d)][f"{metric}_mean"]
                cells.append(f"{v.iloc[0]:>11.3f}" if len(v) else f"{'—':>11}")
            print(f"{e:12} " + "".join(cells))
        for e in ("raw_corr", "raw_returns"):
            v = df[(df.embedder == e)][f"{metric}_mean"]
            if len(v):
                print(f"{e+' (raw)':12} " + f"{v.iloc[0]:>11.3f}  (flat across dim)")
    print(f"\n📝 Saved {outp}")


@app.local_entrypoint()
def pq_sweep(
    universe: str = "sp500",
    period: str = "5y",
    ps: str = "0.5,0.8,1,2,4",
    qs: str = "0.5,1,2,4,7.2",
    seeds: str = "0-2",
    dimensions: int = 64,
    out: str = "results/pq_sweep.csv",
):
    """Node2Vec p/q sensitivity: is the chosen (p=0.8,q=7.2) a cherry-picked peak, or robust?

    If hit@3 is roughly flat across the (p,q) grid, the choice cannot have leaked the test labels
    in any material way -- any reasonable config gives the same verdict. Sarmah's (p=2,q=0.5) and
    ours (p=0.8,q=7.2) both sit in the grid for direct comparison.
    """
    import pandas as pd
    import numpy as np

    from benchmark.run_benchmark import _parse_seeds, RunSpec
    from benchmark.embedding_families import spec_for

    p_list = [float(x) for x in ps.split(",")]
    q_list = [float(x) for x in qs.split(",")]
    seed_list = _parse_seeds(seeds)

    payloads = []
    for p in p_list:
        for q in q_list:
            for s in seed_list:
                base = spec_for("node2vec", universe, period, dimensions)
                spec = RunSpec(**{**vars(base), "walk_p": p, "walk_q": q})
                payloads.append({"spec": vars(spec), "seed": s, "tag": {"p": p, "q": q}})
    print(f"🚀 Dispatching {len(payloads)} p/q-sweep tasks to Modal...")
    results = list(evaluate_runspec_full.map(payloads))
    rows = _agg_rows(results, ["p", "q"])
    df = pd.DataFrame(rows)
    outp = out.replace(".csv", f"_{universe}.csv")
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    df.to_csv(outp, index=False)

    for metric, lab in [("hit@3_sub_industry", "sub hit@3"), ("hit@3_sector", "sector hit@3")]:
        piv = df.pivot_table(index="p", columns="q", values=f"{metric}_mean")
        print("\n" + "=" * 70)
        print(f"NODE2VEC p/q SWEEP  |  {universe}/{period}  [{lab}]   (ours: p=0.8,q=7.2; Sarmah: p=2,q=0.5)")
        print("=" * 70)
        print(piv.round(3).to_string())
        vals = df[f"{metric}_mean"]
        print(f"  range across grid: {vals.min():.3f}–{vals.max():.3f}  (spread {vals.max()-vals.min():.3f})")
    print(f"\n📝 Saved {outp}")
