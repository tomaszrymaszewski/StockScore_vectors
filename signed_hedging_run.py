"""Hedging on pca_market: signed_knn vs knn vs corr_direct (S&P 500, 5 windows, k=10).

Driver for the signed-graph hedging experiment. Reuses the experiment's own machinery
(_train_similarity, _topk_from_sim, _hedge_residual_volatility) with custom RunSpecs:
matrix = spectral_clean="market" (remove top eigenmode), graph = knn | signed_knn.
Saves per-window JSON incrementally so a crash loses at most one window.
"""
import json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import types as _t
_nv = _t.ModuleType("node2vec"); _nv.Node2Vec = None
sys.modules.setdefault("node2vec", _nv)  # spectral/netmf never touch Node2Vec

import benchmark.hedging_experiment as hx
from benchmark.run_benchmark import RunSpec, train_embedding
from benchmark.temporal_validation import load_prices, generate_windows
from benchmark.labels import load_all_levels

OUT = "results/signed_hedging_pcamarket"
os.makedirs(OUT, exist_ok=True)

UNIVERSE, PERIOD, TRAIN_P, TEST_H, NW, K, DIM = "sp500", "2y", "1y", "1y", 5, 10, 64
VARIANTS = {
    "corr_direct":        None,               # signed Pearson baseline
    "spectral_knn":       ("knn", "spectral"),
    "spectral_signedknn": ("signed_knn", "spectral"),
    "netmf_signedknn":    ("signed_knn", "netmf"),
}

prices = load_prices(UNIVERSE, PERIOD)
sub_labels = load_all_levels()["sub_industry"]
windows = generate_windows(prices, TRAIN_P, TEST_H, NW)

def spec_factory(graph_mode, embedder):
    def make(e):
        return RunSpec(method=e, universe=UNIVERSE, period=PERIOD,
                       embedder=e, graph_mode=graph_mode, knn_k=K,
                       spectral_clean="market", dimensions=DIM)
    return make

log = open(f"{OUT}/run.log", "w", buffering=1)
def P(*a):
    msg = " ".join(str(x) for x in a)
    print(msg); log.write(msg + "\n")

for ret_train, ret_test, label, train_end, test_end in windows:
    out_path = f"{OUT}/window_{label}.json"
    if os.path.exists(out_path):
        P(f"skip {label} (exists)"); continue
    t0 = time.time()
    tickers = [t for t in ret_train.columns if t in ret_test.columns and t in sub_labels]
    if len(tickers) < 50:
        P(f"SKIP {label}"); continue
    P(f"▶ {label}: {len(tickers)} stocks  [{time.time()-t0:.0f}s setup]")
    recs = []
    for vname, (gmode, emb) in VARIANTS.items():
        t1 = time.time()
        if gmode is None:
            sim = np.nan_to_num(ret_train[tickers].corr().values)
        else:
            e, _ = train_embedding(spec_factory(gmode, emb)(emb), 0, returns=ret_train[tickers])
            X = np.asarray([e[t] for t in tickers], dtype=np.float64)
            sim = hx._ordered_similarity(X)
        topk = hx._topk_from_sim(sim, K)
        v = hx._hedge_residual_volatility(ret_test[tickers], topk, tickers)
        rec = {"variant": vname, "window": label, "k": K,
               "vol_hedged_mean": float(np.nanmean(v["vol_hedged"])),
               "vol_unhedged_mean": float(np.nanmean(v["vol_unhedged"])),
               "vol_reduction_mean": float(np.nanmean(v["vol_reduction"])),
               "n": int(np.sum(~np.isnan(v["vol_hedged"]))),
               "seconds": round(time.time()-t1, 1)}
        recs.append(rec)
        # per-stock for paired tests
        pd.DataFrame({"ticker": tickers, "vol_unhedged": v["vol_unhedged"],
                      "vol_hedged": v["vol_hedged"]}).assign(
            variant=vname, window=label).to_csv(f"{OUT}/per_stock_{label}_{vname}.csv", index=False)
        P(f"  {vname:20s} hedged={rec['vol_hedged_mean']:.4f} red={rec['vol_reduction_mean']*100:.1f}% ({rec['seconds']}s)")
    json.dump(recs, open(out_path, "w"), indent=1)
    P(f"  ✔ {label} saved [{time.time()-t0:.0f}s total]")

P("ALL DONE")
