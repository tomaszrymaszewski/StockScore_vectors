"""
Generate the paper's figures from the saved results CSVs (+ a few recomputed from vectorspaces).

Every figure is built from real, reproducible outputs under results/ (the same numbers RESULTS.md
reports). Run from the repo root:

    python figures/make_figures.py

Outputs PNG (200 dpi) + PDF (vector) to figures/ and prints a one-line caption for each. See
figures/README.md for the full description of every figure.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Ensure repo root is on path so local `benchmark` package can be imported regardless of how
# the script is invoked (e.g. `python figures/make_figures.py` from repo root).
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

UNIVERSES = ["sp500", "r1000", "r3000"]
UNI_LABEL = {"sp500": "S&P 500 (~470)", "r1000": "Russell 1000 (~880)", "r3000": "Russell 3000 (~2180)"}

# Consistent colours: cool = learned embeddings, warm = no-embedding raw baselines.
C = {
    "node2vec": "#2c7fb8", "spectral": "#41b6c4", "netmf": "#7fcdbb", "contrastive": "#253494",
    "raw_corr": "#fdae6b", "raw_returns": "#e6550d",
    # co-movement palette
    "corr": "#bdbdbd", "demean": "#9ecae1", "market_neutral": "#6baed6",
    "pca_market": "#08519c", "rmt": "#fb6a4a", "lasso": "#cb181d",
    "knn": "#2c7fb8", "fc": "#e6550d",
}
PRETTY = {
    "node2vec": "Node2Vec", "spectral": "Spectral\n(Laplacian)", "netmf": "NetMF\n(matrix fact.)",
    "contrastive": "Contrastive\n(Dolphin)", "raw_corr": "raw\ncorr-rows", "raw_returns": "raw\nreturns",
}
EMB = ["node2vec", "spectral", "netmf", "contrastive"]
RAW = ["raw_corr", "raw_returns"]
FAM_ORDER = EMB + RAW

# Single-column width for a two-column research paper (in inches).
COL_W = 3.5

# Short labels used inside compact figures.
UNI_SHORT = {"sp500": "S&P 500", "r1000": "R1000", "r3000": "R3000"}
PRETTY_SHORT = {
    "node2vec": "Node2Vec", "spectral": "Spectral", "netmf": "NetMF",
    "contrastive": "Contrastive", "raw_corr": "raw corr", "raw_returns": "raw returns",
}


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 13,
        "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelsize": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
        "legend.frameon": False, "figure.constrained_layout.use": True,
        "axes.axisbelow": True, "font.family": "DejaVu Sans",
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "lines.linewidth": 2, "lines.markersize": 8,
    })


def save(fig, name, caption):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  ✓ {name:28} — {caption}")


def bottom_legend(fig, handles, labels, ncol, title=None):
    """One legend centred BELOW all panels so it never covers data.

    Uses matplotlib's space-reserving "outside" placement (3.7+), which makes room for the
    legend under the axes rather than overlaying it.
    """
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol,
               fontsize=11, title=title, frameon=False)


def fam(universe):
    df = pd.read_csv(os.path.join(RES, f"embedding_families_{universe}_5y.csv")).set_index("method")
    return df


# ---------------------------------------------------------------------------
# Fig 1 — the headline: embedding families vs raw baselines, two metrics x 3 universes
# ---------------------------------------------------------------------------
def fig01():
    metrics = [("hit@3_sub_industry", "sub-industry hit@3"),
               ("svm_f1_sector", "sector SVM-F1")]
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 6.5))
    x = np.arange(len(FAM_ORDER)); w = 0.26
    for ax, (metric, title) in zip(axes, metrics):
        for j, u in enumerate(UNIVERSES):
            d = fam(u)
            vals = [d.loc[m, f"{metric}_mean"] for m in FAM_ORDER]
            errs = [d.loc[m, f"{metric}_std"] for m in FAM_ORDER]
            ax.bar(x + (j - 1) * w, vals, w, yerr=errs, capsize=2,
                   color=plt.cm.Greys(0.35 + j * 0.25), edgecolor="k", linewidth=0.4,
                   label=UNI_SHORT[u])
        ax.margins(y=0.22)
        ax.axvline(len(EMB) - 0.5, color="k", ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([PRETTY_SHORT[m] for m in FAM_ORDER], rotation=25, ha="right")
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("hit@3" if metric.startswith("hit") else "SVM-F1")
    handles = [Patch(facecolor=plt.cm.Greys(0.35 + j * 0.25), edgecolor="k", label=UNI_SHORT[u])
               for j, u in enumerate(UNIVERSES)]
    bottom_legend(fig, handles, [UNI_SHORT[u] for u in UNIVERSES], ncol=3)
    fig.suptitle("Embedding families vs raw features", fontsize=13, fontweight="bold")
    save(fig, "fig01_families_headline",
         "raw_returns tops every metric; all 4 embeddings beat raw corr-rows. (RESULTS §6d)")


# ---------------------------------------------------------------------------
# Fig 2 — baseline-dependent verdict: paired Wilcoxon Δ(embedding − raw) ± 95% CI
# ---------------------------------------------------------------------------
def fig02():
    from benchmark.run_benchmark import fetch_returns, RunSpec
    from benchmark import evaluation as ev, significance as sig
    from benchmark.embedding_methods import raw_correlation_vectors, raw_return_vectors
    from benchmark.labels import load_all_levels
    labels = load_all_levels()["sub_industry"]

    fig, axes = plt.subplots(3, 1, figsize=(COL_W, 8.0), sharex=True)
    for ax, u in zip(axes, UNIVERSES):
        ret = fetch_returns(RunSpec(universe=u, period="5y", graph_mode="knn"))
        frames = {}
        for m in EMB:
            p = os.path.join(RES, "vectors", f"{m}_fam_{u}_5y.json")
            emb = json.load(open(p))["seeds"]["0"]
            frames[m] = [ev.neighbor_match_frame(emb, labels, max_k=10)]
        frames["raw_corr"] = [ev.neighbor_match_frame(raw_correlation_vectors(ret), labels, max_k=10)]
        frames["raw_returns"] = [ev.neighbor_match_frame(raw_return_vectors(ret), labels, max_k=10)]
        offs = {"raw_corr": -0.15, "raw_returns": 0.15}
        for null, col, mk in [("raw_corr", "#08519c", "o"), ("raw_returns", "#e6550d", "s")]:
            res = sig.compare_methods({**{m: frames[m] for m in EMB}, null: frames[null]},
                                      reference=null, level="sub_industry", k=3).set_index("method_a")
            ys = np.arange(len(EMB))
            d = [res.loc[m, "delta"] for m in EMB]
            lo = [res.loc[m, "delta"] - res.loc[m, "ci_low"] for m in EMB]
            hi = [res.loc[m, "ci_high"] - res.loc[m, "delta"] for m in EMB]
            ax.errorbar(d, ys + offs[null], xerr=[lo, hi], fmt=mk, color=col, capsize=3,
                        ms=8, label=f"vs raw corr-rows" if null == "raw_corr" else "vs raw returns")
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(range(len(EMB))); ax.set_yticklabels([PRETTY_SHORT[m] for m in EMB])
        ax.set_title(UNI_SHORT[u], fontsize=13)
        ax.invert_yaxis()
    axes[-1].set_xlabel("Δ sub-industry hit@3")
    h, l = axes[0].get_legend_handles_labels()
    bottom_legend(fig, h, l, ncol=2)
    fig.suptitle("Baseline-dependent verdict", fontsize=13, fontweight="bold")
    save(fig, "fig02_baseline_forest",
         "paired Wilcoxon Δ ± 95% bootstrap CI vs each raw baseline, all 3 universes. (RESULTS §6d)")


# ---------------------------------------------------------------------------
# Fig 3 — dimension robustness
# ---------------------------------------------------------------------------
def fig03():
    df = pd.read_csv(os.path.join(RES, "dim_sweep_sp500.csv"))
    dims = sorted(df[df.dim > 0].dim.unique())
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 6.0))
    for ax, (metric, title) in zip(axes, [("hit@3_sub_industry", "sub-industry hit@3"),
                                          ("svm_f1_sector", "sector SVM-F1")]):
        for m in EMB:
            sub = df[(df.embedder == m) & (df.dim > 0)].sort_values("dim")
            ax.plot(sub.dim, sub[f"{metric}_mean"], "-o", color=C[m], label=PRETTY_SHORT[m],
                    markersize=6)
        for m, ls in [("raw_returns", "--"), ("raw_corr", ":")]:
            v = df[(df.embedder == m)][f"{metric}_mean"].iloc[0]
            ax.axhline(v, color=C[m], ls=ls, lw=2, label=PRETTY_SHORT[m])
        ax.set_xscale("log", base=2); ax.set_xticks(dims); ax.set_xticklabels(dims)
        ax.set_xlabel("embedding dimension"); ax.set_title(title, fontsize=13)
        ax.set_ylabel("hit@3" if metric.startswith("hit") else "SVM-F1")
    h, l = axes[0].get_legend_handles_labels()
    bottom_legend(fig, h, l, ncol=2)
    fig.suptitle("Dimension robustness", fontsize=13, fontweight="bold")
    save(fig, "fig03_dimension_sweep",
         "embeddings mature by d≈32 then plateau, always between the two raw baselines. (RESULTS §6f)")


# ---------------------------------------------------------------------------
# Fig 4 — Node2Vec p/q sensitivity heatmap
# ---------------------------------------------------------------------------
def fig04():
    df = pd.read_csv(os.path.join(RES, "pq_sweep_sp500.csv"))
    piv = df.pivot_table(index="p", columns="q", values="hit@3_sub_industry_mean")
    fig, ax = plt.subplots(figsize=(COL_W, 3.5))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("q"); ax.set_ylabel("p")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center",
                    color="w" if piv.values[i, j] < piv.values.mean() else "k", fontsize=10)
    # mark ours (p=0.8,q=7.2) and Sarmah (p=2,q=0.5)
    def cell(p, q):
        return list(piv.columns).index(q), list(piv.index).index(p)
    for (p, q, col) in [(0.8, 7.2, "red"), (2.0, 0.5, "white")]:
        if p in piv.index and q in piv.columns:
            cx, cy = cell(p, q)
            ax.add_patch(plt.Rectangle((cx - .5, cy - .5), 1, 1, fill=False, edgecolor=col, lw=2.5))
    fig.colorbar(im, ax=ax, label="hit@3", shrink=0.8)
    rng = df["hit@3_sub_industry_mean"]
    ax.set_title("Node2Vec p/q sensitivity", fontsize=13)
    save(fig, "fig04_pq_heatmap",
         "p/q grid spread 0.045 — our config isn't a peak; Sarmah's gives the same verdict. (RESULTS §6f)")


# ---------------------------------------------------------------------------
# Fig 5 — temporal stability: hit@3 built on window A vs disjoint window B
# ---------------------------------------------------------------------------
def fig05():
    order = ["raw_returns", "node2vec", "spectral", "netmf", "raw_corr"]
    name = {"raw_returns": "raw returns", "node2vec": "Node2Vec", "spectral": "spectral",
            "netmf": "netmf", "raw_corr": "raw corr-rows"}
    cmap = {"raw_returns": C["raw_returns"], "node2vec": C["node2vec"], "spectral": C["spectral"],
            "netmf": C["netmf"], "raw_corr": C["raw_corr"]}

    agg_path = os.path.join(RES, "temporal_sweep_aggregate.csv")
    per_path = os.path.join(RES, "temporal_sweep_per_window.csv")
    has_sweep = os.path.exists(agg_path) and os.path.exists(per_path)
    if has_sweep:
        agg = pd.read_csv(agg_path)
        per = pd.read_csv(per_path)

    fig, axes = plt.subplots(3, 1, figsize=(COL_W, 8.0), sharey=True)
    for ax, u in zip(axes, UNIVERSES):
        df = pd.read_csv(os.path.join(RES, f"temporal_{u}_5y.csv")).set_index("method")
        for m in order:
            a, b = df.loc[m, "gics_hit3_A"], df.loc[m, "gics_hit3_B"]
            ax.plot([0, 1], [a, b], "-o", color=cmap[m], ms=8, lw=2.5)
        if has_sweep:
            sub = agg[(agg.universe == u) & (agg.train_period == "1y") &
                      (agg.test_horizon == "1y")].set_index("method")
            pw = per[(per.universe == u) & (per.train_period == "1y") &
                     (per.test_horizon == "1y")]
            for m in order:
                if m not in sub.index:
                    continue
                tm = float(sub.loc[m, "gics_hit3_train_mean"])
                ts = float(sub.loc[m, "gics_hit3_test_mean"])
                tstd = float(sub.loc[m, "gics_hit3_train_std"])
                sstd = float(sub.loc[m, "gics_hit3_test_std"])
                pw_m = pw[(pw.method == m) & pw["gics_hit3_train"].notna()]
                if len(pw_m):
                    ax.scatter([0.03] * len(pw_m), pw_m["gics_hit3_train"],
                               color=cmap[m], alpha=0.2, s=12, zorder=5)
                    ax.scatter([0.97] * len(pw_m), pw_m["gics_hit3_test"],
                               color=cmap[m], alpha=0.2, s=12, zorder=5)
                ax.plot([0, 1], [tm, ts], "--", color=cmap[m], lw=1.2, alpha=0.6)
                ax.errorbar([0, 1], [tm, ts], yerr=[tstd, sstd],
                            fmt="none", color=cmap[m], capsize=2, alpha=0.3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["train", "test"])
        ax.set_xlim(-0.15, 1.2); ax.set_title(UNI_SHORT[u], fontsize=13)
    axes[0].set_ylabel("GICS sub-industry hit@3")
    method_handles = [Line2D([], [], color=cmap[m], ls="-", lw=2, marker="o", ms=6,
                             label=name[m]) for m in order]
    style_handles = [
        Line2D([], [], color="k", ls="-", lw=2, marker="o", ms=6, label="half/half split"),
        Line2D([], [], color="k", ls="--", lw=1.2, alpha=0.6, label="1y/1y rolling"),
    ]
    all_handles = method_handles + style_handles
    bottom_legend(fig, all_handles, [h.get_label() for h in all_handles], ncol=2)
    fig.suptitle("GICS recovery across disjoint windows", fontsize=13, fontweight="bold")
    save(fig, "fig05_temporal_stability",
         "train≈test across half/half + rolling 1y/1y windows; ordering preserved. (RESULTS §6e)")


# ---------------------------------------------------------------------------
# Fig 6 — out-of-sample: future-window correlation of past neighbours
# ---------------------------------------------------------------------------
def fig06():
    # Three DISTINCT no-model neighbour rankings — kept separate (do NOT conflate them):
    #   corr_direct = rank by the raw correlation matrix directly (first-order);
    #   raw_returns = cosine of the standardized return series;
    #   raw_corr    = cosine of correlation-row vectors (second-order).
    methods = ["node2vec", "spectral", "netmf", "raw_corr", "corr_direct (null)", "raw_returns"]
    lab = {"node2vec": "Node2Vec", "spectral": "Spectral", "netmf": "NetMF",
           "raw_corr": "raw corr", "corr_direct (null)": "raw correlation",
           "raw_returns": "raw returns"}

    agg_path = os.path.join(RES, "temporal_sweep_aggregate.csv")
    has_sweep = os.path.exists(agg_path)
    if has_sweep:
        agg = pd.read_csv(agg_path)
        if ((agg.train_period == "1y") & (agg.test_horizon == "1y")).any():
            configs = [("half", "half", "half/half split"), ("1y", "1y", "1y/1y rolling")]
        else:
            configs = [("half", "half", "half/half split")]
    else:
        configs = [("half", "half", "half/half split")]

    n_panels = len(configs)
    if n_panels == 1:
        fig, ax_list = plt.subplots(figsize=(COL_W, 4.0))
        ax_list = [ax_list]
    else:
        fig, ax_list = plt.subplots(n_panels, 1, figsize=(COL_W, 6.0))

    x = np.arange(len(methods)); w = 0.26
    for pi, (tp, th, cfg_label) in enumerate(configs):
        ax = ax_list[pi]
        for j, u in enumerate(UNIVERSES):
            if tp == "1y":
                sub = agg[(agg.universe == u) & (agg.train_period == tp) &
                          (agg.test_horizon == th)].set_index("method")
                vals = [float(sub.loc[m, "future_corr_mean"]) if m in sub.index else 0
                        for m in methods]
                errs = [float(sub.loc[m, "future_corr_std"]) if m in sub.index else 0
                        for m in methods]
                oracle = float(sub.loc["oracle_B (ceiling)", "future_corr_mean"]) \
                         if "oracle_B (ceiling)" in sub.index else 0
            else:
                df = pd.read_csv(os.path.join(RES, f"temporal_{u}_5y.csv")).set_index("method")
                vals = [float(df.loc[m, "future_corr"]) if m in df.index else 0 for m in methods]
                errs = [0] * len(methods)
                oracle = float(df.loc["oracle_B (ceiling)", "future_corr"]) \
                         if "oracle_B (ceiling)" in df.index else 0
            ax.bar(x + (j - 1) * w, vals, w, yerr=errs, capsize=2,
                   color=plt.cm.Greys(0.4 + j * 0.22), edgecolor="k", linewidth=0.4,
                   label=UNI_SHORT[u])
            ax.hlines(oracle, x[0] + (j - 1) * w - w / 2, x[-1] + (j - 1) * w + w / 2,
                      color="k", ls="--", lw=1)
        ax.axvline(2.5, color="k", ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([lab[m] for m in methods], rotation=20, ha="right")
        ax.set_ylabel("future ρ")
        ax.set_title(cfg_label, fontsize=13)
        ax.margins(y=0.16)
    h, l = ax_list[0].get_legend_handles_labels()
    bottom_legend(fig, h, l, ncol=3)
    fig.suptitle("Future ρ of top-3 past neighbours", fontsize=13, fontweight="bold")
    save(fig, "fig06_temporal_future_corr",
         "raw correlation leads future co-movement across both configs; no OOS reversal. (RESULTS §6e)")


# ---------------------------------------------------------------------------
# Fig 7 — Dolphin leaderboard with literature reference lines
# ---------------------------------------------------------------------------
def fig07():
    df = pd.read_csv(os.path.join(RES, "dolphin_comparison_modal.csv"))
    g = df.groupby("method")["svm_f1_sector"].agg(["mean", "std"])
    order = g["mean"].sort_values().index.tolist()
    colors = [C.get(m, "#888") for m in order]
    fig, ax = plt.subplots(figsize=(COL_W, 5.5))
    y = np.arange(len(order))
    ax.barh(y, g.loc[order, "mean"], xerr=g.loc[order, "std"], color=colors,
            edgecolor="k", linewidth=0.4, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.set_xlabel("sector SVM-F1")
    ref_handles = []
    for x, lab, c in [(0.60, "Dolphin'22", "#888"), (0.61, "Sarmah", "#888"),
                      (0.66, "Dolphin'23", "#555"), (0.69, "Contrastive", "#000")]:
        ax.axvline(x, color=c, ls="--", lw=1)
        ref_handles.append(Line2D([], [], color=c, ls="--", lw=1, label=lab))
    bottom_legend(fig, ref_handles, [h.get_label() for h in ref_handles], ncol=2)
    ax.set_xlim(0.55, 0.80)
    ax.set_title("Dolphin benchmark", fontsize=13)
    save(fig, "fig07_dolphin_leaderboard",
         "our methods + raw baselines vs Dolphin's reported leaderboard lines. (RESULTS §6b)")


# ---------------------------------------------------------------------------
# Fig 8 — metric disagreement: retrieval vs clustering
# ---------------------------------------------------------------------------
def fig08():
    fig, ax = plt.subplots(figsize=(COL_W, 3.5))
    markers = {"sp500": "o", "r1000": "s", "r3000": "^"}
    for u in UNIVERSES:
        d = fam(u)
        for m in FAM_ORDER:
            ax.scatter(d.loc[m, "hit@3_sub_industry_mean"], d.loc[m, "v_sector_mean"],
                       color=C[m], marker=markers[u], s=120, edgecolor="k", linewidth=0.5, zorder=3)
    for m in FAM_ORDER:
        xs = [fam(u).loc[m, "hit@3_sub_industry_mean"] for u in UNIVERSES]
        ys = [fam(u).loc[m, "v_sector_mean"] for u in UNIVERSES]
        ax.plot(xs, ys, "-", color=C[m], alpha=0.35, lw=1.5, zorder=1)
        ax.text(xs[-1] + 0.004, ys[-1], PRETTY[m].replace("\n", " "), fontsize=10,
                color=C[m], va="center")
    ax.set_xlabel("sub-industry hit@3")
    ax.set_ylabel("sector V-measure")
    ax.margins(x=0.12)
    leg = [Line2D([], [], marker=markers[u], color="k", ls="", label=UNI_SHORT[u]) for u in UNIVERSES]
    bottom_legend(fig, leg, [UNI_SHORT[u] for u in UNIVERSES], ncol=3)
    ax.set_title("Metric disagreement", fontsize=13)
    save(fig, "fig08_metric_disagreement",
         "retrieval vs clustering plane; skip-gram family high-V, raw_returns high-hit. (RESULTS §6d)")


# ---------------------------------------------------------------------------
# Fig 9 — co-movement removal pays off more at scale
# ---------------------------------------------------------------------------
def fig09():
    rows = []
    for u in UNIVERSES:
        cm = pd.read_csv(os.path.join(RES, f"comovement_{u}_modal.csv"))
        for m in ["corr", "demean", "market_neutral", "pca_market", "rmt"]:
            v = cm[cm.method == m]["sub_industry.hit_rate@3"].mean()
            rows.append((u, m, v))
        la = pd.read_csv(os.path.join(RES, f"lasso_{u}_modal.csv"))
        rows.append((u, "lasso", la[la.method == "lasso"]["sub_industry.hit_rate@3"].mean()))
    d = pd.DataFrame(rows, columns=["u", "m", "v"])
    fig, ax = plt.subplots(figsize=(COL_W, 3.0))
    xs = range(len(UNIVERSES))
    for m in ["corr", "demean", "market_neutral", "pca_market", "rmt", "lasso"]:
        ys = [d[(d.u == u) & (d.m == m)]["v"].iloc[0] for u in UNIVERSES]
        lw = 3 if m == "pca_market" else 1.6
        ax.plot(xs, ys, "-o", color=C[m], lw=lw, ms=7 if m == "pca_market" else 6,
                label=m)
    ax.set_xticks(list(xs)); ax.set_xticklabels(["S&P 500", "R1000", "R3000"])
    ax.set_ylabel("sub-industry hit@3")
    h, l = ax.get_legend_handles_labels(); bottom_legend(fig, h, l, ncol=2)
    ax.set_title("PCA market-mode wins", fontsize=13)
    save(fig, "fig09_comovement_scale",
         "Node2Vec on cleaned matrices; pca_market best, advantage grows sp500→r3000. (RESULTS §4)")


# ---------------------------------------------------------------------------
# Fig 10 — Sarmah density sweep: escape the MST tree, don't over-densify
# ---------------------------------------------------------------------------
def fig10():
    df = pd.read_csv(os.path.join(RES, "sarmah_density_modal.csv"))
    g = df.groupby(["method", "param"]).agg(edges=("edges", "mean"), V_avg=("V_avg", "mean"),
                                            hit3=("hit3_sub", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(COL_W, 3.0))
    for meth, col, lab in [("knn", C["knn"], "kNN"), ("fc", C["fc"], "Forced Conn.")]:
        s = g[g.method == meth].sort_values("edges")
        ax.plot(s.edges, s.V_avg, "-o", color=col, label=lab, markersize=6)
    tail = os.path.join(RES, "sarmah_density_tail.csv")
    if os.path.exists(tail):
        t = pd.read_csv(tail); t = t[t.method == "knn"].sort_values("edges")
        kk = g[g.method == "knn"].sort_values("edges")
        ax.plot([kk.edges.iloc[-1]] + list(t.edges), [kk.V_avg.iloc[-1]] + list(t.V_avg),
                ":o", color=C["knn"], alpha=0.7, mfc="white", label="kNN dense tail", markersize=6)
        ax.scatter([t.edges.iloc[-1]], [t.V_avg.iloc[-1]], color=C["knn"], s=80, zorder=5)
        ax.text(t.edges.iloc[-1], t.V_avg.iloc[-1] + 0.006, "complete graph",
                ha="center", va="bottom", fontsize=9, color=C["knn"])
    peak = g[g.method == "knn"]["V_avg"].max()
    ax.axhline(peak, color=C["knn"], ls=":", lw=1, alpha=0.6)
    ax.text(g.edges.min(), peak + 0.002, f"sparse k≈10 peak ({peak:.3f})", fontsize=9, color=C["knn"])
    ax.axhline(0.638, color="k", ls="--", lw=1.5)
    ax.text(ax.get_xlim()[1], 0.638, "Sarmah MST", va="bottom", ha="right", fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("# edges"); ax.set_ylabel("V-measure")
    h, l = ax.get_legend_handles_labels(); bottom_legend(fig, h, l, ncol=2)
    ax.set_title("Escape MST tree → V jumps", fontsize=13)
    save(fig, "fig10_sarmah_density",
         "V peaks at the sparsest setting; extended to the complete graph V plateaus ~0.677. (RESULTS §6a-bis)")


# ---------------------------------------------------------------------------
# Fig 11 — V-measure inflates with granularity (why hit-rate is the honest metric)
# ---------------------------------------------------------------------------
def fig11():
    lit = pd.read_csv(os.path.join(RES, "literature_metrics.csv"))
    lit = lit[(lit.universe == "sp500") & (lit.method == "pca_market")]
    levels = ["sector", "industry_group", "industry", "sub_industry"]
    def series(metric):
        return [lit[(lit.level == lv) & (lit.metric == metric)]["mean"].mean() for lv in levels]
    K = [int(round(lit[(lit.level == lv) & (lit.metric == "n_clusters")]["mean"].mean())) for lv in levels]
    cm = pd.read_csv(os.path.join(RES, "comovement_sp500_modal.csv"))
    cm = cm[cm.method == "pca_market"]
    hit = [cm[f"{lv}.hit_rate@3"].mean() for lv in levels]
    fig, ax = plt.subplots(figsize=(COL_W, 3.0))
    xs = range(len(levels))
    ax.plot(xs, series("v_measure"), "-o", color="#d95f02", lw=2.5, label="V-measure", markersize=7)
    ax.plot(xs, series("ari"), "-s", color="#1b9e77", lw=2, label="ARI", markersize=7)
    ax.plot(xs, hit, "-^", color="#7570b3", lw=2, label="hit@3", markersize=7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{lv}\n(K={k})" for lv, k in zip(levels, K)])
    ax.set_ylabel("score")
    h, l = ax.get_legend_handles_labels(); bottom_legend(fig, h, l, ncol=3)
    ax.set_title("Why hit-rate is primary", fontsize=13)
    save(fig, "fig11_vmeasure_inflation",
         "V-measure climbs sector→sub-industry while ARI/hit@3 drop. (RESULTS §2)")


# ---------------------------------------------------------------------------
# Fig 12 — compact scorecard: method × metric heatmap (S&P 500)
# ---------------------------------------------------------------------------
def fig12():
    d = fam("sp500")
    cols = [("hit@3_sub_industry_mean", "sub hit@3"), ("hit@3_sector_mean", "sector hit@3"),
            ("svm_f1_sector_mean", "sector F1"), ("svm_f1_sub_industry_mean", "sub F1"),
            ("v_sector_mean", "sector V")]
    M = np.array([[d.loc[m, c] for c, _ in cols] for m in FAM_ORDER])
    # column-normalise to [0,1] so each metric's best=1 (relative ranking, since scales differ)
    Mn = (M - M.min(0)) / (M.max(0) - M.min(0) + 1e-9)
    fig, ax = plt.subplots(figsize=(COL_W, 4.0))
    im = ax.imshow(Mn, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c for _, c in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(FAM_ORDER)))
    ax.set_yticklabels([PRETTY_SHORT[m] for m in FAM_ORDER])
    ax.tick_params(axis='y', pad=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="k")
    ax.set_title("Scorecard (S&P 500)", fontsize=13)
    fig.colorbar(im, ax=ax, label="rank", shrink=0.8)
    save(fig, "fig12_scorecard",
         "method×metric heatmap; raw_returns greenest on hit/SVM, skip-gram on V. (RESULTS §6d)")


# ---------------------------------------------------------------------------
# Whole-document export: render an entire markdown file (text + tables + the embedded
# figures above) to a single PDF, so RESULTS.md / README.md can be read/shared as one PDF.
# ---------------------------------------------------------------------------
def _balance_table_columns(html: str) -> str:
    """Force balanced, explicit column widths on every <table>.

    xhtml2pdf's automatic column sizing is poor — it routinely gives the first column ~90% and
    squishes the rest to unreadable slivers, and it ignores both <colgroup> and `table-layout:
    fixed`. The one thing it DOES honour is a `width` set directly on each cell, so we stamp an
    explicit width onto every <th>/<td> (near-even, with a mild boost to the first column, which
    is usually the label/description). Combined with `_wrappable_code`, content then wraps to fit.
    """
    def per_table(m: "re.Match") -> str:
        table = m.group(0)
        rows = re.findall(r"<tr>(.*?)</tr>", table, flags=re.S)
        if not rows:
            return table
        ncol = len(re.findall(r"<t[hd][\s>]", rows[0]))
        if ncol < 2:
            return table
        # Width proportional to each column's LONGEST cell (sqrt-damped so one long column can't
        # swallow the table, floored so nothing is squished), instead of a fixed first-col boost —
        # this sizes path columns wider than a short "§6a" label column automatically.
        maxlen = [1] * ncol
        for row in rows:
            cells = re.findall(r"<(t[hd])(?:\s[^>]*)?>(.*?)</\1>", row, flags=re.S)
            for i, (_, cell) in enumerate(cells[:ncol]):
                text = re.sub(r"<[^>]+>", "", cell).strip()
                maxlen[i] = max(maxlen[i], len(text))
        raw = [x ** 0.5 for x in maxlen]
        s = sum(raw)
        frac = [max(0.09, r / s) for r in raw]            # floor each column at 9%
        s2 = sum(frac)
        widths = [f"{100 * f / s2:.1f}%" for f in frac]

        def per_row(rm: "re.Match") -> str:
            counter = {"i": 0}

            def per_cell(cm: "re.Match") -> str:
                tag, attrs = cm.group(1), cm.group(2) or ""
                w = widths[min(counter["i"], ncol - 1)]
                counter["i"] += 1
                if 'style="' in attrs:
                    attrs = attrs.replace('style="', f'style="width:{w};', 1)
                else:
                    attrs = f'{attrs} style="width:{w}"'
                return f"<{tag}{attrs}>"

            return re.sub(r"<(t[hd])((?:\s[^>]*)?)>", per_cell, rm.group(0))

        return re.sub(r"<tr>.*?</tr>", per_row, table, flags=re.S)

    return re.sub(r"<table\b.*?</table>", per_table, html, flags=re.S)


def export_markdown_pdf(md_name="RESULTS.md", out_name="RESULTS.pdf"):
    """Render `md_name` (with its embedded figures) to a single `out_name` PDF at the repo root.

    Pure-Python path (no system deps): markdown -> HTML -> PDF via xhtml2pdf. Unicode (ρ, Δ, ≈,
    ↑↓, √, …) is covered by registering matplotlib's bundled DejaVu fonts; table columns are
    balanced (see `_balance_table_columns`) so xhtml2pdf cannot squish them.
    """
    try:
        import markdown
        from xhtml2pdf import pisa
        import matplotlib.font_manager as fm
    except Exception as e:
        print(f"  ✗ {out_name:28} — needs `pip install markdown xhtml2pdf` ({type(e).__name__})")
        return
    sans = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
    sans_b = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
    mono = fm.findfont(fm.FontProperties(family="DejaVu Sans Mono"))
    md_text = open(os.path.join(ROOT, md_name), encoding="utf-8").read()
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    body = body.replace("<img ", '<img style="width:240pt" ')      # scale figures to half page width
    body = _balance_table_columns(body)                            # stop xhtml2pdf squishing columns
    css = f"""
      @font-face {{ font-family: DV;  src: url("{sans}"); }}
      @font-face {{ font-family: DV;  font-weight: bold; src: url("{sans_b}"); }}
      @font-face {{ font-family: DVM; src: url("{mono}"); }}
      @page {{ size: a4 portrait; margin: 1.4cm; }}
      body {{ font-family: DV; font-size: 9pt; line-height: 1.32; color: #111; }}
      h1 {{ font-size: 18pt; }}
      h2 {{ font-size: 13pt; border-bottom: 1pt solid #aaa; padding-bottom: 2pt; margin-top: 12pt; }}
      h3 {{ font-size: 11pt; margin-top: 8pt; }}
      table {{ border-collapse: collapse; margin: 4pt 0; width: 100%; table-layout: fixed; }}
      th, td {{ border: 0.5pt solid #bbb; padding: 2pt 3pt; font-size: 7pt;
                vertical-align: top; word-wrap: break-word; overflow-wrap: break-word;
                word-break: break-all; }}
      th {{ background: #eee; }}
      td code, th code {{ font-size: 6.5pt; }}
      code, pre {{ font-family: DVM; font-size: 7.3pt; background: #f5f5f5; }}
      pre {{ padding: 4pt; }}
      img {{ width: 240pt; }}
      em {{ color: #333; }}
    """
    html = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"

    def link_cb(uri, rel):                       # resolve relative image paths against repo root
        return uri if os.path.isabs(uri) else os.path.join(ROOT, uri)

    out = os.path.join(ROOT, out_name)
    with open(out, "wb") as f:
        res = pisa.CreatePDF(html, dest=f, link_callback=link_cb, encoding="utf-8")
    status = "OK" if not res.err else f"{res.err} layout warnings"
    print(f"  ✓ {out_name:28} — full rendered markdown → PDF ({status})")


def main():
    style()
    print("Generating figures → figures/")
    for fn in [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10, fig11, fig12]:
        try:
            fn()
        except Exception as e:
            print(f"  ✗ {fn.__name__} FAILED: {type(e).__name__}: {e}")
    print("Exporting whole documents to PDF → repo root")
    export_markdown_pdf("RESULTS.md", "RESULTS.pdf")
    export_markdown_pdf("README.md", "README.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
