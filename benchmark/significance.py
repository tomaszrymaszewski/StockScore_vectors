"""
Significance testing with the correct unit of analysis.

Hit-rate is computed *per stock*, and the same stocks appear under every method, so
method-vs-method comparisons must be **paired** on stocks -- far more powerful than
comparing two independent means. This module operates on the per-stock match frames
produced by `benchmark.evaluation` and addresses both sources of randomness the reviewer
flagged:

* algorithmic stochasticity (random walks)  -> seed-averaged per-stock scores, plus
  mean +/- std across seeds reported by `run_benchmark`;
* sampling over the universe (which stocks)  -> paired bootstrap over stocks.

Tests
-----
* Paired **Wilcoxon signed-rank** on per-stock hit-fraction@k (continuous).
* **McNemar** on per-stock success@1 (binary hit/miss), exact via the binomial test.
* **Paired bootstrap** CI on the mean hit-rate difference.
* **Multiple-comparison correction** (Holm and Benjamini-Hochberg FDR) for the many
  pairwise comparisons across configs.

Every comparison reports an **effect size and CI next to the p-value** -- with ~2,700
stocks almost everything is "significant", so the honest signal is the effect size, and
`practically_meaningful` flags differences large enough to matter.

scipy-only (no statsmodels): BH uses `scipy.stats.false_discovery_control`; McNemar uses
`scipy.stats.binomtest`; Holm is implemented inline.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

from benchmark import evaluation as ev

# Below this absolute hit-rate difference, treat a "significant" result as negligible.
DEFAULT_MEANINGFUL_DELTA = 0.01


# ---------------------------------------------------------------------------
# Per-stock vectors (aligned + seed-averaged)
# ---------------------------------------------------------------------------


def seed_averaged_hit_fraction(frames: list[pd.DataFrame], k: int) -> pd.Series:
    """Average per-stock hit-fraction@k across seeds -> one vector per method.

    Averaging over seeds folds out random-walk noise while preserving the per-stock
    pairing needed for the signed-rank / bootstrap tests. Pass a single-element list to
    use one seed unchanged.
    """
    cols = [ev.per_stock_hit_fraction(f, k) for f in frames]
    return pd.concat(cols, axis=1).mean(axis=1).rename(f"hit_frac@{k}")


def align(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Restrict two per-stock series to their common stocks, in a shared order."""
    common = sorted(set(a.index) & set(b.index))
    return a.loc[common].to_numpy(), b.loc[common].to_numpy(), common


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for the signed-rank test, in [-1, 1].

    Positive => the first method tends to beat the second. Ties (diff == 0) are dropped,
    matching Wilcoxon's handling.
    """
    d = diff[diff != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    t_pos = ranks[d > 0].sum()
    t_neg = ranks[d < 0].sum()
    total = t_pos + t_neg
    return float((t_pos - t_neg) / total) if total else 0.0


# ---------------------------------------------------------------------------
# Paired tests
# ---------------------------------------------------------------------------


@dataclass
class PairedResult:
    method_a: str
    method_b: str
    level: str
    k: int
    n: int
    mean_a: float
    mean_b: float
    delta: float            # mean_a - mean_b
    test: str
    statistic: float
    p_value: float
    effect_size: float      # rank-biserial (Wilcoxon) or odds-ish (McNemar)
    ci_low: float
    ci_high: float

    def asdict(self) -> dict:
        return asdict(self)


def wilcoxon_compare(
    a: pd.Series,
    b: pd.Series,
    *,
    method_a: str,
    method_b: str,
    level: str,
    k: int,
    n_boot: int = 2000,
    ci: float = 0.95,
    boot_seed: int = 0,
) -> PairedResult:
    """Paired Wilcoxon signed-rank on per-stock hit-fraction@k, with bootstrap CI on the delta."""
    xa, xb, common = align(a, b)
    diff = xa - xb
    if np.allclose(diff, 0):
        p, statv = 1.0, 0.0
    else:
        statv, p = stats.wilcoxon(xa, xb, zero_method="wilcox", alternative="two-sided")
    lo, hi = paired_bootstrap_ci(xa, xb, n_boot=n_boot, ci=ci, seed=boot_seed)
    return PairedResult(
        method_a=method_a, method_b=method_b, level=level, k=k, n=len(common),
        mean_a=float(xa.mean()), mean_b=float(xb.mean()), delta=float(diff.mean()),
        test="wilcoxon", statistic=float(statv), p_value=float(p),
        effect_size=rank_biserial(diff), ci_low=lo, ci_high=hi,
    )


def mcnemar_compare(
    success_a: pd.Series,
    success_b: pd.Series,
    *,
    method_a: str,
    method_b: str,
    level: str,
) -> PairedResult:
    """Exact McNemar test on per-stock success@1 (binary hit/miss).

    Discordant pairs: n10 = a-hit & b-miss, n01 = a-miss & b-hit. Under H0 each discordant
    pair is equally likely either way -> exact two-sided binomial test on n10 vs n10+n01.
    """
    a = success_a.astype(bool)
    b = success_b.astype(bool)
    common = sorted(set(a.index) & set(b.index))
    a = a.loc[common].to_numpy()
    b = b.loc[common].to_numpy()
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    disc = n10 + n01
    if disc == 0:
        p = 1.0
    else:
        p = stats.binomtest(n10, disc, 0.5, alternative="two-sided").pvalue
    # Effect: proportion of discordant pairs favoring A minus favoring B.
    effect = (n10 - n01) / disc if disc else 0.0
    return PairedResult(
        method_a=method_a, method_b=method_b, level=level, k=1, n=len(common),
        mean_a=float(a.mean()), mean_b=float(b.mean()), delta=float(a.mean() - b.mean()),
        test="mcnemar", statistic=float(n10 - n01), p_value=float(p),
        effect_size=float(effect), ci_low=float("nan"), ci_high=float("nan"),
    )


# ---------------------------------------------------------------------------
# Bootstrap over the stock universe
# ---------------------------------------------------------------------------


def paired_bootstrap_ci(
    xa: np.ndarray, xb: np.ndarray, *, n_boot: int = 2000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI for the paired mean difference (mean(xa) - mean(xb)) over resampled stocks."""
    rng = np.random.default_rng(seed)
    n = len(xa)
    if n == 0:
        return float("nan"), float("nan")
    diff = xa - xb
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


def bootstrap_ci(
    x: pd.Series | np.ndarray, *, n_boot: int = 2000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI for a single method's mean hit-rate over resampled stocks."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = x[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------


def holm_correction(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def correct_pvalues(pvals) -> pd.DataFrame:
    """Return Holm and Benjamini-Hochberg (FDR) adjusted p-values side by side."""
    p = np.asarray(list(pvals), dtype=float)
    if len(p) == 0:
        return pd.DataFrame(columns=["p_value", "p_holm", "p_bh"])
    bh = stats.false_discovery_control(p, method="bh")
    return pd.DataFrame({"p_value": p, "p_holm": holm_correction(p), "p_bh": bh})


# ---------------------------------------------------------------------------
# High-level: compare many methods against a reference
# ---------------------------------------------------------------------------


def compare_methods(
    frames_by_method: dict[str, list[pd.DataFrame]],
    reference: str,
    *,
    level: str,
    k: int = 3,
    meaningful_delta: float = DEFAULT_MEANINGFUL_DELTA,
    boot_seed: int = 0,
) -> pd.DataFrame:
    """Wilcoxon every method vs `reference` at one level/k, with FDR/Holm correction.

    `frames_by_method[method]` is the list of per-stock match frames (one per seed) for
    that method at this granularity level. Returns a tidy DataFrame: one row per method,
    sorted by hit-rate, carrying delta-vs-reference, raw + corrected p-values, effect size,
    bootstrap CI, and a `practically_meaningful` flag.
    """
    ref_vec = seed_averaged_hit_fraction(frames_by_method[reference], k)
    results: list[PairedResult] = []
    for method, frames in frames_by_method.items():
        if method == reference:
            continue
        vec = seed_averaged_hit_fraction(frames, k)
        results.append(
            wilcoxon_compare(
                vec, ref_vec, method_a=method, method_b=reference,
                level=level, k=k, boot_seed=boot_seed,
            )
        )

    df = pd.DataFrame([r.asdict() for r in results])
    if df.empty:
        return df
    corrected = correct_pvalues(df["p_value"])
    df["p_holm"] = corrected["p_holm"].to_numpy()
    df["p_bh"] = corrected["p_bh"].to_numpy()
    df["significant_bh"] = df["p_bh"] < 0.05
    df["practically_meaningful"] = df["delta"].abs() >= meaningful_delta
    df["beats_reference"] = (df["delta"] > 0) & df["significant_bh"] & df["practically_meaningful"]
    return df.sort_values("mean_a", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Tiny self-check on synthetic per-stock frames.
    import numpy as np

    rng = np.random.default_rng(0)
    idx = [f"T{i}" for i in range(300)]

    def fake_frame(p):
        m = rng.random((len(idx), 10)) < p
        f = pd.DataFrame(m, index=idx, columns=[f"match_{i+1}" for i in range(10)])
        f.insert(0, "label", "X")
        return f

    frames = {"good": [fake_frame(0.6)], "ok": [fake_frame(0.5)], "ref": [fake_frame(0.4)]}
    out = compare_methods(frames, "ref", level="sub_industry", k=3)
    print(out[["method_a", "mean_a", "delta", "p_value", "p_bh", "effect_size",
               "ci_low", "ci_high", "beats_reference"]].to_string(index=False))
