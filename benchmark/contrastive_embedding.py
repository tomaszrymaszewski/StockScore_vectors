"""
Replication of Dolphin et al.'s contrastive asset embeddings (arXiv:2407.18645, ICAIF 2024) --
the SOTA row (sector F1 0.69) on their own leaderboard, and the one method on it that is NOT a
correlation-graph Node2Vec.

Their method is deliberately simple and (helpfully) NOT a neural encoder: a free embedding matrix
E ∈ R^{N×d} (one learnable vector per asset, d=16) trained by gradient descent -- word2vec-style
negative sampling, but the positive/negative pairs are chosen by a HYPOTHESIS TEST on how often
two assets co-occur as top-k correlated peers across many short subwindows. We reproduce it in
pure NumPy with a hand-rolled Adam (no torch -- the repo stays dependency-light):

  1. **Co-occurrence** -- slide a `window`-day (22) window with `stride` (5); in each subwindow,
     Pearson-correlate every asset with all others and credit C[i,j] when j is in i's top-`topk`
     (5) most-correlated. n_i = sum_j C[i,j].
  2. **Statistical sampling** -- one-sample proportion z-test
        z_{ij} = (p_hat_{ij} - p0) / sqrt(p0 (1-p0) / n_i),  p_hat = C_{ij}/n_i,  p0 = 1/N,
     p-value = 1 - Phi(z). Positives: p < alpha_pos (0.05). Negatives: p > alpha_neg (0.30).
  3. **Contrastive loss** (Individual Sigmoid, their Eq. 7 -- their 0.68 variant, ~ their best 0.69):
        L_i = -mean_{j in P_i} log sig(e_i·e_j) - mean_{j in N_i} log(1 - sig(e_i·e_j))
     trained with Adam (lr 1e-3, 30 epochs, batch 128) and L2 weight decay (1e-3).

Exposed through the standard dispatcher as embedder `"contrastive"`, so it scores through the same
hit-rate / SVM-F1 / V-measure harness as every other method -- including the SVM-F1 sector
protocol that IS Dolphin's leaderboard metric.

Deviation from the paper: negatives are capped per anchor (`neg_cap`) for tractability on large
universes; at their N=611 this rarely binds. This is standard negative-sampling practice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def build_cooccurrence(returns: pd.DataFrame, window: int = 22, stride: int = 5, topk: int = 5
                       ) -> np.ndarray:
    """Co-occurrence counts C[i,j] = #subwindows where j is in i's top-`topk` correlated peers."""
    X = returns.values.astype(np.float64)            # (T, N)
    T, N = X.shape
    C = np.zeros((N, N), dtype=np.float64)
    topk = int(min(topk, N - 1))
    for s in range(0, T - window + 1, stride):
        W = X[s:s + window]                          # (window, N)
        Wc = W - W.mean(axis=0, keepdims=True)
        sd = Wc.std(axis=0, keepdims=True)
        sd[sd < 1e-12] = 1e-12
        Z = Wc / sd
        corr = (Z.T @ Z) / window                    # (N, N) within-window correlation
        np.fill_diagonal(corr, -np.inf)
        nbr = np.argpartition(-corr, kth=topk - 1, axis=1)[:, :topk]   # top-k per asset
        rows = np.repeat(np.arange(N), topk)
        np.add.at(C, (rows, nbr.ravel()), 1.0)
    return C


def hypothesis_sampling(C: np.ndarray, alpha_pos: float = 0.05, alpha_neg: float = 0.30):
    """One-sample proportion z-test on co-occurrence -> boolean positive / negative masks."""
    N = C.shape[0]
    n_i = C.sum(axis=1)
    n_safe = np.where(n_i > 0, n_i, 1.0)
    p0 = 1.0 / N
    p_hat = C / n_safe[:, None]
    se = np.sqrt(p0 * (1.0 - p0) / n_safe)[:, None]
    z = (p_hat - p0) / se
    pval = 1.0 - norm.cdf(z)                          # one-sided upper tail
    pos = pval < alpha_pos
    neg = pval > alpha_neg
    np.fill_diagonal(pos, False)
    np.fill_diagonal(neg, False)
    return pos, neg


def train_contrastive_embedding(
    returns: pd.DataFrame,
    dim: int = 16,
    *,
    window: int = 22,
    stride: int = 5,
    topk: int = 5,
    epochs: int = 30,
    lr: float = 1e-3,
    l2: float = 1e-3,
    batch: int = 128,
    alpha_pos: float = 0.05,
    alpha_neg: float = 0.30,
    neg_cap: int = 100,
    seed: int = 0,
    verbose: bool = False,
) -> dict[str, list[float]]:
    """Train Dolphin-style contrastive embeddings; return `{ticker: vector}`."""
    tickers = list(returns.columns)
    N = len(tickers)
    if N < 3:
        return {t: [0.0] * dim for t in tickers}

    C = build_cooccurrence(returns, window, stride, topk)
    pos, neg = hypothesis_sampling(C, alpha_pos, alpha_neg)
    pos_lists = [np.where(pos[i])[0] for i in range(N)]
    neg_lists = [np.where(neg[i])[0] for i in range(N)]
    if verbose:
        ap = float(np.mean([len(p) for p in pos_lists]))
        an = float(np.mean([len(n) for n in neg_lists]))
        print(f"   contrastive: N={N}, avg |P|={ap:.1f}, avg |N|={an:.1f} "
              f"(neg capped at {neg_cap})")

    rng = np.random.default_rng(seed)
    E = 0.01 * rng.standard_normal((N, dim))
    mE, vE = np.zeros_like(E), np.zeros_like(E)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(epochs):
        order = rng.permutation(N)
        for bs in range(0, N, batch):
            idx = order[bs:bs + batch]
            grad = np.zeros_like(E)
            for i in idx:
                ei = E[i]
                Pi, Ni = pos_lists[i], neg_lists[i]
                if len(Pi):
                    Ep = E[Pi]
                    coef = (_sigmoid(Ep @ ei) - 1.0) / len(Pi)     # d/d. of -log sig
                    grad[i] += coef @ Ep
                    np.add.at(grad, Pi, coef[:, None] * ei)
                if len(Ni):
                    Ns = Ni if len(Ni) <= neg_cap else rng.choice(Ni, neg_cap, replace=False)
                    En = E[Ns]
                    coef = _sigmoid(En @ ei) / len(Ns)              # d/d. of -log(1-sig)
                    grad[i] += coef @ En
                    np.add.at(grad, Ns, coef[:, None] * ei)
            grad += l2 * E                                          # weight decay
            t += 1
            mE = b1 * mE + (1 - b1) * grad
            vE = b2 * vE + (1 - b2) * (grad * grad)
            mhat = mE / (1 - b1 ** t)
            vhat = vE / (1 - b2 ** t)
            E -= lr * mhat / (np.sqrt(vhat) + eps)

    return {tickers[i]: E[i].astype(float).tolist() for i in range(N)}
