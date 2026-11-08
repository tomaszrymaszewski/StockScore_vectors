"""Graph construction from a returns matrix (ported from train_vectors.GraphBuilder)."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV

from .config import Config

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds a stock similarity graph from returns according to a Config."""

    def __init__(self, config: Config):
        self.config = config

    def build(self, returns: pd.DataFrame) -> nx.Graph:
        """Build the graph: optional co-movement removal -> matrix -> edge filtering."""
        if self.config.cross_sectional_demean:
            returns = self._cross_sectional_demean(returns)
        if self.config.market_neutral:
            returns = self._residualize_market(returns)

        G = nx.Graph()
        col_names = returns.columns
        G.add_nodes_from(col_names)

        matrix = self._compute_matrix(returns)

        if self.config.graph_filtering == "mst":
            G = self._build_mst(G, matrix, col_names)
        elif self.config.graph_filtering == "threshold":
            G = self._build_threshold(G, matrix, col_names, returns)
        elif self.config.graph_filtering in ("knn", "signed_knn"):
            G = self._build_knn(G, matrix, col_names)
        else:
            raise ValueError(
                f"Unknown graph filtering method: {self.config.graph_filtering!r}")

        return G

    # -------------------------------------------------------------------------
    # Edge filtering strategies
    # -------------------------------------------------------------------------

    def _build_mst(self, G: nx.Graph, matrix: np.ndarray, col_names) -> nx.Graph:
        logger.info("Building Maximum Spanning Tree (MST)...")
        all_edges = []
        for i in range(len(col_names)):
            for j in range(i + 1, len(col_names)):
                if self.config.mst_distance_transform:
                    w = float(matrix[i, j])  # signed rho
                    if np.isfinite(w):
                        all_edges.append((col_names[i], col_names[j], w))
                    continue
                w = np.abs(matrix[i, j])
                if self.config.mst_squared_corr:
                    w = w ** 2
                if np.isfinite(w) and w > 1e-6:
                    all_edges.append((col_names[i], col_names[j], w))

        temp_G = nx.Graph()
        temp_G.add_weighted_edges_from(all_edges)
        G = nx.maximum_spanning_tree(temp_G)

        if self.config.mst_extra_edges_pct > 0:
            mst_edges = set(tuple(sorted((u, v))) for u, v in G.edges())
            sorted_potential_edges = sorted(all_edges, key=lambda x: x[2], reverse=True)
            remaining_edges = [
                e for e in sorted_potential_edges
                if tuple(sorted((e[0], e[1]))) not in mst_edges
            ]
            num_to_add = int(len(remaining_edges) * self.config.mst_extra_edges_pct)
            logger.info("Adding top %.1f%% of remaining edges (%d edges)...",
                        self.config.mst_extra_edges_pct * 100, num_to_add)
            for i in range(num_to_add):
                u, v, w = remaining_edges[i]
                G.add_edge(u, v, weight=w)

        logger.info("Graph (MST + extras): %d nodes, %d edges",
                    G.number_of_nodes(), G.number_of_edges())
        return G

    def _build_threshold(self, G: nx.Graph, matrix: np.ndarray, col_names,
                         returns: pd.DataFrame) -> nx.Graph:
        thresh = self.config.correlation_threshold
        edge_count = 0
        for i in range(len(col_names)):
            for j in range(i + 1, len(col_names)):
                w = np.abs(matrix[i, j])
                if w > thresh and np.isfinite(w):
                    G.add_edge(col_names[i], col_names[j], weight=float(w))
                    edge_count += 1

        if self.config.repair_islands:
            self._rescue_islands(G, matrix, col_names, returns)
        logger.info("Graph (threshold > %s%s): %d nodes, %d edges",
                    thresh,
                    " + repair" if self.config.repair_islands else " (no repair)",
                    G.number_of_nodes(), edge_count)
        return G

    def _build_knn(self, G: nx.Graph, matrix: np.ndarray, col_names) -> nx.Graph:
        signed = self.config.graph_filtering == "signed_knn"
        k = min(self.config.knn_k, len(col_names) - 1)
        n = len(col_names)
        if signed:
            sim_matrix = np.nan_to_num(matrix)
        else:
            sim_matrix = np.abs(np.nan_to_num(matrix))
        np.fill_diagonal(sim_matrix, -np.inf)
        for i in range(n):
            nbrs = np.argpartition(-sim_matrix[i], kth=k - 1)[:k]
            for j in nbrs:
                w_raw = float(matrix[i, j]) if signed else float(sim_matrix[i, j])
                w = abs(w_raw) if signed else w_raw
                # Floor to a tiny positive weight so walk-sampler weights never sum to 0.
                w = max(w, 1e-8) if np.isfinite(w) else 1e-8
                G.add_edge(col_names[i], col_names[j], weight=w)
        logger.info("Graph (%s k=%d): %d nodes, %d edges",
                    "signed kNN" if signed else "kNN", k,
                    G.number_of_nodes(), G.number_of_edges())
        return G

    # -------------------------------------------------------------------------
    # Co-movement removal
    # -------------------------------------------------------------------------

    def _cross_sectional_demean(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Subtract each day's equal-weighted cross-sectional mean return."""
        logger.info("Cross-sectional demeaning (equal-weighted daily mean removed)...")
        return returns.sub(returns.mean(axis=1), axis=0)

    def _residualize_market(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Regress out the market factor from each stock's returns (keep residuals)."""
        cols = list(returns.columns)
        mkt = self.config.market_ticker
        if mkt in cols:
            market = returns[mkt].to_numpy(dtype=float)
            keep = [c for c in cols if c != mkt]
            logger.info("Market-neutral residualization (factor = %s)...", mkt)
        else:
            market = returns.mean(axis=1).to_numpy(dtype=float)
            keep = cols
            logger.info("Market-neutral residualization (factor = equal-weighted proxy)...")

        R = returns[keep].to_numpy(dtype=float)
        m = market - market.mean()
        var = float(m @ m)
        if var < 1e-12:
            return returns[keep]
        Rc = R - R.mean(axis=0)
        betas = (Rc.T @ m) / var
        resid = Rc - np.outer(m, betas)
        return pd.DataFrame(resid, index=returns.index, columns=keep)

    def _spectral_clean(self, C: np.ndarray, returns: pd.DataFrame) -> np.ndarray:
        """Eigenvalue (RMT) cleaning of the correlation matrix (truncated top-mode solver)."""
        method = self.config.spectral_clean
        n = C.shape[0]
        T = max(returns.shape[0], 1)
        k = max(1, int(self.config.spectral_k))
        n_top = int(min(max(k + 5, 40), n - 1))
        if n_top < 1:
            return C
        Cf = C.astype(float)
        try:
            vals, vecs = eigsh(Cf, k=n_top, which="LA")
        except Exception:
            vals, vecs = np.linalg.eigh(Cf)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        lam_plus = (1.0 + np.sqrt(n / T)) ** 2

        if method in ("market", "topk"):
            m = 1 if method == "market" else k
            Cc = Cf - (vecs[:, :m] * vals[:m]) @ vecs[:, :m].T
            tag = f"removed top {m} mode(s)"
        elif method == "rmt":
            band = [i for i in range(len(vals)) if i >= k and vals[i] > lam_plus]
            if band:
                V, L = vecs[:, band], vals[band]
                Cc = (V * L) @ V.T
                tag = (f"kept {len(band)} band modes (lambda_+={lam_plus:.2f}, "
                       f"dropped top {k}+bulk)")
            else:
                Cc = Cf - (vecs[:, :k] * vals[:k]) @ vecs[:, :k].T
                tag = f"no modes above lambda_+={lam_plus:.2f}; removed top {k}"
        else:
            return C

        d = np.sqrt(np.clip(np.diag(Cc), 1e-12, None))
        Cc = Cc / np.outer(d, d)
        np.fill_diagonal(Cc, 0.0)
        logger.info("Spectral clean [%s]: %s", method, tag)
        return np.nan_to_num(Cc)

    # -------------------------------------------------------------------------
    # Similarity matrix
    # -------------------------------------------------------------------------

    def _compute_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        if self.config.use_precision_matrix:
            n = returns.shape[1]
            try:
                X = returns.values
                stds = X.std(axis=0)
                stds[stds == 0] = 1e-8
                X_norm = (X - X.mean(axis=0)) / stds
                if self.config.lasso_alpha > 0:
                    logger.info("GraphicalLasso (alpha=%s) on %d nodes, T=%d...",
                                self.config.lasso_alpha, n, X.shape[0])
                    model = GraphicalLasso(alpha=self.config.lasso_alpha,
                                           max_iter=200).fit(X_norm)
                else:
                    logger.info("GraphicalLassoCV on %d nodes, T=%d (slow CV; "
                                "set lasso_alpha>0 to skip)...", n, X.shape[0])
                    model = GraphicalLassoCV(max_iter=200).fit(X_norm)
                # Normalize precision -> partial correlation in [-1, 1].
                P = model.precision_
                dinv = 1.0 / np.sqrt(np.clip(np.diag(P), 1e-12, None))
                pcorr = -P * np.outer(dinv, dinv)
                np.fill_diagonal(pcorr, 0.0)
                pcorr = np.nan_to_num(pcorr, nan=0.0, posinf=1.0, neginf=-1.0)
                return np.clip(pcorr, -1.0, 1.0)
            except Exception as e:
                logger.warning("Lasso failed (%s), using correlation", e)

        logger.info("Computing correlation matrix...")
        C = np.nan_to_num(returns.corr().values)
        if self.config.spectral_clean != "none":
            C = self._spectral_clean(C, returns)
        return C

    # -------------------------------------------------------------------------
    # Forced Connectivity
    # -------------------------------------------------------------------------

    def _rescue_islands(self, G, matrix, col_names, returns):
        """Ensure every node has at least `min_degree` edges (Forced Connectivity)."""
        min_degree = self.config.min_degree
        corr_matrix = None
        added_edges_count = 0

        node_to_idx = {name: i for i, name in enumerate(col_names)}
        all_nodes_indices = set(range(len(col_names)))

        logger.info("Ensuring minimum connectivity (degree >= %d)...", min_degree)

        for node in list(G.nodes()):
            idx = node_to_idx[node]
            while G.degree(node) < min_degree:
                current_neighbors = list(G.neighbors(node))
                neighbor_indices = [node_to_idx[n] for n in current_neighbors]
                neighbor_indices.append(idx)

                row = np.abs(matrix[idx]).copy()
                row[neighbor_indices] = 0
                best_idx = np.argmax(row)
                weight = float(row[best_idx])

                if weight <= 1e-9:
                    if corr_matrix is None:
                        corr_matrix = np.nan_to_num(returns.corr().values)
                    row_corr = np.abs(corr_matrix[idx]).copy()
                    row_corr[neighbor_indices] = 0
                    best_idx = np.argmax(row_corr)
                    weight = float(row_corr[best_idx])

                if weight <= 1e-9:
                    available = list(all_nodes_indices - set(neighbor_indices))
                    if not available:
                        break
                    best_idx = available[0]
                    weight = 0.0001

                target = col_names[best_idx]
                G.add_edge(node, target, weight=weight)
                added_edges_count += 1

        if added_edges_count > 0:
            logger.info("Added %d edges to ensure min degree >= %d.",
                        added_edges_count, min_degree)
