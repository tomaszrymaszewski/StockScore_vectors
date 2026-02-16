"""
Stock Vector Database Training Pipeline
========================================
Trains node2vec embeddings on stock correlation/precision matrices.
Reads historical data from local CSV only. Metadata fetched via Polygon.

Usage:
    python train_vectors.py --run              # Run the pipeline using Config settings
    python train_vectors.py --clear            # Clear the index
    python train_vectors.py --stats            # Show index stats
    python train_vectors.py --fetch AAPL       # Inspect AAPL vector
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import networkx as nx
import requests
from dotenv import load_dotenv
from node2vec import Node2Vec
from pinecone import Pinecone, ServerlessSpec
from sklearn.covariance import GraphicalLassoCV


load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """
    Central control panel.
    Edit the values in 'EXPERIMENT PARAMETERS' to change the training behavior.
    """

    # -------------------------------------------------------------------------
    # 1. EXPERIMENT PARAMETERS (Edit these!)
    # -------------------------------------------------------------------------
    
    # Random Walk Behavior (Node2Vec)
    # p = Return parameter (High p > max => avoid returning to start => explore further)
    # q = In-out parameter (High q > max => stay local (BFS), Low q < 1 => go far (DFS))
    walk_p: float = 0.8
    walk_q: float = 7.2

    # Graph Construction
    # 'threshold' = Connect stocks if correlation > X.
    # 'mst'       = Minimum Spanning Tree.
    graph_filtering: str = "threshold"

    # Only used if filtering is 'threshold'
    correlation_threshold: float = 0.65   # Lower = fewer weak connections are broken completely

    # Only used if filtering is 'mst'
    mst_extra_edges_pct: float = 0.05   # 0.0 = Standard MST. 0.05 = Add top 5% of strongest unused edges.
    mst_squared_corr: bool = True    # If True, use (correlation)^2 as weight. Emphasizes very strong relationships.
    # -----------------------------------------------------------

    # Mathematical Method
    # True = Graphical Lasso (Partial Correlation). Removes indirect market noise.
    # False = Standard Pearson Correlation. Captures raw movement similarity.
    use_precision_matrix: bool = False

    # Data Scope
    # Note: Will filter the provided CSV to this range ending at today.
    training_period: str = "2y"       # e.g., "5y", "2y", "6m", "90d"
    include_metadata: bool = False     # Fetch and sync company info? Required for evaluation
    skip_pinecone: bool = False        # If True, runs purely locally and returns embeddings directly


    # -------------------------------------------------------------------------
    # 2. INFRASTRUCTURE & CONSTANTS
    # -------------------------------------------------------------------------
    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")

    index_name: str = "r3000"
    dimensions: int = 64
    cloud_region: str = "us-east-1"

    # UPDATED: Points to the static file provided
    cache_file: str = "data/russell_3000_2y.csv"
    cutoff_date: str = "2025-11-30"

    # -------------------------------------------------------------------------
    # 3. INTERNAL STATE
    # -------------------------------------------------------------------------
    training_period_days: int = field(init=False)

    def __post_init__(self):
        """Parse period string into days."""
        self.training_period_days = self._parse_period(self.training_period)

    @staticmethod
    def _parse_period(period_str: str) -> int:
        """Parse '1y', '6m' into days."""
        period_str = period_str.lower().strip()
        if period_str.endswith("y"):
            return int(float(period_str[:-1]) * 365)
        elif period_str.endswith("m"):
            return int(float(period_str[:-1]) * 30)
        elif period_str.endswith("d"):
            return int(period_str[:-1])
        return 365 # Default fallback


# =============================================================================
# PINECONE CLIENT
# =============================================================================

class PineconeClient:
    """Handles all Pinecone operations."""

    def __init__(self, config: Config):
        self.config = config
        if not config.skip_pinecone:
            self.pc = Pinecone(api_key=config.pinecone_api_key)
            self.index = self._get_or_create_index()
        else:
            self.pc = None
            self.index = None

    def _get_or_create_index(self):
        existing_indexes = self.pc.list_indexes().names()
        
        if self.config.index_name in existing_indexes:
            # Check dimensions
            index_meta = self.pc.describe_index(self.config.index_name)
            if int(index_meta.dimension) != int(self.config.dimensions):
                print(f"⚠️  Dimension mismatch (Index: {index_meta.dimension}, Config: {self.config.dimensions})")
                print(f"♻️  Deleting and recreating index: {self.config.index_name}...")
                self.pc.delete_index(self.config.index_name)
                time.sleep(10) # Wait for deletion to propagate
                existing_indexes = [] # Force creation branch
            else:
                return self.pc.Index(self.config.index_name)

        if self.config.index_name not in existing_indexes:
            print(f"➡️  Creating new index: {self.config.index_name}")
            self.pc.create_index(
                name=self.config.index_name,
                dimension=self.config.dimensions,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.config.cloud_region),
            )
            time.sleep(5) # Wait for creation to initialize
        
        return self.pc.Index(self.config.index_name)

    def upsert_vectors(self, vectors: list[dict], batch_size: int = 100):
        total_batches = (len(vectors) + batch_size - 1) // batch_size
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch)
            print(f"   📤 Upserted batch {i // batch_size + 1}/{total_batches}")

    def delete_ticker(self, ticker: str):
        try:
            self.index.delete(ids=[ticker])
            print(f"🗑️  Deleted {ticker}")
        except Exception as e:
            print(f"❌ Failed to delete {ticker}: {e}")

    def clear(self):
        try:
            self.index.delete(delete_all=True)
            print("⚠️  Index cleared")
        except Exception as e:
            print(f"❌ Failed to clear index: {e}")

    def stats(self) -> dict:
        stats = self.index.describe_index_stats()
        count = stats.get("total_vector_count", 0)
        print(f"📊 Database Status: {count} vectors stored")
        return stats

    def fetch_vector(self, ticker: str):
        res = self.index.fetch(ids=[ticker])
        if ticker in res["vectors"]:
            values = res["vectors"][ticker]["values"][:5]
            meta = res["vectors"][ticker].get("metadata", {})
            print(f"🧬 {ticker}")
            print(f"   Values (first 5): {values}")
            print(f"   Metadata: {meta}")
        else:
            print(f"❌ No vector found for {ticker}")


# =============================================================================
# DATA FETCHER (LOCAL ONLY)
# =============================================================================

class DataFetcher:
    """Handles loading stock price data from local CSV only."""

    def __init__(self, config: Config):
        self.config = config

    def fetch_returns(self, tickers: list[str] = None) -> pd.DataFrame:
        """Load local data and compute returns based on Config period."""
        period_days = self.config.training_period_days

        # Training window (ending today, going back X days)
        # Note: We assume the CSV contains data up to 'now' or close to it.
        if self.config.cutoff_date:
            training_end = datetime.strptime(self.config.cutoff_date, "%Y-%m-%d")
        else:
            training_end = datetime.now()

        training_start = training_end - timedelta(days=period_days)

        print(f"📅 Training period: {training_start.date()} to {training_end.date()} ({period_days} days)")

        # Load cached data
        if not os.path.exists(self.config.cache_file):
            print(f"❌ Error: File {self.config.cache_file} not found.")
            sys.exit(1)

        print(f"📂 Loading local file: {self.config.cache_file}")
        df = self._load_csv_data()

        # Filter by date range
        df = df.loc[training_start:training_end]

        # Calculate returns
        returns = self._prepare_returns(df, tickers)
        print(f"   ℹ️  Final dataset: {returns.shape[0]} days × {returns.shape[1]} tickers")
        return returns

    def _load_csv_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.config.cache_file, index_col=0)
            df.index = pd.to_datetime(df.index).normalize()
            # Handle potential duplicates in index or columns
            df = df.groupby(df.index).last().sort_index()
            return df.apply(pd.to_numeric, errors="coerce")
        except Exception as e:
            print(f"❌ Error parsing CSV: {e}")
            return pd.DataFrame()

    def _prepare_returns(self, df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        if df.empty: return pd.DataFrame()

        # Determine valid columns (intersection of requested tickers and what's in CSV)
        if tickers:
            available = [t for t in tickers if t in df.columns]
        else:
            available = df.columns.tolist()

        # If the CSV has tickers NOT in the input list, but valid, we might want them.
        # However, strictly speaking, we usually limit to the universe `tickers`.
        # To be safe and use everything in the CSV:
        # available = df.columns.tolist()

        subset = df[available].copy()

        # Drop non-trading days (rows where all columns are NaN)
        subset = subset.dropna(how='all')

        # Drop tickers with too much missing data in this specific window
        subset = subset.dropna(axis=1, thresh=int(len(subset) * 0.8))
        if subset.empty: return pd.DataFrame()

        # Forward fill holes, then drop remaining rows with NaNs
        subset = subset.ffill().dropna(axis=0)

        if len(subset) < 30:
            print(f"   ⚠️  Insufficient data in this time window: only {len(subset)} rows")
            return pd.DataFrame()

        # Calculate Returns
        returns = subset.pct_change().dropna()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        # Drop stocks that didn't move (flat lines)
        returns = returns.loc[:, returns.std() > 1e-8]
        return returns


# =============================================================================
# METADATA FETCHER
# =============================================================================

class MetadataFetcher:
    def __init__(self, config: Config):
        self.api_key = config.polygon_api_key

    def fetch(self, ticker: str) -> dict:
        try:
            url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={self.api_key}"
            res = requests.get(url).json()
            if res.get("status") == "OK" and "results" in res:
                r = res["results"]
                return {
                    "name": r.get("name", ""),
                    "description": self._truncate(r.get("description", ""), 500),
                    "sector": r.get("sic_description", ""),
                    "market_cap": r.get("market_cap", 0),
                    "employees": r.get("total_employees", 0),
                }
        except Exception:
            pass
        return {"name": ticker}

    def fetch_batch(self, tickers: list[str]) -> dict[str, dict]:
        print(f"📋 Fetching metadata for {len(tickers)} tickers...")
        metadata = {}
        for i, ticker in enumerate(tickers):
            metadata[ticker] = self.fetch(ticker)
            if (i + 1) % 20 == 0: print(f"   ... {i + 1}/{len(tickers)}")
        return metadata

    @staticmethod
    def _truncate(text: str, max_length: int) -> str:
        if len(text) <= max_length: return text
        return text[:max_length - 3] + "..."


# =============================================================================
# GRAPH BUILDER
# =============================================================================

class GraphBuilder:
    def __init__(self, config: Config):
        self.config = config

    def build(self, returns: pd.DataFrame) -> nx.Graph:
        """Builds graph based on Config filtering strategy (MST vs Threshold)."""
        G = nx.Graph()
        col_names = returns.columns
        G.add_nodes_from(col_names)

        # 1. Compute Matrix
        matrix = self._compute_matrix(returns)

        # 2. Filter Edges
        if self.config.graph_filtering == "mst":
            print("🌲 Building Maximum Spanning Tree (MST)...")

            # --- MST/PMFG SPECIFIC LOGIC START ---
            # Prepare all potential edges
            all_edges = []
            for i in range(len(col_names)):
                for j in range(i + 1, len(col_names)):
                    w = np.abs(matrix[i, j])

                    # [MST OPTION] Squared Correlation
                    if self.config.mst_squared_corr:
                        w = w ** 2

                    if np.isfinite(w) and w > 1e-6:
                        all_edges.append((col_names[i], col_names[j], w))

            # Create full weighted graph to compute MST
            temp_G = nx.Graph()
            temp_G.add_weighted_edges_from(all_edges)

            # Compute the backbone (Maximum Spanning Tree)
            G = nx.maximum_spanning_tree(temp_G)

            # [MST OPTION] PMFG Simulation (Extra Edges)
            if self.config.mst_extra_edges_pct > 0:
                mst_edges = set(tuple(sorted((u, v))) for u, v in G.edges())
                sorted_potential_edges = sorted(all_edges, key=lambda x: x[2], reverse=True)
                remaining_edges = [
                    e for e in sorted_potential_edges
                    if tuple(sorted((e[0], e[1]))) not in mst_edges
                ]
                num_to_add = int(len(remaining_edges) * self.config.mst_extra_edges_pct)
                print(f"   ➕ Adding top {self.config.mst_extra_edges_pct:.1%} of remaining edges ({num_to_add} edges)...")

                for i in range(num_to_add):
                    u, v, w = remaining_edges[i]
                    G.add_edge(u, v, weight=w)
            # --- MST/PMFG SPECIFIC LOGIC END ---

            print(f"🕸️  Graph (MST + Extras): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        elif self.config.graph_filtering == "threshold":
            # Thresholding
            thresh = self.config.correlation_threshold
            edge_count = 0
            for i in range(len(col_names)):
                for j in range(i + 1, len(col_names)):
                    w = np.abs(matrix[i, j])
                    if w > thresh and np.isfinite(w):
                        G.add_edge(col_names[i], col_names[j], weight=float(w))
                        edge_count += 1

            # Rescue islands
            self._rescue_islands(G, matrix, col_names, returns)
            print(f"🕸️  Graph (Threshold > {thresh}): {G.number_of_nodes()} nodes, {edge_count} edges")
        else:
            raise ValueError(f"Unknown graph filtering method: {self.config.graph_filtering}")

        return G

    def _compute_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        if self.config.use_precision_matrix:
            print("🧠 Computing precision matrix (GraphicalLasso)...")
            try:
                X = returns.values
                stds = X.std(axis=0)
                stds[stds == 0] = 1e-8
                X_norm = (X - X.mean(axis=0)) / stds
                model = GraphicalLassoCV(max_iter=1000).fit(X_norm)
                return np.nan_to_num(model.precision_)
            except Exception as e:
                print(f"⚠️  Lasso failed ({e}), using correlation")

        print("🔗 Computing correlation matrix...")
        return np.nan_to_num(returns.corr().values)

    def _rescue_islands(self, G, matrix, col_names, returns):
        """
        Ensures every node has at least 3 edges by connecting to strongest available neighbors.
        """
        corr_matrix = None  # Lazy computation
        added_edges_count = 0

        # Pre-compute index mapping for speed
        node_to_idx = {name: i for i, name in enumerate(col_names)}
        all_nodes_indices = set(range(len(col_names)))

        print("   🔧 Ensuring minimum connectivity (degree >= 3)...")

        for node in list(G.nodes()):
            idx = node_to_idx[node]

            # We loop until degree is at least 3
            while G.degree(node) < 3:
                # Identify who we are already connected to
                current_neighbors = list(G.neighbors(node))
                # Get their indices
                neighbor_indices = [node_to_idx[n] for n in current_neighbors]
                # Also exclude self
                neighbor_indices.append(idx)

                # 1. Try primary matrix
                row = np.abs(matrix[idx]).copy()
                # Zero out existing connections + self
                row[neighbor_indices] = 0
                
                best_idx = np.argmax(row)
                weight = float(row[best_idx])
                
                # 2. Fallback only if primary matrix yields 0 weight for best remaining candidate
                if weight <= 1e-9:
                    if corr_matrix is None:
                        corr_matrix = np.nan_to_num(returns.corr().values)
                    
                    row_corr = np.abs(corr_matrix[idx]).copy()
                    row_corr[neighbor_indices] = 0
                    
                    best_idx = np.argmax(row_corr)
                    weight = float(row_corr[best_idx])
                
                # 3. Last Resort: Force connection if data is broken
                if weight <= 1e-9:
                    # Pick a node not in neighbor_indices
                    available = list(all_nodes_indices - set(neighbor_indices))
                    if not available:
                        break # Cannot add more edges (already fully connected)
                    best_idx = available[0] # Just pick first valid one
                    weight = 0.0001
                
                target = col_names[best_idx]
                G.add_edge(node, target, weight=weight)
                added_edges_count += 1
                
        if added_edges_count > 0:
            print(f"   ⛑️  Added {added_edges_count} edges to ensure min degree >= 3.")


# =============================================================================
# EMBEDDING TRAINER
# =============================================================================
class EmbeddingTrainer:
    def __init__(self, config: Config):
        self.config = config

    def train(self, G: nx.Graph) -> dict[str, list[float]]:
        p, q = self.config.walk_p, self.config.walk_q
        print(f"🏋️  Training Node2Vec (p={p}, q={q}, dim={self.config.dimensions})...")

        isolates = list(nx.isolates(G))
        if isolates:
            G.remove_nodes_from(isolates)
            print(f"   ⚠️ Removed {len(isolates)} isolated nodes.")

        if not G.nodes(): return {}

        n2v = Node2Vec(
            G, dimensions=self.config.dimensions, walk_length=20, num_walks=100,
            p=p, q=q, workers=1, quiet=True
        )
        model = n2v.fit(window=10, min_count=1)

        return {t: model.wv[t].tolist() for t in model.wv.index_to_key}


# =============================================================================
# EVALUATOR & REPORTING
# =============================================================================

class Evaluator:
    def __init__(self, config: Config, embeddings: dict, returns: pd.DataFrame, graph: nx.Graph):
        self.config = config
        self.embeddings = embeddings
        self.returns = returns
        self.graph = graph
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = "reports"

        # Representative stocks to sample in the report
        self.sample_tickers = ["AAPL", "JPM", "JNJ", "XOM", "TSLA", "AMD", "LOW", "KO", "PEP", "GLD", "AMZN"]

    def generate_report(self):
        """Runs metrics and saves a text report to disk."""
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

        # 1. Alignment Score (Matrix Reconstruction)
        alignment_msg = self._calculate_alignment_score()

        # 2. Graph Statistics
        graph_stats = self._calculate_graph_stats()

        # 3. Qualitative Check (Neighbors)
        neighbor_samples = self._get_neighbor_samples()

        # 4. Compile Report
        report_content = self._format_report(alignment_msg, graph_stats, neighbor_samples)

        # 5. Save
        filename = f"{self.report_dir}/run_{self.timestamp}.txt"
        with open(filename, "w") as f:
            f.write(report_content)

        print(f"\n📝 Report generated: {filename}")
        
        return graph_stats, neighbor_samples

    def _calculate_graph_stats(self) -> dict:
        G = self.graph
        if not G or G.number_of_nodes() == 0:
            return {}

        # Basic Counts
        nodes = G.number_of_nodes()
        edges = G.number_of_edges()

        # Density: Ratio of edges present to total possible edges
        density = nx.density(G)

        # Average Degree: Avg number of connections per node
        avg_degree = sum(dict(G.degree()).values()) / nodes

        # Connected Components: Is the graph fractured?
        components = nx.number_connected_components(G)

        # Clustering Coefficient: "Cliquiness". (Triangle density)
        # Global clustering coeff (Transitivity) is usually robust for financial markets
        transitivity = nx.transitivity(G)

        # Average local clustering (can be slow on massive graphs, but okay for ~500 nodes)
        avg_clustering = nx.average_clustering(G)

        return {
            "nodes": nodes,
            "edges": edges,
            "density": density,
            "avg_degree": avg_degree,
            "components": components,
            "transitivity": transitivity,
            "avg_clustering": avg_clustering
        }

    def _calculate_alignment_score(self):
        """Calculates Pearson Correlation between vector similarity and price correlation."""
        # Find common tickers
        common_tickers = [t for t in self.embeddings.keys() if t in self.returns.columns]
        if len(common_tickers) < 5:
            return "Insufficient data for alignment check."

        # 1. Ground Truth Matrix (Price Correlation)
        df_subset = self.returns[common_tickers]
        truth_matrix = df_subset.corr().values

        # 2. Embedding Similarity Matrix
        vec_matrix = np.array([self.embeddings[t] for t in common_tickers])
        norms = np.linalg.norm(vec_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        vec_matrix_norm = vec_matrix / norms
        sim_matrix = np.dot(vec_matrix_norm, vec_matrix_norm.T)

        # 3. Flatten upper triangles
        iu = np.triu_indices(len(common_tickers), k=1)
        truth_values = truth_matrix[iu]

        # We can perform a correlation check here if desired,
        # but for now we just return the count of stocks evaluated.
        return f"Evaluated on {len(common_tickers)} stocks ({len(truth_values)} pairs)."

    def _get_neighbor_samples(self) -> dict:
        """Finds nearest neighbors for the sample list."""
        tickers = list(self.embeddings.keys())
        if not tickers: return {}

        matrix = np.array([self.embeddings[t] for t in tickers])

        # Build local NN model for quick lookup. We ask for 16 to get top 15 neighbors.
        nbrs = NearestNeighbors(n_neighbors=16, metric='cosine').fit(matrix)

        samples = {}
        for target in self.sample_tickers:
            if target in tickers:
                target_idx = tickers.index(target)
                vec = matrix[target_idx].reshape(1, -1)
                distances, indices = nbrs.kneighbors(vec)

                # Get neighbor names and scores (skip index 0, which is the target itself)
                neighbors_with_scores = []
                for i in range(1, len(indices[0])):
                    neighbor_index = indices[0][i]
                    neighbor_ticker = tickers[neighbor_index]
                    similarity = 1 - distances[0][i]
                    neighbors_with_scores.append((neighbor_ticker, similarity))

                samples[target] = neighbors_with_scores

        return samples

    def _format_report(self, alignment_msg, graph_stats, samples):
        return f"""
STOCK VECTOR TRAINING REPORT
============================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Note: {alignment_msg}

1. CONFIGURATION
----------------
Methods:
  - Filtering:   {self.config.graph_filtering.upper()}
  - Precision:   {self.config.use_precision_matrix} (Lasso)
  - Period:      {self.config.training_period}

Hyperparameters:
  - Walk P:      {self.config.walk_p} (Return)
  - Walk Q:      {self.config.walk_q} (In-Out)
  - Threshold:   {self.config.correlation_threshold}
  - Dimensions:  {self.config.dimensions}

MST / PMFG Options:
  - Extra Edges: {self.config.mst_extra_edges_pct:.1%}
  - Squared Corr:{self.config.mst_squared_corr}

2. GRAPH STATISTICS
-------------------
  - Nodes:             {graph_stats.get('nodes', 0)}
  - Edges:             {graph_stats.get('edges', 0)}
  - Density:           {graph_stats.get('density', 0):.4f} (Edge Probability)
  - Avg Degree:        {graph_stats.get('avg_degree', 0):.2f} neighbors
  - Components:        {graph_stats.get('components', 0)} (Islands)
  - Avg Clustering:    {graph_stats.get('avg_clustering', 0):.4f} (Local Cliquiness)
  - Transitivity:      {graph_stats.get('transitivity', 0):.4f} (Global Triangles)

3. SIMILARITY SAMPLES (Top 15)
------------------------------
(Checking if neighbors make semantic sense)

{self._format_samples(samples)}

4. INFRASTRUCTURE
-----------------
  - Cache File:  {self.config.cache_file}
  - Index Name:  {self.config.index_name}
  - Vectors:     {len(self.embeddings)}

"""

    def _format_samples(self, samples):
        txt = ""
        for ticker, neighbors in samples.items():
            neighbor_list = [f"{neighbor} ({score:.2f})" for neighbor, score in neighbors]
            txt += f"{ticker.ljust(6)} -> {', '.join(neighbor_list)}\n"
        return txt if txt else "No sample tickers found in trained set."


# =============================================================================
# PIPELINE
# =============================================================================

class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.pinecone = PineconeClient(config)
        self.fetcher = DataFetcher(config)
        self.graph_builder = GraphBuilder(config)
        self.trainer = EmbeddingTrainer(config)
        self.trainer = EmbeddingTrainer(config)
        self.meta_fetcher = MetadataFetcher(config) if config.include_metadata else None

    def run_experiment(self):
        """
        Streamlined run for experiments.py.
        Returns (graph, embeddings) directly without upserting or reporting.
        """
        print(f"   🧪 Running Experiment: Mode={self.config.graph_filtering.upper()}")
        
        returns = self.fetcher.fetch_returns(tickers=None)
        graph = self.graph_builder.build(returns)
        embeddings = self.trainer.train(graph)
        
        return graph, embeddings, returns


    def run(self):
        print("\n" + "=" * 60)
        print(f"🔄 STOCK VECTOR PIPELINE START")
        print(f"   Mode: {self.config.graph_filtering.upper()} | Precision: {self.config.use_precision_matrix}")
        print("=" * 60)

        # 1. Fetch (Local Only)
        returns = self.fetcher.fetch_returns(tickers=None)

        # 2. Build Graph
        graph = self.graph_builder.build(returns)

        # 3. Train
        embeddings = self.trainer.train(graph)

        # 5. Metadata (Sync only, not used for eval)
        metadata = {}
        if self.meta_fetcher:
            metadata = self.meta_fetcher.fetch_batch(list(embeddings.keys()))

        # 6. Sync
        print(f"🚀 Upserting {len(embeddings)} vectors...")
        vectors = []
        for ticker, values in embeddings.items():
            vec = {"id": ticker, "values": values}
            if ticker in metadata:
                vec["metadata"] = metadata[ticker]
            vectors.append(vec)

        self.pinecone.upsert_vectors(vectors)
        print("\n✅ Pipeline complete!")
        self.pinecone.stats()

        # Evaluation
        print("📊 Running Evaluation...")
        evaluator = Evaluator(self.config, embeddings, returns, graph)
        return evaluator.generate_report()


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stock Vector Pipeline")

    # Utility Commands
    parser.add_argument("--run", action="store_true", help="Run pipeline with Config settings")
    parser.add_argument("--clear", action="store_true", help="Clear Pinecone index")
    parser.add_argument("--stats", action="store_true", help="Show index stats")
    parser.add_argument("--delete", type=str, metavar="TICKER", help="Delete specific ticker")
    parser.add_argument("--fetch", type=str, metavar="TICKER", help="Inspect vector")

    args = parser.parse_args()
    config = Config()  # Loads settings from the class definition

    if args.clear:
        PineconeClient(config).clear()
    elif args.stats:
        PineconeClient(config).stats()
    elif args.delete:
        PineconeClient(config).delete_ticker(args.delete)
    elif args.fetch:
        PineconeClient(config).fetch_vector(args.fetch)
    elif args.run:
        Pipeline(config).run()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()