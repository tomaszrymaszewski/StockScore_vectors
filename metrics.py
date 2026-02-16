import networkx as nx
import numpy as np
import community.community_louvain as community_louvain
from collections import Counter
from sklearn.neighbors import NearestNeighbors

class NetworkMetrics:
    def __init__(self, graph: nx.Graph, embeddings: dict, sector_map: dict = None):
        self.graph = graph
        self.embeddings = embeddings
        self.sector_map = sector_map or {}

    @staticmethod
    def get_gini(array):
        """Calculate the Gini coefficient of a numpy array."""
        array = np.array(array).flatten()
        if np.amin(array) < 0:
            array -= np.amin(array)  # Values cannot be negative
        array += 0.0000001  # Values cannot be 0
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    def calculate_metrics(self) -> dict:
        """
        Calculates Reciprocity, Saturation, Cliffs, Hubness, and advanced graph metrics.
        Returns a dictionary of results.
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            return {}

        metrics = {}

        # --- 0. STRUCTURAL INTEGRITY ---
        # 1. PageRank Gini
        try:
            pr = list(nx.pagerank(self.graph, weight='weight').values())
            metrics['gini_score'] = self.get_gini(pr)
        except Exception:
            metrics['gini_score'] = 0.0

        # 2. Modularity
        try:
            partition = community_louvain.best_partition(self.graph, weight='weight')
            metrics['modularity'] = community_louvain.modularity(partition, self.graph, weight='weight')
        except Exception:
            metrics['modularity'] = 0.0

        # 3. Assortativity
        try:
            metrics['assortativity'] = nx.degree_assortativity_coefficient(self.graph, weight='weight')
        except Exception:
            metrics['assortativity'] = 0.0

        # --- 1. CLUSTER STABILITY (Reciprocity) ---
        # Note: Reciprocity is naturally 100% in an undirected graph (NetworkX default).
        # However, if the input graph was built from a directed KNN process (like in test_existing_vectors.py),
        # we might need to handle it differently.
        # But here self.graph is likely an undirected Graph from train_vectors.py.
        # If we strictly want to measure the reciprocity of the *KNN graph* implied by embeddings,
        # we would need to rebuild the KNN graph here.
        # For now, we will calculate reciprocity based on the provided graph structure if it's directed,
        # or skip if undirected (assumed 1.0).
        if self.graph.is_directed():
            metrics['reciprocity'] = nx.reciprocity(self.graph)
        else:
            # If undirected, check logical reciprocity if edges have metadata, or default to 1.0
            metrics['reciprocity'] = 1.0

        # --- 2. SCORE SATURATION ---
        # Collect all edge weights
        all_scores = [d['weight'] for u, v, d in self.graph.edges(data=True) if 'weight' in d]
        if all_scores:
            metrics['avg_score'] = np.mean(all_scores)
            perfect_matches = sum(1 for s in all_scores if s > 0.99)
            metrics['saturation'] = perfect_matches / len(all_scores)
        else:
            metrics['avg_score'] = 0.0
            metrics['saturation'] = 0.0

        # --- 3. DATA DENSITY (Cliff Detection) ---
        # This metric requires neighbor lists sorted by score.
        # We iterate over nodes and their edges.
        cliff_count = 0
        severe_cliffs = []
        for node in self.graph.nodes():
            # Get neighbors and weights
            neighbors = [(nbr, d['weight']) for nbr, d in self.graph[node].items() if 'weight' in d]
            # Sort by weight descending
            scores = sorted([s for n, s in neighbors], reverse=True)
            
            if len(scores) > 1:
                drops = np.diff(scores) * -1
                max_drop = np.max(drops)
                if max_drop > 0.15:
                    cliff_count += 1
                    if max_drop > 0.25:
                        severe_cliffs.append((node, max_drop))
        
        metrics['cliff_count'] = cliff_count
        metrics['cliff_pct'] = cliff_count / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0

        # --- 4. HUBNESS ---
        degrees = dict(self.graph.degree())
        if degrees:
            # Sort by degree
            top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            # Store top hub as "Ticker (Count)"
            metrics['top_hub'] = f"{top_hubs[0][0]} ({top_hubs[0][1]})" if top_hubs else "None"
            
            # Check if any hub dominates > 25% of nodes (only if graph is dense enough)
            # For sparse graphs (like MST), avg degree is small, so this metric might need interpretation.
            # In test_existing_vectors, it used count of *being a neighbor*.
            # In an undirected graph, degree = number of neighbors.
            metrics['hubness_critical'] = any(count > (self.graph.number_of_nodes() * 0.25) for _, count in top_hubs)
        else:
            metrics['top_hub'] = "None"
            metrics['hubness_critical'] = False

        # --- 5. SEMANTIC HIT RATE ---
        metrics['hit_rate'] = self.calculate_hit_rate(k=3)

        return metrics

    def calculate_hit_rate(self, k: int = 3) -> float:
        """
        Calculates the percentage of k-nearest neighbors that share the same Sub-Industry.
        K=3 (excludes self).
        """
        if not self.embeddings or not self.sector_map:
            return 0.0

        # Filter embeddings to those in sector map
        valid_tickers = [t for t in self.embeddings.keys() if t in self.sector_map]
        
        if len(valid_tickers) < k + 1:
            return 0.0

        # Build matrix
        matrix = np.array([self.embeddings[t] for t in valid_tickers])
        
        # KNN
        try:
            # k+1 because the first one is the node itself
            nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(matrix)
            distances, indices = nbrs.kneighbors(matrix)
        except Exception:
            return 0.0

        total_hits = 0
        total_neighbors = 0

        for i, ticker in enumerate(valid_tickers):
            target_sector = self.sector_map[ticker]
            
            # Start from 1 to skip self
            for j in range(1, k + 1):
                neighbor_idx = indices[i][j]
                neighbor_ticker = valid_tickers[neighbor_idx]
                neighbor_sector = self.sector_map.get(neighbor_ticker, "UNKNOWN")
                
                if neighbor_sector == target_sector:
                    total_hits += 1
                
                total_neighbors += 1

        return total_hits / total_neighbors if total_neighbors > 0 else 0.0

    def get_semantic_samples(self, probe_tickers: list[str], top_k: int = 5) -> str:
        """
        Returns a string representation of the nearest neighbors for specific probe tickers.
        Example output: "AAPL: [SWKS, QRVO]; JPM: [BAC, C]"
        """
        if not self.embeddings:
            return ""

        # Prepare matrix
        tickers = list(self.embeddings.keys())
        matrix = np.array([self.embeddings[t] for t in tickers])
        
        # Build NN
        try:
            nbrs = NearestNeighbors(n_neighbors=top_k + 1, metric='cosine').fit(matrix)
        except ValueError:
            return "Error: Not enough samples for KNN"

        results = []
        
        for probe in probe_tickers:
            if probe in tickers:
                idx = tickers.index(probe)
                vec = matrix[idx].reshape(1, -1)
                distances, indices = nbrs.kneighbors(vec)
                
                # Get neighbor names (skip self at index 0)
                neighbors = []
                for i in range(1, len(indices[0])):
                    nbr_idx = indices[0][i]
                    neighbors.append(tickers[nbr_idx])
                
                results.append(f"{probe}: {neighbors}")
            else:
                # Optionally note missing probes, or just skip
                pass
                
        return "; ".join(results)
