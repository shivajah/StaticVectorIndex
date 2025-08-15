import time
import numpy as np
import faiss
import bisect
from collections import defaultdict
from .kmeans_builder import KMeansBuilder

class MultiLevelIndex:
    def __init__(self, n_lowest_clusters):
        self.n_lowest_clusters = n_lowest_clusters
        self.kmeans_builder = KMeansBuilder(n_lowest_clusters)
        self.data = None
        self.vector_metadata = None
        self.built = False
        
    def build_index(self, data):
        """Build the multi-level index from data."""
        print("Building multi-level vector index...")
        self.data = data.copy()
        
        # Build the hierarchical clustering
        self.kmeans_builder.create_multi_level_clusters(data)
        self.built = True
        print("Multi-level index built successfully!")
        
    def build_metadata(self, data, N_CROSS=1):
        """Build cross-pollination metadata for the given N_CROSS value.
        
        This creates the vector_metadata structure that maps:
        vector_metadata[outer_id][inner_id] = list of (dist, cos_sim, idx) tuples
        """
        if not self.built:
            raise ValueError("Index must be built before building metadata")
            
        print(f"Building cross-pollination metadata with N_CROSS = {N_CROSS}")
        
        # Get the lowest level (level 0) and highest level assignments
        lowest_level_kmeans = self.kmeans_builder.get_level_kmeans(0)
        highest_level = self.kmeans_builder.num_levels() - 1
        
        if highest_level == 0:
            # Only one level, create a dummy upper level
            print("Only one level found, creating single-level metadata")
            vector_metadata = defaultdict(dict)
            _, data_assignments = lowest_level_kmeans.index.search(data, N_CROSS)
            
            for idx, inner_ids in enumerate(data_assignments):
                for inner_id in inner_ids[:N_CROSS]:
                    outer_id = 0  # Single outer cluster
                    centroid = lowest_level_kmeans.centroids[inner_id]
                    vec = data[idx]
                    dist = np.linalg.norm(vec - centroid)
                    cos_sim = np.dot(vec, centroid) / (np.linalg.norm(vec) * np.linalg.norm(centroid) + 1e-8)
                    vector_metadata[outer_id].setdefault(inner_id, []).append((dist, cos_sim, idx))
        else:
            # Multi-level case
            highest_level_kmeans = self.kmeans_builder.get_level_kmeans(highest_level)
            
            # Get mapping from lowest level clusters to highest level clusters
            lowest_centroids = lowest_level_kmeans.centroids
            _, inner_to_outer = highest_level_kmeans.index.search(lowest_centroids, 1)
            
            # Assign data points to multiple clusters at the lowest level
            _, data_assignments = lowest_level_kmeans.index.search(data, N_CROSS)
            
            # Build the cross-pollination metadata
            vector_metadata = defaultdict(dict)
            for idx, inner_ids in enumerate(data_assignments):
                for inner_id in inner_ids[:N_CROSS]:
                    outer_id = inner_to_outer[inner_id][0]
                    centroid = lowest_level_kmeans.centroids[inner_id]
                    vec = data[idx]
                    dist = np.linalg.norm(vec - centroid)
                    cos_sim = np.dot(vec, centroid) / (np.linalg.norm(vec) * np.linalg.norm(centroid) + 1e-8)
                    vector_metadata[outer_id].setdefault(inner_id, []).append((dist, cos_sim, idx))
        
        # Sort each list by Euclidean distance to centroid
        for outer_id in vector_metadata:
            for inner_id in vector_metadata[outer_id]:
                vector_metadata[outer_id][inner_id].sort()
                
        self.vector_metadata = vector_metadata
        return vector_metadata
    
    def search_query_cross_pollination(self, query_vector, k, N_PROBE=1, probe_strategy="nprobe", tshirt_size="small"):
        """Search for k nearest neighbors using cross-pollination algorithm."""
        if not self.built or self.vector_metadata is None:
            raise ValueError("Index and metadata must be built before searching")
            
        # Get the clustering models
        lowest_level_kmeans = self.kmeans_builder.get_level_kmeans(0)
        highest_level = self.kmeans_builder.num_levels() - 1
        
        if highest_level == 0:
            # Single level case
            outer_ids = [0]
        else:
            highest_level_kmeans = self.kmeans_builder.get_level_kmeans(highest_level)
            # Find top outer clusters for the query
            _, outer_assignments = highest_level_kmeans.index.search(query_vector.reshape(1, -1), 3)
            outer_ids = outer_assignments[0]
        
        d = query_vector.shape[0]
        
        if probe_strategy == "nprobe":
            return self._search_nprobe(query_vector, outer_ids, lowest_level_kmeans, k, d, N_PROBE)
        else:
            return self._search_tshirt(query_vector, outer_ids, lowest_level_kmeans, k, d, tshirt_size)
    
    def _search_nprobe(self, x, outer_ids, inner_kmeans, k, d, N_PROBE):
        """Search using nprobe strategy."""
        best_heap = []
        tau = float("inf")
        inner_probed = 0
        seen_indices = set()
        outer_idx = 0
        total_outer = len(outer_ids)
        
        while inner_probed < N_PROBE and outer_idx < total_outer:
            outer_id = outer_ids[outer_idx]
            outer_idx += 1
            
            if outer_id not in self.vector_metadata:
                continue
                
            inner_ids = list(self.vector_metadata[outer_id].keys())
            if not inner_ids:
                continue
                
            inner_centroids_subset = inner_kmeans.centroids[inner_ids]
            index_l2 = faiss.IndexFlatL2(d)
            index_l2.add(inner_centroids_subset)
            _, inner_ranks_local = index_l2.search(x.reshape(1, -1), N_PROBE)
            selected_inner_ids = [inner_ids[j] for j in inner_ranks_local[0] if j < len(inner_ids)]
            
            for inner_id in selected_inner_ids:
                idxs_meta = self.vector_metadata[outer_id][inner_id]
                if not idxs_meta:
                    continue
                    
                centroid = inner_kmeans.centroids[inner_id]
                d_qc = np.linalg.norm(x - centroid)
                
                for dist_ic, cos_theta, idx2 in idxs_meta:
                    if idx2 in seen_indices:
                        continue
                        
                    # Lower bound pruning
                    lower_bound = abs(d_qc - dist_ic)
                    if lower_bound > tau:
                        continue
                        
                    # Estimate distance using law of cosines
                    est_dist = np.sqrt(max(0.0, d_qc ** 2 + dist_ic ** 2 - 2 * d_qc * dist_ic * cos_theta))
                    if est_dist > tau:
                        continue
                        
                    # Compute actual distance
                    actual_dist = np.linalg.norm(x - self.data[idx2])
                    best_heap.append((actual_dist, idx2))
                    seen_indices.add(idx2)
                    
                    if len(best_heap) > k:
                        best_heap.sort()
                        best_heap = best_heap[:k]
                        tau = best_heap[-1][0]
                        
                inner_probed += 1
                if inner_probed >= N_PROBE:
                    break
                    
        return best_heap
    
    def _search_tshirt(self, x, outer_ids, inner_kmeans, k, d, tshirt_size):
        """Search using t-shirt sizing strategy."""
        tshirt_settings = {
            "small": 0.10,
            "medium": 0.20,
            "large": 0.30
        }
        
        best_heap = []
        tau = float("inf")
        probed_inner_ids = set()
        seen_indices = set()
        pct = tshirt_settings[tshirt_size]
        n_outer_probe = max(1, int(np.ceil(len(outer_ids) * pct)))
        outer_ids = outer_ids[:n_outer_probe]
        
        for outer_id in outer_ids:
            if outer_id not in self.vector_metadata:
                continue
                
            inner_ids = list(self.vector_metadata[outer_id].keys())
            if not inner_ids:
                continue
                
            n_inner_probe = max(1, int(np.ceil(len(inner_ids) * pct)))
            inner_ids_to_probe = [iid for iid in inner_ids if iid not in probed_inner_ids][:n_inner_probe]
            
            if not inner_ids_to_probe:
                continue
                
            inner_centroids_subset = inner_kmeans.centroids[inner_ids_to_probe]
            index_l2 = faiss.IndexFlatL2(d)
            index_l2.add(inner_centroids_subset)
            _, inner_ranks_local = index_l2.search(x.reshape(1, -1), len(inner_ids_to_probe))
            selected_inner_ids = [inner_ids_to_probe[j] for j in inner_ranks_local[0] if j < len(inner_ids_to_probe)]
            
            for inner_id in selected_inner_ids:
                probed_inner_ids.add(inner_id)
                idxs_meta = self.vector_metadata[outer_id][inner_id]
                if not idxs_meta:
                    continue
                    
                centroid = inner_kmeans.centroids[inner_id]
                d_qc = np.linalg.norm(x - centroid)
                
                for dist_ic, cos_theta, idx2 in idxs_meta:
                    if idx2 in seen_indices:
                        continue
                        
                    lower_bound = abs(d_qc - dist_ic)
                    if lower_bound > tau:
                        continue
                        
                    est_dist = np.sqrt(max(0.0, d_qc ** 2 + dist_ic ** 2 - 2 * d_qc * dist_ic * cos_theta))
                    if est_dist > tau:
                        continue
                        
                    actual_dist = np.linalg.norm(x - self.data[idx2])
                    best_heap.append((actual_dist, idx2))
                    seen_indices.add(idx2)
                    
                    if len(best_heap) > k:
                        best_heap.sort()
                        best_heap = best_heap[:k]
                        tau = best_heap[-1][0]
                        
        return best_heap
    
    def search(self, query_vector, k, N_PROBE=1, probe_strategy="nprobe", tshirt_size="small"):
        """Main search interface."""
        return self.search_query_cross_pollination(query_vector, k, N_PROBE, probe_strategy, tshirt_size)
    
    def get_level_info(self):
        """Get information about each level in the hierarchy."""
        if not self.built:
            return "Index not built"
            
        info = []
        for i in range(self.kmeans_builder.num_levels()):
            kmeans = self.kmeans_builder.get_level_kmeans(i)
            info.append(f"Level {i}: {kmeans.centroids.shape[0]} clusters")
        return "\n".join(info)
    
    def __repr__(self):
        if not self.built:
            return f"MultiLevelIndex(n_lowest_clusters={self.n_lowest_clusters}, built=False)"
        return f"MultiLevelIndex(n_lowest_clusters={self.n_lowest_clusters}, levels={self.kmeans_builder.num_levels()}, built=True)"