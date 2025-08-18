import time
import numpy as np
import faiss
import bisect
from collections import defaultdict
from .kmeans_builder import KMeansBuilder
from .hierarchy_metadata import HierarchyMetadata

class MultiLevelIndex:
    def __init__(self, n_lowest_clusters):
        # Number of clusters at the lowest level of the hierarchy
        self.n_lowest_clusters = n_lowest_clusters
        # KMeansBuilder instance for hierarchical clustering
        self.kmeans_builder = KMeansBuilder(n_lowest_clusters)
        # HierarchyMetadata instance for managing metadata
        self.hierarchy_metadata = HierarchyMetadata(self.kmeans_builder)
        # Data matrix (set in build_index)
        self.data = None
        # Flag indicating if the index has been built
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
        """Build hierarchical metadata for the given N_CROSS value.
        
        This creates hierarchical metadata structure that supports top-down search
        through all levels of the hierarchy.
        """
        if not self.built:
            raise ValueError("Index must be built before building metadata")
        
        # Store reference to data for distance calculations
        self.hierarchy_metadata.get_data_reference(data)
        
        # Use the HierarchyMetadata class to build hierarchical metadata
        cluster_metadata = self.hierarchy_metadata.build_hierarchical_metadata(data, N_CROSS)
        return cluster_metadata
    
    def search_query_hierarchical(self, query_vector, k, n_probe_per_level=2):
        """Search for k nearest neighbors using true hierarchical search.
        
        This method performs top-down hierarchical search starting from the highest
        level and traversing down to the lowest level by following parent-child
        relationships.
        """
        if not self.built or not self.hierarchy_metadata.built:
            raise ValueError("Index and metadata must be built before searching")
            
        return self.hierarchy_metadata.hierarchical_search(query_vector, k, n_probe_per_level)
    
    def search_query_cross_pollination_legacy(self, query_vector, k, N_PROBE=1, probe_strategy="nprobe", tshirt_size="small"):
        """Legacy cross-pollination search (for compatibility).
        
        This method maintains backward compatibility with the old outer/inner approach
        but now uses the hierarchical metadata internally.
        """
        if not self.built or not self.hierarchy_metadata.built:
            raise ValueError("Index and metadata must be built before searching")
        
        # For backward compatibility, convert hierarchical search to cross-pollination format
        # by searching at multiple levels and combining results
        num_levels = self.kmeans_builder.num_levels()
        
        if num_levels == 1:
            # Single level case - use direct search
            return self._search_single_level(query_vector, k)
        
        # Multi-level case - use hierarchical search with varying probe counts
        if probe_strategy == "nprobe":
            n_probe_per_level = max(1, N_PROBE // num_levels)
        else:
            # T-shirt sizing
            size_map = {"small": 1, "medium": 2, "large": 3}
            n_probe_per_level = size_map.get(tshirt_size, 1)
        
        return self.hierarchy_metadata.hierarchical_search(query_vector, k, n_probe_per_level)
    
    def _search_single_level(self, query_vector, k):
        """Search in single-level case."""
        lowest_level_kmeans = self.kmeans_builder.get_level_kmeans(0)
        _, assignments = lowest_level_kmeans.index.search(query_vector.reshape(1, -1), k)
        
        best_heap = []
        for cluster_id in assignments[0]:
            if cluster_id in self.hierarchy_metadata.cluster_metadata[0]:
                for dist_to_centroid, cos_sim, vector_idx in self.hierarchy_metadata.cluster_metadata[0][cluster_id]:
                    actual_dist = np.linalg.norm(query_vector - self.data[vector_idx])
                    best_heap.append((actual_dist, vector_idx))
        
        best_heap.sort()
        return best_heap[:k]
    
    def search(self, query_vector, k, N_PROBE=1, probe_strategy="nprobe", tshirt_size="small", use_hierarchical=True):
        """Main search interface with option to use hierarchical or legacy search.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            N_PROBE: Number of probes (for legacy compatibility)
            probe_strategy: Strategy for probing (for legacy compatibility)
            tshirt_size: T-shirt sizing strategy (for legacy compatibility)
            use_hierarchical: If True, use true hierarchical search; if False, use legacy approach
        """
        if use_hierarchical:
            # Use new hierarchical search
            n_probe_per_level = max(1, N_PROBE) if probe_strategy == "nprobe" else {"small": 1, "medium": 2, "large": 3}.get(tshirt_size, 1)
            return self.search_query_hierarchical(query_vector, k, n_probe_per_level)
        else:
            # Use legacy cross-pollination approach for compatibility
            return self.search_query_cross_pollination_legacy(query_vector, k, N_PROBE, probe_strategy, tshirt_size)
    
    def get_level_info(self):
        """Get information about each level in the hierarchy."""
        if not self.built:
            return "Index not built"
            
        info = []
        for i in range(self.kmeans_builder.num_levels()):
            kmeans = self.kmeans_builder.get_level_kmeans(i)
            info.append(f"Level {i}: {kmeans.centroids.shape[0]} clusters")
        return "\n".join(info)
    
    def get_metadata_info(self):
        """Get information about the metadata structure."""
        return self.hierarchy_metadata.get_metadata_stats()
    
    def trace_vector_hierarchy(self, vector_idx):
        """Trace a vector through the complete hierarchy.
        
        Args:
            vector_idx: Index of vector in original dataset
            
        Returns:
            dict: Information about the vector's path through hierarchy
        """
        if not self.built:
            raise ValueError("Index must be built first")
        
        # Find which lowest-level cluster this vector belongs to
        lowest_level_kmeans = self.kmeans_builder.get_level_kmeans(0)
        vector = self.data[vector_idx].reshape(1, -1)
        _, assignments = lowest_level_kmeans.index.search(vector, 1)
        lowest_cluster = assignments[0][0]
        
        # Trace path through hierarchy
        if self.hierarchy_metadata.child_to_parent is not None:
            path = self.hierarchy_metadata.trace_path_to_highest(lowest_cluster)
            return {
                'vector_idx': vector_idx,
                'hierarchy_path': path,
                'level_clusters': {f"Level {i}": path[i] for i in range(len(path))}
            }
        else:
            return {
                'vector_idx': vector_idx,
                'hierarchy_path': [lowest_cluster],
                'level_clusters': {'Level 0': lowest_cluster}
            }
    
    def __repr__(self):
        if not self.built:
            return f"MultiLevelIndex(n_lowest_clusters={self.n_lowest_clusters}, built=False)"
        return f"MultiLevelIndex(n_lowest_clusters={self.n_lowest_clusters}, levels={self.kmeans_builder.num_levels()}, built=True)"
