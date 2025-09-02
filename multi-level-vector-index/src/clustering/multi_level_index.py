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
    

    def search(self, query_vector, k, n_probe_per_level=2, probe_strategy="nprobe", tshirt_size="small"):
        """Search for k nearest neighbors using hierarchical search.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            n_probe_per_level: Number of clusters to probe at each level
            probe_strategy: Probing strategy (maintained for compatibility)
            tshirt_size: Size parameter (maintained for compatibility)
        """
        if not self.built or not self.hierarchy_metadata.built:
            raise ValueError("Index and metadata must be built before searching")
        
        if (probe_strategy != "nprobe" or tshirt_size not in ["small", "medium", "large"]):
            raise ValueError("Invalid probe_strategy or tshirt_size")
        
        if (probe_strategy == 'tshirt'):
            if tshirt_size == 'small':
                n_probe_per_level = 1
            elif tshirt_size == 'medium':
                n_probe_per_level = 2
            elif tshirt_size == 'large':
                n_probe_per_level = 3
        
        return self.hierarchy_metadata.hierarchical_search(query_vector, k, n_probe_per_level, probe_strategy)
    
    
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
