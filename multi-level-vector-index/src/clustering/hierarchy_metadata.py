import numpy as np
from collections import defaultdict

class HierarchyMetadata:
    """Manages simplified hierarchical metadata for multi-level vector index.
    
    This class uses a memory-efficient approach:
    - Builds hierarchy mappings between levels (child-to-parent and parent-to-children)
    - Stores detailed vector metadata (dist, cos_sim, idx) only at the lowest level
    - Uses centroid-based navigation for higher levels to save memory
    - Supports efficient top-down hierarchical search
    """
    
    def __init__(self, kmeans_builder):
        self.kmeans_builder = kmeans_builder
        self.child_to_parent = None  # {level: {child_id: parent_id}}
        self.parent_to_children = None  # {level: {parent_id: [child_ids]}}
        self.cluster_metadata = None  # {level: {cluster_id: [(dist, cos_sim, idx)]}}
        self.built = False
    
    def build_hierarchy_mapping(self):
        """Build complete hierarchy mapping for both directions.
        
        Returns:
            tuple: (child_to_parent, parent_to_children) mappings
        """
        print("Building hierarchy mapping...")
        num_levels = self.kmeans_builder.num_levels()
        child_to_parent = {}
        parent_to_children = {}
        
        # Build mapping for each adjacent level pair
        for level in range(num_levels - 1):
            current_level_kmeans = self.kmeans_builder.get_level_kmeans(level)
            next_level_kmeans = self.kmeans_builder.get_level_kmeans(level + 1)
            
            # Map current level centroids to next level clusters
            current_centroids = current_level_kmeans.centroids
            _, assignments = next_level_kmeans.index.search(current_centroids, 1)
            
            # Build child-to-parent mapping: level → {child_cluster_id: parent_cluster_id}
            child_to_parent[level] = {
                child_id: assignments[child_id][0]
                for child_id in range(len(current_centroids))
            }
            
            # Build parent-to-children mapping: level+1 → {parent_cluster_id: [child_cluster_ids]}
            parent_to_children[level + 1] = defaultdict(list)
            for child_id, parent_id in child_to_parent[level].items():
                parent_to_children[level + 1][parent_id].append(child_id)
            
            print(f"  Level {level} → Level {level + 1}: {len(current_centroids)} clusters mapped")
        
        self.child_to_parent = child_to_parent
        self.parent_to_children = parent_to_children
        return child_to_parent, parent_to_children
    
    def trace_to_highest_level(self, cluster_id, start_level=0):
        """Trace a cluster ID from start_level to highest level.
        
        Args:
            cluster_id: Cluster ID at start_level
            start_level: Starting level (default: 0 = lowest level)
            
        Returns:
            int: Cluster ID at the highest level
        """
        if self.child_to_parent is None:
            raise ValueError("Hierarchy mapping must be built first")
        
        current_id = cluster_id
        current_level = start_level
        highest_level = self.kmeans_builder.num_levels() - 1
        
        # Traverse up the hierarchy
        while current_level < highest_level:
            if current_level in self.child_to_parent:
                current_id = self.child_to_parent[current_level][current_id]
            current_level += 1
        
        return current_id
    
    def trace_path_to_highest(self, cluster_id, start_level=0):
        """Trace complete path from start_level to highest level.
        
        Args:
            cluster_id: Cluster ID at start_level
            start_level: Starting level (default: 0 = lowest level)
            
        Returns:
            list: Path of cluster IDs from start_level to highest level
        """
        if self.child_to_parent is None:
            raise ValueError("Hierarchy mapping must be built first")
        
        path = [cluster_id]
        current_id = cluster_id
        current_level = start_level
        highest_level = self.kmeans_builder.num_levels() - 1
        
        # Traverse up the hierarchy
        while current_level < highest_level:
            if current_level in self.child_to_parent:
                current_id = self.child_to_parent[current_level][current_id]
                path.append(current_id)
            current_level += 1
        
        return path
    
    def build_hierarchical_metadata(self, data, N_CROSS=1):
        """Build simplified hierarchical metadata.
        
        Only stores detailed vector information (dist, cos_sim, idx) at the lowest level.
        Higher levels just use centroid-based navigation.
        
        Args:
            data: Input data vectors
            N_CROSS: Number of clusters each vector is assigned to at the lowest level
            
        Returns:
            dict: cluster_metadata[0][cluster_id] = list of (dist, cos_sim, idx) tuples for lowest level only
        """
        print(f"Building hierarchical metadata with N_CROSS = {N_CROSS}")
        
        num_levels = self.kmeans_builder.num_levels()
        
        # Ensure hierarchy mapping is built
        if self.child_to_parent is None:
            self.build_hierarchy_mapping()
        
        # Only build metadata for the lowest level (level 0)
        cluster_metadata = {}
        
        # Assign vectors to multiple clusters at the lowest level
        lowest_level_kmeans = self.kmeans_builder.get_level_kmeans(0)
        _, data_assignments = lowest_level_kmeans.index.search(data, N_CROSS)
        
        # Build detailed metadata only for level 0
        cluster_metadata[0] = defaultdict(list)
        for idx, cluster_ids in enumerate(data_assignments):
            for cluster_id in cluster_ids[:N_CROSS]:
                centroid = lowest_level_kmeans.centroids[cluster_id]
                vec = data[idx]
                dist = np.linalg.norm(vec - centroid)
                cos_sim = np.dot(vec, centroid) / (np.linalg.norm(vec) * np.linalg.norm(centroid) + 1e-8)
                cluster_metadata[0][cluster_id].append((dist, cos_sim, idx))
        
        # Sort metadata by distance at the lowest level
        for cluster_id in cluster_metadata[0]:
            cluster_metadata[0][cluster_id].sort()
        
        self.cluster_metadata = cluster_metadata
        self.built = True
        
        # Print statistics
        print(f"Built hierarchical metadata for {num_levels} levels")
        clusters_at_level = len(cluster_metadata[0])
        total_vectors = sum(len(vectors) for vectors in cluster_metadata[0].values())
        print(f"  Level 0: {clusters_at_level} clusters, {total_vectors} vector assignments")
        
        # For higher levels, we just use centroid-based navigation (no detailed metadata needed)
        for level in range(1, num_levels):
            level_kmeans = self.kmeans_builder.get_level_kmeans(level)
            print(f"  Level {level}: {level_kmeans.centroids.shape[0]} clusters (centroid-based navigation)")
        
        return cluster_metadata
    
    def get_candidate_clusters_at_level(self, query_vector, level, n_candidates=3, parent_clusters=None):
        """Get candidate clusters at a specific level.
        
        Args:
            query_vector: Query vector
            level: Level to search at
            n_candidates: Number of candidates to return
            parent_clusters: If provided, only consider children of these parent clusters
            
        Returns:
            list: Cluster IDs at the specified level
        """
        level_kmeans = self.kmeans_builder.get_level_kmeans(level)
        
        if parent_clusters is None:
            # Search all clusters at this level
            n_candidates = min(n_candidates, level_kmeans.centroids.shape[0])
            _, assignments = level_kmeans.index.search(query_vector.reshape(1, -1), n_candidates)
            return assignments[0].tolist()
        else:
            # Only search among children of parent clusters
            if level == 0 or level not in self.parent_to_children:
                # No parent-child mapping or at lowest level
                n_candidates = min(n_candidates, level_kmeans.centroids.shape[0])
                _, assignments = level_kmeans.index.search(query_vector.reshape(1, -1), n_candidates)
                return assignments[0].tolist()
            
            # Collect all children of parent clusters
            candidate_children = []
            for parent_id in parent_clusters:
                if parent_id in self.parent_to_children[level]:
                    candidate_children.extend(self.parent_to_children[level][parent_id])
            
            if not candidate_children:
                return []
            
            # Search only among these children
            candidate_centroids = level_kmeans.centroids[candidate_children]
            
            # Create temporary index for these centroids
            import faiss
            index = faiss.IndexFlatL2(candidate_centroids.shape[1])
            index.add(candidate_centroids)
            
            n_candidates = min(n_candidates, len(candidate_children))
            _, local_assignments = index.search(query_vector.reshape(1, -1), n_candidates)
            
            # Map back to original cluster IDs
            return [candidate_children[local_id] for local_id in local_assignments[0]]
    
    def hierarchical_search(self, query_vector, k, n_probe_per_level=2):
        """Perform simplified hierarchical search using centroid-based navigation.
        
        At higher levels: Use centroid distances to find closest clusters
        At lowest level: Use detailed vector metadata for final candidate selection
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            n_probe_per_level: Number of clusters to probe at each level
            
        Returns:
            list: (distance, vector_idx) tuples sorted by distance
        """
        if not self.built or self.cluster_metadata is None:
            raise ValueError("Hierarchical metadata must be built first")
        
        num_levels = self.kmeans_builder.num_levels()
        
        # Start from highest level (level num_levels - 1) using centroid-based search
        current_clusters = self.get_candidate_clusters_at_level(
            query_vector, num_levels - 1, n_probe_per_level
        )
        
        # Navigate down the hierarchy level by level using centroid distances
        for target_level in range(num_levels - 2, -1, -1):  # Go from second-highest to lowest
            if not current_clusters:
                break
            
            # Find children clusters at target_level that belong to current parent clusters
            next_clusters = []
            
            if target_level in self.child_to_parent:
                # Collect all children of current parent clusters
                for parent_cluster in current_clusters:
                    if parent_cluster in self.parent_to_children.get(target_level + 1, {}):
                        children = self.parent_to_children[target_level + 1][parent_cluster]
                        next_clusters.extend(children)
            
            if not next_clusters:
                # If no children found, search all clusters at target level
                target_level_kmeans = self.kmeans_builder.get_level_kmeans(target_level)
                n_candidates = min(n_probe_per_level * len(current_clusters), target_level_kmeans.centroids.shape[0])
                _, assignments = target_level_kmeans.index.search(query_vector.reshape(1, -1), n_candidates)
                next_clusters = assignments[0].tolist()
            else:
                # Select the best clusters among children using centroid distances
                if len(next_clusters) > n_probe_per_level * len(current_clusters):
                    target_level_kmeans = self.kmeans_builder.get_level_kmeans(target_level)
                    candidate_centroids = target_level_kmeans.centroids[next_clusters]
                    distances = np.linalg.norm(candidate_centroids - query_vector, axis=1)
                    sorted_indices = np.argsort(distances)
                    max_candidates = min(n_probe_per_level * len(current_clusters), len(next_clusters))
                    selected_indices = sorted_indices[:max_candidates]
                    next_clusters = [next_clusters[i] for i in selected_indices]
            
            current_clusters = next_clusters
        
        # At this point, current_clusters contains the lowest-level clusters to search
        if not current_clusters:
            return []
        
        # Now use detailed metadata at the lowest level for final candidate selection with pruning
        best_heap = []
        seen_indices = set()
        tau = float("inf")  # Current threshold for pruning
        
        for cluster_id in current_clusters:
            if cluster_id in self.cluster_metadata[0]:
                # Get the centroid for this cluster for distance calculations
                cluster_centroid = self.kmeans_builder.get_level_kmeans(0).centroids[cluster_id]
                d_qc = np.linalg.norm(query_vector - cluster_centroid)  # Distance from query to centroid
                
                # Use the stored detailed metadata at lowest level with pruning
                for dist_to_centroid, cos_sim, vector_idx in self.cluster_metadata[0][cluster_id]:
                    if vector_idx in seen_indices:
                        continue
                    
                    # Pruning step 1: Triangle inequality lower bound
                    # The minimum possible distance between query and vector is |d_qc - dist_to_centroid|
                    lower_bound = abs(d_qc - dist_to_centroid)
                    if lower_bound > tau:
                        continue  # Skip this vector, it can't be better than current best
                    
                    # Pruning step 2: Cosine law distance estimation
                    # Estimate distance using law of cosines: d² = d_qc² + dist_ic² - 2*d_qc*dist_ic*cos(θ)
                    est_dist = np.sqrt(max(0.0, d_qc ** 2 + dist_to_centroid ** 2 - 2 * d_qc * dist_to_centroid * cos_sim))
                    if est_dist > tau:
                        continue  # Skip this vector, estimated distance exceeds threshold
                    
                    # If we pass both pruning tests, calculate actual distance
                    actual_dist = np.linalg.norm(query_vector - self.data[vector_idx])
                    best_heap.append((actual_dist, vector_idx))
                    seen_indices.add(vector_idx)
                    
                    # Update tau if we have more than k results
                    if len(best_heap) > k:
                        best_heap.sort()
                        best_heap = best_heap[:k]
                        tau = best_heap[-1][0]  # Update threshold to worst distance in top-k
        # Sort and return top k
        best_heap.sort()
        return best_heap[:k]
    
    def get_data_reference(self, data):
        """Store reference to data for distance calculations."""
        self.data = data
    
    def get_metadata_stats(self):
        """Get statistics about the simplified metadata structure."""
        if not self.built or self.cluster_metadata is None:
            return "Metadata not built"
        
        stats = []
        num_levels = self.kmeans_builder.num_levels()
        stats.append(f"Hierarchy levels: {num_levels}")
        
        # Only level 0 has detailed metadata
        if 0 in self.cluster_metadata:
            clusters_at_level = len(self.cluster_metadata[0])
            vectors_at_level = sum(len(vector_list) for vector_list in self.cluster_metadata[0].values())
            stats.append(f"  Level 0: {clusters_at_level} clusters, {vectors_at_level} vector assignments")
        
        # Higher levels use centroid-based navigation
        for level in range(1, num_levels):
            level_kmeans = self.kmeans_builder.get_level_kmeans(level)
            clusters_at_level = level_kmeans.centroids.shape[0]
            stats.append(f"  Level {level}: {clusters_at_level} clusters (centroid-based navigation)")
        
        # Show parent-child relationships
        if self.parent_to_children:
            for level, parent_mapping in self.parent_to_children.items():
                avg_children = np.mean([len(children) for children in parent_mapping.values()])
                stats.append(f"  Level {level} avg children per parent: {avg_children:.1f}")
        
        return "\n".join(stats)
    
    def __repr__(self):
        if not self.built:
            return f"HierarchyMetadata(levels={self.kmeans_builder.num_levels()}, built=False)"
        return f"HierarchyMetadata(levels={self.kmeans_builder.num_levels()}, hierarchical=True, built=True)"
