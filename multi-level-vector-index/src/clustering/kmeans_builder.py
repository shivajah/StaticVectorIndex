from collections import defaultdict
import faiss
import numpy as np

class KMeansBuilder:
    def __init__(self, n_lowest_clusters):
        """ 
        levels : list that stores the actual KMeans objects for each level of the hierarchy 
        level_assignments: list that stores the assignments of data points to clusters at each level
        """
        self.n_lowest_clusters = n_lowest_clusters
        self.levels = []
        self.level_assignments = []  # Store assignments for each level

    def build_kmeans(self, data, n_clusters, niter=20):
        """Build a single K-means model."""
        d = data.shape[1]
        # Sample data if too large for training
        # Randomly sample up to 50,000 points from the input data for training.
        # If dataset is smaller than 50,000, use the entire dataset.
        train_sample = data[np.random.choice(len(data), size=min(50000, len(data)), replace=False)]
        
        # Initialize a FAISS KMeans object.
        # d: dimensionality of the data
        # n_clusters: number of clusters to form
        # niter: number of iterations for the k-means algorithm
        # verbose: print progress during training
        # spherical: use standard k-means (not spherical)
        kmeans = faiss.Kmeans(d, n_clusters, niter=niter, verbose=True, spherical=False)
        
        # Train the k-means model on the sampled data.
        kmeans.train(train_sample)
        return kmeans

    def create_multi_level_clusters(self, data):
        """Create hierarchical multi-level clustering.
        
        Algorithm:
        1. Start with the lowest level (most clusters = n_lowest_clusters)
        2. Build upper levels by dividing by 20 until < 10 clusters
        3. Store each level and assignments
        """
        print(f"Building multi-level index with {self.n_lowest_clusters} lowest-level clusters")
        
        # Level 0: Lowest level with most clusters
        level_0_kmeans = self.build_kmeans(data, self.n_lowest_clusters)
        self.levels.append(level_0_kmeans)
        
        # Get assignments for original data to level 0
        _, level_0_assignments = level_0_kmeans.index.search(data, 1)
        self.level_assignments.append(level_0_assignments)
        
        # Build upper levels using centroids
        current_centroids = level_0_kmeans.centroids
        level = 1
        
        while True:
            n_upper_clusters = max(10, len(current_centroids) // 20)
            if n_upper_clusters >= len(current_centroids) or n_upper_clusters < 10:
                break
                
            print(f"Level {level}: Building {n_upper_clusters} clusters from {len(current_centroids)} centroids")
            
            # Build k-means for this level
            upper_kmeans = self.build_kmeans(current_centroids, n_upper_clusters)
            self.levels.append(upper_kmeans)
            
            # Get assignments of lower-level centroids to upper-level clusters
            _, upper_assignments = upper_kmeans.index.search(current_centroids, 1)
            self.level_assignments.append(upper_assignments)
            
            # Move to next level
            current_centroids = upper_kmeans.centroids
            level += 1
            
        print(f"Built {len(self.levels)} levels:")
        for i, level_kmeans in enumerate(self.levels):
            print(f"  Level {i}: {level_kmeans.centroids.shape[0]} clusters")

    def get_level_kmeans(self, level):
        """Get K-means model for a specific level."""
        if 0 <= level < len(self.levels):
            return self.levels[level]
        return None
    
    def get_assignments_to_level(self, level):
        """Get assignments to a specific level."""
        if 0 <= level < len(self.level_assignments):
            return self.level_assignments[level]
        return None

    def get_centroids(self):
        """Get centroids for all levels."""
        return [kmeans.centroids for kmeans in self.levels]
    
    def num_levels(self):
        """Get the number of levels in the hierarchy."""
        return len(self.levels)