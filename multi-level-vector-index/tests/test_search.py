import unittest
from src.clustering.multi_level_index import MultiLevelIndex
from src.clustering.kmeans_builder import build_kmeans
import numpy as np

class TestMultiLevelIndex(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = np.random.rand(100, 10).astype(np.float32)
        self.lowest_level_clusters = 5
        self.index = MultiLevelIndex(self.lowest_level_clusters)

    def test_add_clusters(self):
        # Test adding clusters to the multi-level index
        kmeans = build_kmeans(self.data, self.data.shape[1], self.lowest_level_clusters)
        self.index.add_clusters(kmeans.centroids)

        self.assertEqual(len(self.index.clusters), self.lowest_level_clusters)

    def test_retrieve_centroids(self):
        # Test retrieving centroids from the multi-level index
        kmeans = build_kmeans(self.data, self.data.shape[1], self.lowest_level_clusters)
        self.index.add_clusters(kmeans.centroids)

        centroids = self.index.get_centroids()
        self.assertEqual(len(centroids), self.lowest_level_clusters)

    def test_hierarchical_structure(self):
        # Test the hierarchical structure of the multi-level index
        kmeans = build_kmeans(self.data, self.data.shape[1], self.lowest_level_clusters)
        self.index.add_clusters(kmeans.centroids)

        upper_level_clusters = self.index.create_upper_level_clusters()
        self.assertTrue(len(upper_level_clusters) < self.lowest_level_clusters)

if __name__ == '__main__':
    unittest.main()