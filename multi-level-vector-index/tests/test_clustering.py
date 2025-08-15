import unittest
from src.clustering.multi_level_index import MultiLevelIndex
from src.clustering.kmeans_builder import build_kmeans

class TestMultiLevelClustering(unittest.TestCase):

    def setUp(self):
        self.data = [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]]
        self.lowest_level_clusters = 5
        self.index = MultiLevelIndex(self.lowest_level_clusters)

    def test_add_clusters(self):
        self.index.add_clusters(self.data)
        self.assertGreater(len(self.index.levels), 0)
        self.assertEqual(len(self.index.levels[0].clusters), 1)

    def test_get_centroids(self):
        self.index.add_clusters(self.data)
        centroids = self.index.get_centroids()
        self.assertEqual(len(centroids), 1)

    def test_hierarchical_structure(self):
        self.index.add_clusters(self.data)
        self.assertTrue(hasattr(self.index, 'levels'))
        self.assertGreater(len(self.index.levels), 0)

    def test_build_kmeans(self):
        kmeans = build_kmeans(self.data, d=2, n_clusters=self.lowest_level_clusters)
        self.assertEqual(kmeans.k, self.lowest_level_clusters)

if __name__ == '__main__':
    unittest.main()