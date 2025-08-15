import unittest
from src.data.loader import download_fashion_mnist, load_fashion_mnist

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.cache_path = "test_fashion_mnist.hdf5"
        self.url = "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"

    def test_download_fashion_mnist(self):
        download_fashion_mnist(self.cache_path, self.url)
        self.assertTrue(os.path.exists(self.cache_path))

    def test_load_fashion_mnist(self):
        download_fashion_mnist(self.cache_path, self.url)
        xb, xq, gt = load_fashion_mnist(self.cache_path)
        self.assertEqual(xb.shape[0], 60000)  # Fashion-MNIST training set size
        self.assertEqual(xq.shape[0], 10000)   # Fashion-MNIST test set size
        self.assertEqual(gt.shape[0], 10000)   # Ground truth neighbors size

    def tearDown(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)

if __name__ == '__main__':
    unittest.main()