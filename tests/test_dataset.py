import unittest
import numpy as np
import torch
import time
from dynadetectv2.core.dataset import DatasetHandler, NumericalDataset
from dataclasses import dataclass

@dataclass
class MockConfig:
    name: str
    sample_size: int
    dataset_type: str = 'numerical'

class TestDatasetHandler(unittest.TestCase):
    def setUp(self):
        # Create mock dataset
        X = np.random.randn(1000, 3072)  # Similar to CIFAR100 shape
        y = np.random.randint(0, 100, size=1000)
        self.dataset = NumericalDataset(X, y)
        self.config = MockConfig(name="WineQuality", sample_size=1000)
        
    def test_feature_extraction_caching(self):
        handler = DatasetHandler(self.config)
        
        # First extraction
        start_time = time.time()
        features1, labels1 = handler.extract_features(self.dataset)
        first_extraction_time = time.time() - start_time
        
        # Second extraction (should use cache)
        start_time = time.time()
        features2, labels2 = handler.extract_features(self.dataset)
        cached_extraction_time = time.time() - start_time
        
        # Verify cache is working
        self.assertTrue(np.array_equal(features1, features2))
        self.assertTrue(np.array_equal(labels1, labels2))
        
        # Cached extraction should be significantly faster
        self.assertLess(cached_extraction_time, first_extraction_time / 10,
                       f"Cached extraction ({cached_extraction_time:.3f}s) not significantly faster than first extraction ({first_extraction_time:.3f}s)")
        
        # Verify cache key is correct
        cache_key = (id(self.dataset), len(self.dataset))
        self.assertIn(cache_key, handler._feature_cache)
        
    def test_cache_with_different_datasets(self):
        handler = DatasetHandler(self.config)
        
        # Create two different datasets with same content
        X = np.random.randn(1000, 3072)
        y = np.random.randint(0, 100, size=1000)
        dataset1 = NumericalDataset(X, y)
        dataset2 = NumericalDataset(X.copy(), y.copy())
        
        # Extract features from both
        features1, _ = handler.extract_features(dataset1)
        features2, _ = handler.extract_features(dataset2)
        
        # Verify different datasets create different cache entries
        self.assertEqual(len(handler._feature_cache), 2)
        
        # But features should be equal since input data is same
        self.assertTrue(np.array_equal(features1, features2))
        
    def test_label_flipping_cache(self):
        handler = DatasetHandler(self.config)
        labels = np.random.randint(0, 10, size=1000)
        
        # First label flipping
        start_time = time.time()
        flipped1, params1 = handler.label_flipping(labels, mode='random_to_random', poison_rate=0.1)
        first_flip_time = time.time() - start_time
        
        # Second label flipping (should use cache)
        start_time = time.time()
        flipped2, params2 = handler.label_flipping(labels, mode='random_to_random', poison_rate=0.1)
        cached_flip_time = time.time() - start_time
        
        # Verify cache is working
        self.assertTrue(np.array_equal(flipped1, flipped2))
        self.assertEqual(params1, params2)
        
        # Cached operation should be significantly faster
        self.assertLess(cached_flip_time, first_flip_time / 10,
                       f"Cached label flipping ({cached_flip_time:.3f}s) not significantly faster than first flip ({first_flip_time:.3f}s)")
        
        # Different poison rate should create new cache entry
        flipped3, _ = handler.label_flipping(labels, mode='random_to_random', poison_rate=0.2)
        self.assertEqual(len(handler._label_flip_cache), 2)
        self.assertFalse(np.array_equal(flipped1, flipped3))

if __name__ == '__main__':
    unittest.main() 