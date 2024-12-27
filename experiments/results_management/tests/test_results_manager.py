"""
Tests for the ResultsManager class.
"""

import unittest
import os
import shutil
import json
import pandas as pd
from ..results_manager import ResultsManager

class TestResultsManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = 'test_results'
        self.results_manager = ResultsManager(base_dir=self.test_dir)
        
        # Sample result for testing
        self.sample_result = {
            'dataset': 'CIFAR100',
            'classifier': 'LogisticRegression',
            'mode': 'standard',
            'modification_method': 'label_flipping',
            'poison_rate': 0.01,
            'num_poisoned': 10,
            'latency': 1.234,
            'attack_params': {'flip_type': 'random_to_random'},
            'train_metrics': {
                'accuracy': 1.0,
                'f1': 1.0,
                'per_class_metrics': {
                    0: {'precision': 1.0, 'recall': 1.0}
                }
            },
            'test_metrics': {
                'accuracy': 0.95,
                'f1': 0.94,
                'per_class_metrics': {
                    0: {'precision': 0.95, 'recall': 0.93}
                }
            }
        }
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_initialization(self):
        """Test ResultsManager initialization."""
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertEqual(len(self.results_manager._current_batch), 0)
        
    def test_add_and_save_result(self):
        """Test adding and saving results."""
        self.results_manager.add_result(self.sample_result)
        self.assertEqual(len(self.results_manager._current_batch), 1)
        
        # Save with fixed timestamp
        timestamp = "20241226_000000"
        self.results_manager.save_batch(timestamp)
        
        # Check files exist
        json_file = os.path.join(self.test_dir, f'experiment_results_{timestamp}.json')
        csv_file = os.path.join(self.test_dir, f'experiment_results_{timestamp}.csv')
        summary_csv = os.path.join(self.test_dir, f'experiment_results_{timestamp}_summary.csv')
        
        self.assertTrue(os.path.exists(json_file))
        self.assertTrue(os.path.exists(csv_file))
        self.assertTrue(os.path.exists(summary_csv))
        
        # Check JSON content
        with open(json_file, 'r') as f:
            saved_results = json.load(f)
        self.assertEqual(len(saved_results), 1)
        self.assertEqual(saved_results[0]['dataset'], 'CIFAR100')
        
        # Check CSV content
        df = pd.read_csv(csv_file)
        self.assertEqual(len(df), 1)
        self.assertEqual(df['dataset'].iloc[0], 'CIFAR100')
        self.assertEqual(df['train_accuracy'].iloc[0], 1.0)
        
        # Check summary CSV content
        summary_df = pd.read_csv(summary_csv)
        self.assertEqual(len(summary_df), 1)
        self.assertEqual(summary_df['dataset'].iloc[0], 'CIFAR100')
        self.assertEqual(summary_df['train_accuracy'].iloc[0], 1.0)
        
    def test_load_results(self):
        """Test loading results."""
        self.results_manager.add_result(self.sample_result)
        timestamp = "20241226_000000"
        self.results_manager.save_batch(timestamp)
        
        # Load results
        loaded_results = self.results_manager.load_results(timestamp)
        self.assertEqual(len(loaded_results), 1)
        self.assertEqual(loaded_results[0]['dataset'], 'CIFAR100')
        
    def test_load_nonexistent_results(self):
        """Test loading nonexistent results raises error."""
        with self.assertRaises(FileNotFoundError):
            self.results_manager.load_results("nonexistent")
            
if __name__ == '__main__':
    unittest.main() 