"""
Test suite for DynaDetect v2 implementation.
"""

import logging
import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from experiments.config import get_test_configs, DatasetConfig
from experiments.results_management import ResultsManager
from experiments.data_management import DatasetHandler, FeatureExtractor
from experiments.evaluator import Evaluator

@pytest.fixture
def setup_logging(tmp_path):
    """Setup logging for tests."""
    log_dir = tmp_path / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / 'test.log'
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    yield logger
    
    logger.removeHandler(console_handler)
    logger.removeHandler(file_handler)

@pytest.fixture
def dataset_config():
    """Get test dataset configuration."""
    return DatasetConfig(
        name='CIFAR100',
        num_classes=100,
        input_shape=(3, 32, 32),
        train_size=1000,
        test_size=100
    )

def test_dataset_handler(setup_logging, dataset_config):
    """Test DatasetHandler functionality."""
    handler = DatasetHandler(dataset_config)
    train_data = handler.get_train_data()
    test_data = handler.get_test_data()
    
    assert train_data is not None
    assert test_data is not None
    assert len(train_data) == dataset_config.train_size
    assert len(test_data) == dataset_config.test_size

def test_feature_extractor(setup_logging, dataset_config):
    """Test FeatureExtractor functionality."""
    handler = DatasetHandler(dataset_config)
    extractor = FeatureExtractor()
    
    train_data = handler.get_train_data()
    features = extractor.extract_features(train_data[0][0].unsqueeze(0))
    
    assert features is not None
    assert isinstance(features, torch.Tensor)
    assert features.dim() == 2

def test_evaluator(setup_logging, dataset_config, tmp_path):
    """Test Evaluator functionality."""
    results_dir = tmp_path / 'results'
    results_dir.mkdir(exist_ok=True)
    
    evaluator = Evaluator(
        dataset_config=dataset_config,
        results_dir=str(results_dir),
        classifier='LogisticRegression'
    )
    
    metrics = evaluator.evaluate()
    
    assert metrics is not None
    assert 'train_accuracy' in metrics
    assert 'test_accuracy' in metrics
    assert 'train_f1' in metrics
    assert 'test_f1' in metrics 