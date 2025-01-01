"""
Shared test fixtures for DynaDetect v2 tests.
"""

import os
import sys
import pytest
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

@pytest.fixture(scope='session')
def test_dir(tmp_path_factory):
    """Create and return a temporary test directory."""
    test_dir = tmp_path_factory.mktemp('test_dynadetect')
    (test_dir / 'logs').mkdir(exist_ok=True)
    (test_dir / 'results').mkdir(exist_ok=True)
    (test_dir / 'data').mkdir(exist_ok=True)
    return test_dir

@pytest.fixture(scope='session')
def setup_logging(test_dir):
    """Setup logging for tests."""
    log_file = test_dir / 'logs' / 'test.log'
    
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

@pytest.fixture(scope='session')
def results_dir(test_dir):
    """Return the results directory."""
    return test_dir / 'results'

@pytest.fixture(scope='session')
def data_dir(test_dir):
    """Return the data directory."""
    return test_dir / 'data' 