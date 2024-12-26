"""Test script for DynaDetect implementation."""

import logging
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from experiments.config import get_test_configs, DatasetConfig
from experiments.results_collector import ResultsCollector
from dynadetectv2 import DatasetEvaluator, DynaDetectTrainer

def run_test():
    """Run test experiments focusing on CIFAR100 validation."""
    try:
        # Setup logging with both stdout and file
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'validation')
        log_file = os.path.join(log_dir, 'validation_test.log')
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'validation')
        
        # Ensure directories exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Remove existing log file if it exists
        if os.path.exists(log_file):
            os.remove(log_file)
        
        # Create console handler with higher log level
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        
        # Get the root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        logger.handlers = []
        
        # Add the handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        print("Starting DynaDetect Validation Tests...")  # Direct print for immediate feedback
        logging.info("\n=== Starting DynaDetect Validation Tests ===")
        logging.info(f"Log file location: {log_file}")
        sys.stdout.flush()  # Force stdout flush
        
        # Initialize results collector
        results_collector = ResultsCollector(output_dir=results_dir)
        
        def convert_to_serializable(obj):
            """Convert numpy arrays and other non-serializable objects to Python native types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        # Define test parameters
        sample_sizes = {
            'CIFAR100': [1000, 2000, 5000],  # Minimum 10 samples per class
            'GTSRB': [1000, 2000, 4000],     # About 23-93 samples per class
            'ImageNette': [1000, 2000, 4000], # About 100-400 samples per class
        }
        
        classifiers = ['SoftKNN']  # Start with SoftKNN, expand later
        
        # Baseline testing (no poisoning)
        logging.info("\n=== Running Baseline Tests (No Poisoning) ===")
        for dataset_name in ['CIFAR100', 'GTSRB', 'ImageNette']:
            logging.info(f"\n=== Testing {dataset_name} Baseline ===")
            for size in sample_sizes[dataset_name]:
                logging.info(f"\n=== Sample Size: {size} ===")
                config = DatasetConfig(
                    name=dataset_name,
                    dataset_type='image',
                    sample_size=size,
                    metric='accuracy',
                    modification_method='none',
                    poison_rate=0.0,
                    attack_params={},
                    mode='standard'
                )
                evaluator = DatasetEvaluator(config, 'SoftKNN')
                evaluator.run_evaluation(0)
        
        # Poisoning tests with established baselines
        logging.info("\n=== Running Poisoning Tests ===")
        poison_rates = [0.01, 0.05, 0.10]  # Test multiple poison rates
        
        for dataset_name in ['CIFAR100', 'GTSRB', 'ImageNette']:
            for size in sample_sizes[dataset_name]:
                for rate in poison_rates:
                    config = DatasetConfig(
                        name=dataset_name,
                        dataset_type='image',
                        sample_size=size,
                        metric='accuracy',
                        modification_method='none',
                        poison_rate=rate,
                        attack_params={},
                        mode='standard'
                    )
                    try:
                        logging.info("\n" + "="*50)
                        logging.info(f"Testing configuration:")
                        logging.info(f"Dataset: {config.name}")
                        logging.info(f"Mode: {config.mode}")
                        logging.info(f"Sample size: {config.sample_size}")
                        logging.info(f"Modification: {config.modification_method}")
                        logging.info(f"Poison rate: {config.poison_rate}")
                        logging.info(f"Classifier: {'SoftKNN'}")
                        
                        evaluator = DatasetEvaluator(config, 'SoftKNN')
                        metrics, num_poisoned, poisoned_classes, flip_type, latency = evaluator.run_evaluation()
                        
                        # Save results
                        result = {
                            'dataset': config.name,
                            'mode': config.mode,
                            'sample_size': config.sample_size,
                            'modification_method': config.modification_method,
                            'poison_rate': config.poison_rate,
                            'classifier': 'SoftKNN',
                            'hyperparameters': None,
                            'metrics': convert_to_serializable(metrics),
                            'num_poisoned': int(num_poisoned),
                            'poisoned_classes': convert_to_serializable(poisoned_classes),
                            'flip_type': flip_type,
                            'latency': float(latency)
                        }
                        results_collector.add_result(result)
                        
                        # Log results
                        logging.info("\nResults:")
                        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
                        logging.info(f"Precision: {metrics['precision']:.4f}")
                        logging.info(f"Recall: {metrics['recall']:.4f}")
                        logging.info(f"F1 Score: {metrics['f1']:.4f}")
                        
                        logging.info(f"Number of samples poisoned: {num_poisoned}")
                        logging.info(f"Execution time: {latency:.2f} seconds")
                        sys.stdout.flush()
                    
                    except Exception as e:
                        logging.error(f"Error in configuration: {str(e)}")
                        continue
        
        logging.info("\nValidation tests completed")
        
    except Exception as e:
        logging.error(f"Critical error in test execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    run_test()
