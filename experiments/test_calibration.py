"""Test script for confidence calibration system."""

import sys
import os
import torch
import numpy as np
from dynadetectv2.core.experimental import (
    monitor,
    CalibratedDDKNN
)
import logging
import pandas as pd
from datetime import datetime
from itertools import product

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_data():
    """Load the most recent experiment results for testing."""
    results_dir = "results"
    files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not files:
        raise ValueError("No results files found")
        
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
    df = pd.read_csv(os.path.join(results_dir, latest_file))
    return df

def generate_synthetic_data(n_samples: int, n_features: int, n_classes: int, pattern_strength: float = 0.5):
    """Generate synthetic data with class patterns.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        n_classes: Number of classes
        pattern_strength: Strength of class-specific patterns (0-1)
    """
    X = torch.randn(n_samples, n_features).cuda()
    class_centers = torch.randn(n_classes, n_features).cuda()
    labels = torch.randint(0, n_classes, (n_samples,)).cuda()
    
    for i in range(n_classes):
        mask = (labels == i)
        X[mask] = X[mask] + class_centers[i] * pattern_strength
        
    return X, labels

def create_poison_mask(n_samples: int, poison_rate: float):
    """Create poison mask for given number of samples and poison rate."""
    poison_mask = torch.zeros(n_samples, dtype=torch.bool)
    poison_indices = torch.randperm(n_samples)[:int(n_samples * poison_rate)]
    poison_mask[poison_indices] = True
    return poison_mask.cuda()

def run_experiment(config: dict):
    """Run a single experiment with given configuration."""
    # Generate data
    X, labels = generate_synthetic_data(
        config['n_samples'],
        config['n_features'],
        config['n_classes'],
        config['pattern_strength']
    )
    
    # Create poison mask
    poison_mask = create_poison_mask(config['n_samples'], config['poison_rate'])
    
    # Split data
    train_size = int(config['train_ratio'] * config['n_samples'])
    X_train = X[:train_size]
    y_train = labels[:train_size]
    X_val = X[train_size:]
    y_val = labels[train_size:]
    val_poison_mask = poison_mask[train_size:]
    
    # Initialize and train model
    model = CalibratedDDKNN(
        n_neighbors=config['n_neighbors'],
        temperature=config['init_temperature'],
        calibration_method=config['calibration_method']
    ).cuda()
    
    with monitor.measure_time(f"{config['calibration_method']}_train"):
        model.fit(X_train, y_train)
    
    # Fit calibration
    with monitor.measure_time(f"{config['calibration_method']}_calibration"):
        model.fit_calibration(X_val, y_val, val_poison_mask)
    
    # Get predictions
    with torch.no_grad():
        probs = model(X_val)
    
    # Get metrics
    metrics = model.get_calibration_metrics(probs, y_val, val_poison_mask)
    
    # Add configuration to metrics
    metrics.update(config)
    return metrics

def run_calibration_test():
    """Run calibration system test with multiple configurations."""
    try:
        # Define parameter ranges to test
        param_grid = {
            'n_samples': [5000, 10000],
            'n_features': [128],
            'n_classes': [10],
            'pattern_strength': [0.3, 0.5, 0.7],
            'poison_rate': [0.1],
            'train_ratio': [0.8, 0.9],
            'n_neighbors': [3, 5, 7],
            'init_temperature': [0.5, 1.0, 2.0],
            'calibration_method': ['temperature', 'isotonic', 'ensemble']
        }
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        configs = [dict(zip(keys, v)) for v in product(*values)]
        
        logger.info(f"Running {len(configs)} different configurations...")
        
        # Run experiments
        results = []
        for i, config in enumerate(configs, 1):
            logger.info(f"\nRunning configuration {i}/{len(configs)}:")
            logger.info(f"Parameters: {config}")
            
            try:
                result = run_experiment(config)
                results.append(result)
                logger.info(f"Results: ECE={result['ece']:.4f}, Accuracy={result['accuracy']:.4f}")
            except Exception as e:
                logger.error(f"Error in configuration {i}: {str(e)}")
                continue
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f"results/calibration_test_full_{timestamp}.csv", index=False)
        
        # Create summary with mean and std for each metric grouped by key parameters
        group_cols = ['calibration_method', 'n_neighbors', 'pattern_strength', 'train_ratio']
        metric_cols = ['ece', 'accuracy', 'avg_confidence', 'clean_ece', 'poison_ece']
        
        summary = results_df.groupby(group_cols)[metric_cols].agg(['mean', 'std']).round(4)
        summary.to_csv(f"results/calibration_summary_{timestamp}.csv")
        
        # Print summary
        logger.info("\nResults Summary:")
        logger.info("\nTop 5 Configurations by ECE:")
        logger.info(results_df.nsmallest(5, 'ece')[group_cols + ['ece', 'accuracy']].to_string())
        
        logger.info("\nTop 5 Configurations by Accuracy:")
        logger.info(results_df.nlargest(5, 'accuracy')[group_cols + ['ece', 'accuracy']].to_string())
        
        logger.info("\nPerformance Monitoring Report:")
        logger.info(monitor.get_performance_report())
        
    except Exception as e:
        logger.error(f"Error in test suite: {str(e)}")
        raise

if __name__ == "__main__":
    run_calibration_test() 