"""Test script for adaptive features."""

import sys
import os
import traceback
import torch
import numpy as np
import logging
from datetime import datetime
import pandas as pd
from itertools import product

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
logger.debug(f"Python path: {sys.path}")

try:
    from dynadetectv2.core.experimental import (
        AdaptiveNeighborhood,
        DynamicFeatureWeighting,
        UncertaintyQuantification,
        PerformanceMonitor
    )
    logger.debug("Successfully imported experimental modules")
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

def generate_test_data(
    n_samples: int,
    n_features: int,
    n_classes: int,
    pattern_strength: float = 0.5
) -> tuple:
    """Generate synthetic test data with varying density regions."""
    try:
        logger.debug(f"Generating test data with {n_samples} samples, {n_features} features...")
        X = []
        y = []
        
        # Generate class centers
        centers = torch.randn(n_classes, n_features)
        samples_per_class = n_samples // n_classes
        
        for class_idx in range(n_classes):
            # Generate samples for this class with varying density
            class_samples = torch.randn(samples_per_class, n_features)
            # Add class pattern
            class_samples = class_samples + centers[class_idx] * pattern_strength
            
            X.append(class_samples)
            y.extend([class_idx] * samples_per_class)
        
        X = torch.cat(X, dim=0)
        y = torch.tensor(y)
        
        # Shuffle data
        perm = torch.randperm(len(y))
        logger.debug("Successfully generated test data")
        return X[perm].cuda(), y[perm].cuda()
    except Exception as e:
        logger.error(f"Error generating test data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def run_experiment(config: dict) -> dict:
    """Run a single experiment with given configuration."""
    try:
        logger.info(f"Running experiment with config: {config}")
        results = {}
        
        # Generate data
        X, y = generate_test_data(
            n_samples=config['n_samples'],
            n_features=config['n_features'],
            n_classes=config['n_classes'],
            pattern_strength=config['pattern_strength']
        )
        
        # Test adaptive neighborhood
        adaptive_k = AdaptiveNeighborhood(
            base_k=config['base_k'],
            min_k=config['min_k'],
            max_k=config['max_k'],
            bandwidth=config['bandwidth']
        )
        
        adaptive_k.fit(X)
        k_values = adaptive_k.get_adaptive_k(X)
        
        results.update({
            'mean_k': float(k_values.float().mean()),
            'min_k': int(k_values.min()),
            'max_k': int(k_values.max()),
            'k_std': float(k_values.float().std())
        })
        
        # Test feature weighting
        feature_weights = DynamicFeatureWeighting(
            n_features=config['n_features'],
            learning_rate=config['learning_rate'],
            regularization=config['regularization']
        )
        
        feature_weights.initialize_weights(X.device)
        distances = torch.cdist(X, X)
        feature_weights.update_weights(X, y, distances)
        
        weights = feature_weights.weights.detach()
        results.update({
            'mean_weight': float(weights.mean()),
            'weight_std': float(weights.std()),
            'min_weight': float(weights.min()),
            'max_weight': float(weights.max())
        })
        
        # Test uncertainty quantification
        uncertainty = UncertaintyQuantification(config['n_classes'])
        
        # Generate mock probabilities
        probs = torch.rand(len(X), config['n_classes']).cuda()
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        # Use k nearest neighbors for uncertainty
        k_dist = torch.cdist(X, X)[:, :config['base_k']]
        
        aleatoric, epistemic = uncertainty.compute_uncertainty(probs, k_dist)
        
        results.update({
            'mean_aleatoric': float(aleatoric.mean()),
            'aleatoric_std': float(aleatoric.std()),
            'mean_epistemic': float(epistemic.mean()),
            'epistemic_std': float(epistemic.std())
        })
        
        # Add configuration to results
        results.update(config)
        return results
        
    except Exception as e:
        logger.error(f"Error in experiment: {str(e)}")
        logger.error(traceback.format_exc())
        return {**config, 'error': str(e)}

def run_adaptive_tests():
    """Run all adaptive feature tests with multiple configurations."""
    try:
        # Define parameter grid
        param_grid = {
            'n_samples': [1000, 5000],
            'n_features': [64, 128],
            'n_classes': [5, 10],
            'pattern_strength': [0.3, 0.5, 0.7],
            'base_k': [5, 7],
            'min_k': [3],
            'max_k': [15],
            'bandwidth': [0.1, 0.2],
            'learning_rate': [0.01, 0.001],
            'regularization': [0.1, 0.01]
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
            try:
                result = run_experiment(config)
                results.append(result)
                logger.info(f"Results: mean_k={result['mean_k']:.2f}, mean_weight={result['mean_weight']:.4f}")
            except Exception as e:
                logger.error(f"Error in configuration {i}: {str(e)}")
                continue
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f"results/adaptive_test_full_{timestamp}.csv", index=False)
        
        # Create summary with mean and std for each metric grouped by key parameters
        group_cols = ['n_samples', 'n_features', 'pattern_strength', 'base_k']
        metric_cols = ['mean_k', 'mean_weight', 'mean_aleatoric', 'mean_epistemic']
        
        summary = results_df.groupby(group_cols)[metric_cols].agg(['mean', 'std']).round(4)
        summary.to_csv(f"results/adaptive_summary_{timestamp}.csv")
        
        # Print summary
        logger.info("\nResults Summary:")
        logger.info("\nTop 5 Configurations by k-value stability (lowest k std):")
        logger.info(results_df.nsmallest(5, 'k_std')[group_cols + ['mean_k', 'k_std']].to_string())
        
        logger.info("\nTop 5 Configurations by feature weight discrimination (highest weight std):")
        logger.info(results_df.nlargest(5, 'weight_std')[group_cols + ['mean_weight', 'weight_std']].to_string())
        
        logger.info("\nTop 5 Configurations by uncertainty balance (similar aleatoric and epistemic):")
        results_df['uncertainty_balance'] = abs(results_df['mean_aleatoric'] - results_df['mean_epistemic'])
        logger.info(results_df.nsmallest(5, 'uncertainty_balance')[group_cols + ['mean_aleatoric', 'mean_epistemic']].to_string())
        
    except Exception as e:
        logger.error(f"Error in test suite: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logger.debug("Starting test script...")
        run_adaptive_tests()
        logger.debug("Test script completed successfully")
    except Exception as e:
        logger.error(f"Test script failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1) 