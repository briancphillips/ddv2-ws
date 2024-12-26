"""
Main experiment runner for DynaDetect v2 evaluation
"""
import logging
import time
import psutil
import argparse
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.config import (
    generate_experiment_configs,
    get_baseline_configs,
    get_classifiers,
    ExperimentConfig,
    DATASET_CONFIGS
)
from experiments.results_collector import ResultsCollector
from dynadetectv2 import DatasetEvaluator

def measure_memory_usage() -> float:
    """Measure current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def run_single_experiment(
    config: ExperimentConfig,
    classifier: str,
    collector: ResultsCollector,
    iteration: int
) -> None:
    """Run a single experiment with given configuration."""
    try:
        evaluator = DatasetEvaluator(config, classifier)
        
        # Measure baseline performance
        start_time = time.time()
        baseline_memory = measure_memory_usage()
        
        metrics, num_poisoned, poisoned_classes, flip_type, latency = evaluator.run_evaluation(iteration)
        
        end_memory = measure_memory_usage()
        
        # Add memory and timing metrics
        metrics.update({
            'total_time': time.time() - start_time,
            'memory_usage': end_memory - baseline_memory,
            'classifier': classifier,
            'sample_size': config.sample_size
        })
        
        # Save results
        collector.save_metrics(metrics, config, 'baseline', iteration)
        
    except Exception as e:
        logging.error(f"Error in experiment {config.name} with classifier {classifier}: {str(e)}")
        raise

def run_experiment_set(
    configs: list[ExperimentConfig],
    test_mode: bool,
    collector: ResultsCollector,
    iterations: int = 3
) -> None:
    """Run a set of experiments with given configurations."""
    classifiers = get_classifiers(test_mode)
    
    total_experiments = len(configs) * len(classifiers) * iterations
    current_experiment = 0
    
    for config in configs:
        for classifier in classifiers:
            for iteration in range(iterations):
                current_experiment += 1
                logging.info(f"Running experiment {current_experiment}/{total_experiments}")
                logging.info(f"Config: {config.name} (sample size: {config.sample_size}), "
                           f"Classifier: {classifier}, Iteration: {iteration + 1}")
                
                run_single_experiment(config, classifier, collector, iteration)

def setup_logging(log_file: str) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def validate_sample_size(sample_size: Optional[int]) -> None:
    """Validate sample size against dataset limits."""
    if sample_size is not None:
        min_sizes = [cfg['min_sample_size'] for cfg in DATASET_CONFIGS.values()]
        max_sizes = [cfg['max_sample_size'] for cfg in DATASET_CONFIGS.values()]
        
        min_allowed = min(min_sizes)
        max_allowed = max(max_sizes)
        
        if sample_size < min_allowed:
            raise ValueError(f"Sample size {sample_size} is below minimum allowed size {min_allowed}")
        if sample_size > max_allowed:
            raise ValueError(f"Sample size {sample_size} is above maximum allowed size {max_allowed}")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run DynaDetect v2 experiments')
    parser.add_argument('--test', action='store_true',
                      help='Run in test mode with reduced dataset')
    parser.add_argument('--iterations', type=int, default=3,
                      help='Number of iterations for each experiment')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to store results')
    parser.add_argument('--log-file', type=str, default='experiments.log',
                      help='Log file path')
    parser.add_argument('--baseline-only', action='store_true',
                      help='Run only baseline experiments')
    parser.add_argument('--poison-only', action='store_true',
                      help='Run only poisoning experiments')
    parser.add_argument('--sample-size', type=int,
                      help='Override default sample size for all datasets')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate sample size if provided
    if args.sample_size:
        validate_sample_size(args.sample_size)
    
    # Setup logging
    setup_logging(args.log_file)
    
    # Initialize results collector
    collector = ResultsCollector(args.output_dir)
    
    try:
        if not args.poison_only:
            # Run baseline experiments
            logging.info("Running baseline experiments...")
            baseline_configs = get_baseline_configs(args.test, args.sample_size)
            run_experiment_set(baseline_configs, args.test, collector, args.iterations)
        
        if not args.baseline_only:
            # Run poisoning experiments
            logging.info("Running poisoning experiments...")
            poison_configs = generate_experiment_configs(args.test, args.sample_size)
            run_experiment_set(poison_configs, args.test, collector, args.iterations)
        
        # Analyze results
        logging.info("Analyzing results...")
        collector.analyze_results()
        
        logging.info("Experiments completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during experiment execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
