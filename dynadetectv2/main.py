"""Main entry point for DynaDetect v2."""

from typing import Optional
import logging
import time
from datetime import datetime
from dataclasses import asdict

from .core.dataset import DatasetHandler
from .evaluation.evaluator import DatasetEvaluator
from .core.models import ModelFactory
from .config import ExperimentConfig, DatasetConfig


def main(experiment_config: ExperimentConfig, timestamp: Optional[str] = None) -> None:
    """Run DynaDetect v2 evaluation.
    
    Args:
        experiment_config: Experiment configuration
        timestamp: Optional timestamp for logging
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    # Run evaluation for each dataset configuration
    for dataset_config in experiment_config.datasets:
        logging.info(f"Processing dataset: {dataset_config.name}")
        
        # Initialize components
        dataset_handler = DatasetHandler(dataset_config)
        
        # Run evaluation for each classifier
        for classifier_name in experiment_config.classifiers:
            logging.info(f"Running evaluation with classifier: {classifier_name}")
            evaluator = DatasetEvaluator(dataset_config, classifier_name)
            evaluator.run_evaluation(timestamp=timestamp)
            
        logging.info(f"Evaluation completed for {dataset_config.name}")
