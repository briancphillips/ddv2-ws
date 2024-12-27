"""
Experiment runner for DynaDetect v2.
"""

import logging
import time
from datetime import datetime
from typing import Optional

from ..config import ExperimentConfig
from ..core.dataset import DatasetHandler
from .evaluator import DatasetEvaluator
from ..core.models import ModelFactory

class ExperimentRunner:
    """Handles experiment execution and coordination."""
    
    def __init__(self, config: ExperimentConfig, test_mode: bool = False):
        """
        Initialize ExperimentRunner.
        
        Args:
            config: Experiment configuration
            test_mode: Whether running in test mode
        """
        self.config = config
        self.test_mode = test_mode
        self.logger = logging.getLogger(__name__)
        
    def run(self, timestamp: Optional[str] = None) -> None:
        """
        Run the experiment.
        
        Args:
            timestamp: Optional timestamp for logging
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        self.logger.info("Starting experiment run")
        self.logger.info(f"Test mode: {self.test_mode}")
        
        total_datasets = len(self.config.datasets)
        total_classifiers = len(self.config.classifiers)
        total_modes = len(self.config.modes)
        total_iterations = self.config.iterations
        total_combinations = total_datasets * total_classifiers * total_modes * total_iterations
        current_combination = 0
        
        # Run evaluation for each dataset configuration
        for dataset_config in self.config.datasets:
            self.logger.info(f"\nProcessing dataset: {dataset_config.name}")
            
            # Initialize components
            dataset_handler = DatasetHandler(dataset_config)
            
            # Run evaluation for each classifier
            for classifier_name in self.config.classifiers:
                # Run for each mode
                for mode in self.config.modes:
                    # Run for each iteration
                    for iteration in range(total_iterations):
                        current_combination += 1
                        progress = (current_combination / total_combinations) * 100
                        
                        self.logger.info(f"\nProgress: {progress:.1f}% - Evaluating {dataset_config.name} with {classifier_name} ({mode} mode, iteration {iteration + 1}/{total_iterations})")
                        evaluator = DatasetEvaluator(dataset_config, classifier_name, mode=mode)
                        evaluator.run_evaluation(iteration=iteration, timestamp=timestamp)
                
            self.logger.info(f"Completed evaluation for {dataset_config.name}")
            
        self.logger.info("\nExperiment run completed") 