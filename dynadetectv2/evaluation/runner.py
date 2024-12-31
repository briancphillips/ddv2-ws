"""
Experiment runner for DynaDetect v2.
"""

import logging
import time
from datetime import datetime
from typing import Optional, Callable

from ..config import ExperimentConfig
from ..core.dataset import DatasetHandler
from .evaluator import DatasetEvaluator
from ..core.models import ModelFactory

class ExperimentRunner:
    """Runner for DynaDetect experiments."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.progress_callback: Optional[Callable[[str, str, str, int], None]] = None
        
    def run(self, timestamp: Optional[str] = None) -> None:
        """
        Run the experiment.
        
        Args:
            timestamp: Optional timestamp for logging
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        self.logger.info("Starting experiment run")
        
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
                    for iteration in range(self.config.iterations):
                        if self.progress_callback:
                            self.progress_callback(dataset_config.name, classifier_name, mode, iteration)
                        
                        evaluator = DatasetEvaluator(dataset_config, classifier_name, mode=mode)
                        evaluator.run_evaluation(iteration=iteration, timestamp=timestamp)
                
            self.logger.info(f"Completed evaluation for {dataset_config.name}")
            
        self.logger.info("\nExperiment run completed") 