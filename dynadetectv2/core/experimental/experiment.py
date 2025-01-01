"""Experiment wrapper for DynaDetect v2."""

from typing import Dict, List, Any, Optional, Tuple, Callable
import torch
import numpy as np
from dataclasses import dataclass
from .monitoring import monitor
from .model_monitoring import create_monitored_model
import pandas as pd
import logging
from datetime import datetime
import json
import os

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    model_class: type
    model_params: Dict[str, Any]
    poison_rates: List[float]
    datasets: List[str]
    attack_types: List[str]
    iterations: int = 1
    save_results: bool = True
    results_dir: str = "results/experiments"

class ExperimentRunner:
    """Runner for model experiments with monitoring."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results: List[Dict[str, Any]] = []
        self.monitored_model_class = create_monitored_model(config.model_class)
        
        # Ensure results directory exists
        if config.save_results:
            os.makedirs(config.results_dir, exist_ok=True)
            
    def run_single_experiment(
        self,
        dataset: str,
        poison_rate: float,
        attack_type: str,
        iteration: int
    ) -> Dict[str, Any]:
        """Run a single experiment with given parameters."""
        monitor.clear_metrics()
        monitor.set_context(
            dataset=dataset,
            poison_rate=poison_rate,
            attack_type=attack_type,
            iteration=iteration
        )
        
        model = self.monitored_model_class(**self.config.model_params)
        
        try:
            # Training phase
            with monitor.measure_time("training"):
                model.fit(X_train, y_train)  # Placeholder for actual training data
                monitor.record_memory("post_training")
            
            # Evaluation phase
            with monitor.measure_time("evaluation"):
                predictions = model.predict(X_test)  # Placeholder for actual test data
                monitor.record_memory("post_evaluation")
            
            # Get performance metrics
            metrics = model.get_performance_metrics()
            performance_report = monitor.get_performance_report()
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "experiment_name": self.config.name,
                "dataset": dataset,
                "poison_rate": poison_rate,
                "attack_type": attack_type,
                "iteration": iteration,
                "metrics": metrics,
                "performance": performance_report
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in experiment: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "experiment_name": self.config.name,
                "dataset": dataset,
                "poison_rate": poison_rate,
                "attack_type": attack_type,
                "iteration": iteration
            }
            
    def run_experiments(self):
        """Run all experiments according to configuration."""
        total_experiments = (
            len(self.config.datasets) *
            len(self.config.poison_rates) *
            len(self.config.attack_types) *
            self.config.iterations
        )
        
        self.logger.info(f"Starting {total_experiments} experiments")
        
        for dataset in self.config.datasets:
            for poison_rate in self.config.poison_rates:
                for attack_type in self.config.attack_types:
                    for iteration in range(self.config.iterations):
                        self.logger.info(
                            f"Running experiment: {dataset}, "
                            f"poison_rate={poison_rate}, "
                            f"attack={attack_type}, "
                            f"iteration={iteration}"
                        )
                        
                        result = self.run_single_experiment(
                            dataset=dataset,
                            poison_rate=poison_rate,
                            attack_type=attack_type,
                            iteration=iteration
                        )
                        
                        self.results.append(result)
                        
                        if self.config.save_results:
                            self._save_interim_results()
                            
        if self.config.save_results:
            self._save_final_results()
            
    def _save_interim_results(self):
        """Save current results to interim file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            self.config.results_dir,
            f"interim_{self.config.name}_{timestamp}.json"
        )
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
            
    def _save_final_results(self):
        """Save final results to CSV and JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join(
            self.config.results_dir,
            f"final_{self.config.name}_{timestamp}"
        )
        
        # Save detailed results as JSON
        with open(f"{base_filename}.json", 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Extract main metrics for CSV
        flat_results = []
        for result in self.results:
            if "error" in result:
                continue
                
            flat_result = {
                "timestamp": result["timestamp"],
                "dataset": result["dataset"],
                "poison_rate": result["poison_rate"],
                "attack_type": result["attack_type"],
                "iteration": result["iteration"]
            }
            
            # Add main metrics
            if "metrics" in result:
                for metric_name, metric_data in result["metrics"].items():
                    if isinstance(metric_data, dict) and "mean" in metric_data:
                        flat_result[f"{metric_name}_mean"] = metric_data["mean"]
                        flat_result[f"{metric_name}_std"] = metric_data["std"]
                    else:
                        flat_result[metric_name] = metric_data
                        
            flat_results.append(flat_result)
            
        # Save as CSV
        if flat_results:
            pd.DataFrame(flat_results).to_csv(
                f"{base_filename}.csv",
                index=False
            )
            
    def analyze_results(self) -> pd.DataFrame:
        """Analyze experiment results."""
        if not self.results:
            return pd.DataFrame()
            
        # Convert results to DataFrame
        flat_results = []
        for result in self.results:
            if "error" in result:
                continue
                
            flat_result = {
                "dataset": result["dataset"],
                "poison_rate": result["poison_rate"],
                "attack_type": result["attack_type"],
                "iteration": result["iteration"]
            }
            
            # Add metrics
            if "metrics" in result:
                for metric_name, metric_data in result["metrics"].items():
                    if isinstance(metric_data, dict) and "mean" in metric_data:
                        flat_result[f"{metric_name}_mean"] = metric_data["mean"]
                        
            flat_results.append(flat_result)
            
        df = pd.DataFrame(flat_results)
        
        # Add analysis
        analysis = df.groupby(["dataset", "poison_rate", "attack_type"]).agg({
            col: ["mean", "std"] for col in df.columns
            if col.endswith("_mean") and col != "iteration"
        }).round(4)
        
        return analysis