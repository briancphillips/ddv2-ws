"""Evaluation components for DynaDetect v2."""

from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
import logging
import time
import json
import csv
import os
from sklearn.base import BaseEstimator
from ..core.trainer import DynaDetectTrainer
from ..core.dataset import DatasetHandler
from ..core.models import ModelFactory
from .metrics import compute_metrics


class ModelEvaluator:
    """Handles model evaluation and metrics computation."""
    
    def __init__(self, classifier_name: str = 'RF', mode: str = 'standard'):
        """Initialize model evaluator."""
        self.classifier_name = classifier_name
        self.mode = mode
        self.model = None
        self.dataset_name = None
        self.modification_method = None
        self.num_poisoned = 0
        self.attack_params = None
        self.poison_rate = None
        self.results = []
        
    def train_model(
        self,
        model: BaseEstimator,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> BaseEstimator:
        """Train the model with given features and labels."""
        if sample_weights is not None:
            model.fit(features, labels, sample_weight=sample_weights)
        else:
            model.fit(features, labels)
        return model
    
    def predict(
        self,
        model: BaseEstimator,
        features: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Make predictions and measure latency."""
        start_time = time.time()
        predictions = model.predict(features)
        latency = time.time() - start_time
        return predictions, latency
    
    def process_training_set(self, dataset: torch.utils.data.Dataset) -> np.ndarray:
        """Process training dataset into features."""
        features = []
        for data, _ in dataset:
            if isinstance(data, torch.Tensor):
                features.append(data.numpy().flatten())
            else:
                features.append(np.array(data).flatten())
        return np.array(features)
    
    def process_validation_set(self, dataset: torch.utils.data.Dataset) -> np.ndarray:
        """Process validation dataset into features."""
        return self.process_training_set(dataset)
    
    def log_results(
        self,
        metrics: Dict[str, Any],
        dataset_name: str,
        modification_method: str,
        num_poisoned: int,
        poisoned_classes: List[Any],
        flip_type: str,
        latency: float,
        iteration: int,
        classifier_name: str,
        total_images: int,
        timestamp: str
    ) -> None:
        """Log evaluation results."""
        # Create results dictionary
        result = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'modification': modification_method,
            'poisoned_samples': num_poisoned,
            'total_samples': total_images,
            'poison_rate': num_poisoned / total_images if total_images > 0 else 0,
            'flip_type': flip_type,
            'classifier': classifier_name,
            'mode': self.mode,
            'iteration': iteration,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'latency': latency
        }
        
        # Add to results list
        self.results.append(result)
        
        # Log to console
        log_msg = (
            f"\nEvaluation Results (Iteration {iteration}):\n"
            f"Dataset: {dataset_name}\n"
            f"Modification: {modification_method}\n"
            f"Poisoned Samples: {num_poisoned}/{total_images}\n"
            f"Flip Type: {flip_type}\n"
            f"Classifier: {classifier_name}\n"
            f"Metrics: {metrics}\n"
            f"Latency: {latency:.4f}s\n"
        )
        logging.info(log_msg)
        
        # Save results
        self.save_results(timestamp)
        
    def save_results(self, timestamp: str) -> None:
        """Save results to JSON and CSV files."""
        # Create results directory if it doesn't exist
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(base_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save to JSON
        json_file = os.path.join(results_dir, f'experiment_results_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        # Save to CSV
        csv_file = os.path.join(results_dir, f'experiment_results_{timestamp}.csv')
        if self.results:
            fieldnames = self.results[0].keys()
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)


class DatasetEvaluator(ModelEvaluator):
    """Evaluates datasets with different models and poisoning methods."""
    
    def __init__(self, config: Dict[str, Any], classifier_name: str):
        """Initialize evaluator with configuration."""
        super().__init__(classifier_name=classifier_name, mode=config.get('mode', 'standard'))
        
        self.dataset_name = config['name']
        self.dataset_type = config.get('type', 'image')
        self.sample_size = config.get('sample_size')
        self.modification_method = config.get('modification_method')
        self.attack_params = config.get('attack_params', {})
        self.poison_rate = config.get('poison_rate', 0.0)
        self.metric = config.get('metric', 'cosine')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.dataset_handler = DatasetHandler(
            dataset_name=self.dataset_name,
            dataset_type=self.dataset_type,
            sample_size=self.sample_size
        )
        
        self.model_factory = ModelFactory()
        
        # Initialize DynaDetect trainer if needed
        if self.mode == 'dynadetect':
            self.trainer = DynaDetectTrainer(
                n_components=100,
                contamination=0.1
            )
        else:
            self.trainer = None
            
    def run_evaluation(self, iteration: int = 0, timestamp: str = None) -> None:
        """Run evaluation pipeline.
        
        Args:
            iteration: Iteration number
            timestamp: Timestamp for logging
        """
        # Load and preprocess data
        train_dataset, val_dataset = self.dataset_handler.load_data()
        
        # Process training set
        train_features = self.process_training_set(train_dataset)
        train_labels = np.array(self.dataset_handler.get_targets(train_dataset))
        
        # Process validation set
        val_features = self.process_validation_set(val_dataset)
        val_labels = np.array(self.dataset_handler.get_targets(val_dataset))
        
        # Create and train model
        model = self.model_factory.create_model(self.classifier_name)
        
        if self.mode == 'dynadetect':
            # Apply DynaDetect training
            train_features, sample_weights = self.trainer.fit_transform(train_features, train_labels)
            model = self.train_model(model, train_features, train_labels, sample_weights)
            val_features = self.trainer.transform(val_features)
        else:
            # Standard training
            model = self.train_model(model, train_features, train_labels)
        
        # Make predictions
        predictions, latency = self.predict(model, val_features)
        
        # Compute and log metrics
        metrics = compute_metrics(val_labels, predictions)
        self.log_results(
            metrics=metrics,
            dataset_name=self.dataset_name,
            modification_method=self.modification_method,
            num_poisoned=len(train_dataset) * self.poison_rate,
            poisoned_classes=[],  # TODO: Track poisoned classes
            flip_type=self.attack_params.get('mode', 'none'),
            latency=latency,
            iteration=iteration,
            classifier_name=self.classifier_name,
            total_images=len(train_dataset),
            timestamp=timestamp
        )
