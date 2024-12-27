"""Evaluation components for DynaDetect v2."""

from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
import logging
import time
import json
import csv
import os
from datetime import datetime
from sklearn.base import BaseEstimator
from ..core.trainer import DynaDetectTrainer
from ..core.dataset import DatasetHandler
from ..core.models import ModelFactory
from .metrics import calculate_metrics
from ..config import DatasetConfig
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class TimeoutError(Exception):
    pass

class time_limit:
    """Context manager for timeout."""
    def __init__(self, seconds):
        self.seconds = seconds
        self.timer = None
        
    def __enter__(self):
        import threading
        self.timer = threading.Timer(self.seconds, self._timeout)
        self.timer.start()
        
    def __exit__(self, type, value, traceback):
        if self.timer:
            self.timer.cancel()
            
    def _timeout(self):
        raise TimeoutError("Timed out!")

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
        sample_weights: Optional[np.ndarray] = None,
        timeout: int = 60  # 60 second timeout
    ) -> BaseEstimator:
        """Train the model with given features and labels."""
        try:
            with time_limit(timeout):
                if sample_weights is not None and not isinstance(model, (KNeighborsClassifier, SVC)):
                    # Ensure sample_weights matches the number of samples
                    if len(sample_weights) != len(features):
                        logging.warning(f"Sample weights length ({len(sample_weights)}) does not match features length ({len(features)}). Ignoring weights.")
                        model.fit(features, labels)
                    else:
                        model.fit(features, labels, sample_weight=sample_weights)
                else:
                    model.fit(features, labels)
                return model
        except TimeoutError:
            logging.warning(f"Model training timed out after {timeout} seconds. Using simpler model.")
            # Fall back to a simpler model
            if isinstance(model, LogisticRegression):
                model = LogisticRegression(
                    multi_class='ovr',  # One-vs-rest instead of multinomial
                    solver='lbfgs',
                    max_iter=100,
                    tol=1e-2,
                    n_jobs=-1
                )
            model.fit(features, labels)
            return model
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise
    
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
        
    def process_training_set(self, features: np.ndarray, labels: np.ndarray, mode: str) -> np.ndarray:
        """Process the training set based on the specified mode."""
        if mode == 'standard':
            return np.ones(len(features))  # No weighting for standard mode
        elif mode == 'dynadetect':
            trainer = DynaDetectTrainer(n_components=50)  # Initialize without dataset
            _, weights = trainer.fit_transform(features, labels)
            if len(weights) != len(features):
                logging.warning(f"Sample weights length ({len(weights)}) does not match features length ({len(features)}). Ignoring weights.")
                return np.ones(len(features))
            return weights
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
    def process_validation_set(self, dataset: torch.utils.data.Dataset) -> np.ndarray:
        """Process validation set."""
        return None
        
    def log_results(
        self,
        metrics: Dict[str, Any],
        dataset_name: str,
        modification_method: str,
        num_poisoned: int,
        poisoned_classes: List[int],
        flip_type: str,
        latency: float,
        iteration: int,
        classifier_name: str,
        total_images: int,
        timestamp: str
    ) -> None:
        """Log evaluation results."""
        result = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'classifier': classifier_name,
            'mode': self.mode,
            'iteration': iteration,
            'total_images': total_images,
            'num_poisoned': num_poisoned,
            'poisoned_classes': ','.join(map(str, poisoned_classes)),
            'modification_method': modification_method,
            'flip_type': flip_type,
            'latency': latency,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        }
        
        # Add per-class metrics
        for i, (prec, rec, f1) in enumerate(zip(
            metrics['precision_per_class'],
            metrics['recall_per_class'],
            metrics['f1_per_class']
        )):
            result[f'precision_class_{i}'] = prec
            result[f'recall_class_{i}'] = rec
            result[f'f1_class_{i}'] = f1
        
        self.results.append(result)
        
    def save_results(self, timestamp: str) -> None:
        """Save results to CSV file."""
        if not self.results:
            return
            
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(base_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f'experiment_results_{timestamp}.csv')
        
        # Get all unique keys from all results
        fieldnames = set()
        for result in self.results:
            fieldnames.update(result.keys())
        
        # Sort fieldnames to ensure consistent column order
        sorted_fieldnames = sorted(fieldnames, key=lambda x: (
            # Primary sort for main metrics
            0 if x in ['timestamp', 'dataset', 'classifier', 'iteration', 
                      'total_images', 'num_poisoned', 'poisoned_classes',
                      'modification_method', 'flip_type', 'latency',
                      'accuracy', 'precision', 'recall', 'f1'] else 1,
            # Secondary sort alphabetically
            x
        ))
        
        with open(results_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted_fieldnames)
            if f.tell() == 0:  # File is empty
                writer.writeheader()
            writer.writerows(self.results)


class DatasetEvaluator(ModelEvaluator):
    """Evaluator for dataset experiments."""
    
    def __init__(self, config: DatasetConfig, classifier_name: str, mode: str = 'standard'):
        """Initialize dataset evaluator."""
        super().__init__(classifier_name=classifier_name, mode=mode)
        self.config = config
        self.dataset_handler = DatasetHandler(config)
        self._cache = {}
        
    def run_evaluation(self, iteration: int = 0, timestamp: str = None) -> None:
        """Run evaluation for the dataset."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        start_time = time.time()
        
        # Get datasets
        logging.info("Loading datasets...")
        t0 = time.time()
        train_dataset = self.dataset_handler.get_train_dataset()
        val_dataset = self.dataset_handler.get_val_dataset()
        logging.info(f"Dataset loading completed in {time.time() - t0:.2f}s")
        
        # Extract features once for validation set
        logging.info("Extracting validation features...")
        t0 = time.time()
        cache_key = (id(val_dataset), len(val_dataset))
        if cache_key in self._cache:
            val_features, val_labels = self._cache[cache_key]
            logging.info("Using cached validation features")
        else:
            val_features, val_labels = self.dataset_handler.extract_features(val_dataset)
            self._cache[cache_key] = (val_features, val_labels)
        logging.info(f"Validation feature extraction completed in {time.time() - t0:.2f}s")
        
        # Extract features once for training set
        logging.info("Extracting training features...")
        t0 = time.time()
        cache_key = (id(train_dataset), len(train_dataset))
        if cache_key in self._cache:
            train_features, train_labels = self._cache[cache_key]
            logging.info("Using cached training features")
        else:
            train_features, train_labels = self.dataset_handler.extract_features(train_dataset)
            self._cache[cache_key] = (train_features, train_labels)
        logging.info(f"Training feature extraction completed in {time.time() - t0:.2f}s")
        
        # Get attack parameters
        attack_params = self.config.attack_params or {}
        poison_rates = attack_params.get('poison_rates', [0.0])
        attack_type = attack_params.get('type', 'none')
        attack_modes = attack_params.get('modes', ['random_to_random'])
        
        # Initialize model once
        model = ModelFactory.create_model(self.classifier_name)
        
        # For each poison rate
        for poison_rate in poison_rates:
            rate_start = time.time()
            logging.info(f"\nProcessing poison rate: {poison_rate}")
            
            # Apply poisoning if rate > 0
            if poison_rate > 0:
                t0 = time.time()
                poisoned_labels = self.dataset_handler.apply_label_flipping_to_labels(
                    train_labels.copy(),
                    poison_rate=poison_rate,
                    flip_type=attack_modes[0]
                )
                logging.info(f"Label flipping completed in {time.time() - t0:.2f}s")
            else:
                poisoned_labels = train_labels
            
            # Process datasets
            t0 = time.time()
            train_weights = self.process_training_set(train_features, train_labels, self.mode)
            logging.info(f"Training set processing completed in {time.time() - t0:.2f}s")
            
            # Train model with timeout
            t0 = time.time()
            try:
                self.model = self.train_model(model, train_features, poisoned_labels, train_weights, timeout=30)
            except TimeoutError:
                logging.warning("Model training timed out, using simpler model configuration")
                # Fall back to simpler model
                model = LogisticRegression(
                    multi_class='ovr',
                    solver='lbfgs',
                    max_iter=50,
                    tol=1e-1,
                    n_jobs=-1
                )
                self.model = self.train_model(model, train_features, poisoned_labels, train_weights, timeout=30)
            logging.info(f"Model training completed in {time.time() - t0:.2f}s")
            
            # Make predictions
            t0 = time.time()
            predictions, latency = self.predict(self.model, val_features)
            logging.info(f"Prediction completed in {time.time() - t0:.2f}s")
            
            # Compute and log metrics
            metrics = calculate_metrics(val_labels, predictions)
            self.log_results(
                metrics=metrics,
                dataset_name=self.config.name,
                modification_method=attack_type,
                num_poisoned=int(len(train_labels) * poison_rate) if poison_rate > 0 else 0,
                poisoned_classes=[],  # Will be populated by apply_label_flipping
                flip_type=attack_modes[0] if poison_rate > 0 else 'none',
                latency=latency,
                iteration=iteration,
                classifier_name=self.classifier_name,
                total_images=len(val_dataset),
                timestamp=timestamp
            )
            
            logging.info(f"Poison rate {poison_rate} completed in {time.time() - rate_start:.2f}s")
        
        # Save results
        self.save_results(timestamp)
        logging.info(f"Total evaluation time: {time.time() - start_time:.2f}s")
