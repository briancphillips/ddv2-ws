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
from ..config import DatasetConfig, DATASET_SIZES
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier


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
    
    def __init__(self, classifier_name: str = 'RF', mode: str = 'standard', timeout: int = 600):
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
        self.timeout = timeout
        
    def train_model(
        self,
        model: BaseEstimator,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        timeout: int = 600  # Increased to 10 minutes
    ) -> BaseEstimator:
        """Train the model with given features and labels."""
        logging.info(f"Fitting {type(model).__name__} model with data shape: {features.shape}")
        
        try:
            with time_limit(timeout):
                if sample_weights is not None and not isinstance(model, (KNeighborsClassifier)):
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
            logging.warning(f"Model training timed out after {timeout} seconds. Using simpler configuration.")
            # Fall back to a simpler model configuration
            if isinstance(model, LogisticRegression):
                model = LogisticRegression(
                    multi_class='ovr',  # One-vs-rest instead of multinomial
                    solver='lbfgs',
                    max_iter=100,
                    tol=1e-2,
                    n_jobs=-1
                )
            elif isinstance(model, SGDClassifier):
                # For SVM (SGDClassifier), use an even simpler configuration
                logging.info("Creating simplified SGDClassifier configuration")
                model = SGDClassifier(
                    loss='hinge',
                    penalty='l2',
                    alpha=0.001,  # Stronger regularization
                    max_iter=25,   # Even fewer iterations
                    tol=1e-1,     # Even looser tolerance
                    learning_rate='constant',
                    eta0=0.1,     # Higher learning rate
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=2,  # Fewer iterations before early stopping
                    class_weight='balanced',
                    n_jobs=-1,
                    verbose=1     # Enable verbose output
                )
                
            try:
                logging.info("Attempting to fit with simplified model configuration")
                with time_limit(timeout // 2):  # Use half the original timeout for the simplified model
                    if sample_weights is not None and not isinstance(model, (KNeighborsClassifier)):
                        model.fit(features, labels, sample_weight=sample_weights)
                    else:
                        model.fit(features, labels)
                    logging.info("Successfully fitted simplified model")
                    return model
            except TimeoutError:
                logging.error("Simplified model also timed out. This is a critical issue that needs investigation.")
                raise
            except Exception as e:
                logging.error(f"Error fitting simplified model: {str(e)}")
                raise
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
        
        # Add confusion matrix
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    result[f'confusion_matrix_{i}_{j}'] = cm[i][j]
        
        # Add per-class metrics
        for i, (prec, rec, f1) in enumerate(zip(
            metrics['precision_per_class'],
            metrics['recall_per_class'],
            metrics['f1_per_class']
        )):
            result[f'precision_class_{i}'] = prec
            result[f'recall_class_{i}'] = rec
            result[f'f1_class_{i}'] = f1
            
        # Add class accuracies if available
        if 'class_accuracies' in metrics:
            for i, acc in enumerate(metrics['class_accuracies']):
                result[f'accuracy_class_{i}'] = acc
                
        # Add attack-specific metrics if available
        if 'attack_type' in metrics:
            result['attack_type'] = metrics['attack_type']
        if 'target_class' in metrics:
            result['target_class'] = metrics['target_class']
        if 'source_class' in metrics:
            result['source_class'] = metrics['source_class']
        if 'poison_rate' in metrics:
            result['poison_rate'] = metrics['poison_rate']
        if 'num_poisoned_samples' in metrics:
            result['num_poisoned_samples'] = metrics['num_poisoned_samples']
            
        # Add training metrics if available
        if 'train_metrics' in metrics:
            train_metrics = metrics['train_metrics']
            result['train_accuracy'] = train_metrics.get('accuracy', 0)
            result['train_precision'] = train_metrics.get('precision', 0)
            result['train_recall'] = train_metrics.get('recall', 0)
            result['train_f1'] = train_metrics.get('f1', 0)
            
            # Add per-class training metrics
            if 'precision_per_class' in train_metrics:
                for i, (prec, rec, f1) in enumerate(zip(
                    train_metrics['precision_per_class'],
                    train_metrics['recall_per_class'],
                    train_metrics['f1_per_class']
                )):
                    result[f'train_precision_class_{i}'] = prec
                    result[f'train_recall_class_{i}'] = rec
                    result[f'train_f1_class_{i}'] = f1
                    
        # Add validation metrics if available
        if 'val_metrics' in metrics:
            val_metrics = metrics['val_metrics']
            result['val_accuracy'] = val_metrics.get('accuracy', 0)
            result['val_precision'] = val_metrics.get('precision', 0)
            result['val_recall'] = val_metrics.get('recall', 0)
            result['val_f1'] = val_metrics.get('f1', 0)
            
            # Add per-class validation metrics
            if 'precision_per_class' in val_metrics:
                for i, (prec, rec, f1) in enumerate(zip(
                    val_metrics['precision_per_class'],
                    val_metrics['recall_per_class'],
                    val_metrics['f1_per_class']
                )):
                    result[f'val_precision_class_{i}'] = prec
                    result[f'val_recall_class_{i}'] = rec
                    result[f'val_f1_class_{i}'] = f1
                    
        # Add resource usage metrics
        if 'memory_usage' in metrics:
            result['memory_usage'] = metrics['memory_usage']
        if 'cpu_usage' in metrics:
            result['cpu_usage'] = metrics['cpu_usage']
        if 'gpu_usage' in metrics:
            result['gpu_usage'] = metrics['gpu_usage']
            
        # Add timing metrics
        if 'training_time' in metrics:
            result['training_time'] = metrics['training_time']
        if 'inference_time' in metrics:
            result['inference_time'] = metrics['inference_time']
        if 'feature_extraction_time' in metrics:
            result['feature_extraction_time'] = metrics['feature_extraction_time']
            
        # Add model-specific metrics
        if 'model_size' in metrics:
            result['model_size'] = metrics['model_size']
        if 'num_parameters' in metrics:
            result['num_parameters'] = metrics['num_parameters']
            
        # Add dataset-specific metrics
        if 'dataset_size' in metrics:
            result['dataset_size'] = metrics['dataset_size']
        if 'num_classes' in metrics:
            result['num_classes'] = metrics['num_classes']
        if 'class_distribution' in metrics:
            for i, count in enumerate(metrics['class_distribution']):
                result[f'class_{i}_count'] = count
                
        self.results.append(result)
        
    def save_results(self, timestamp: str) -> None:
        """Save results to CSV file and archive old results."""
        if not self.results:
            return
            
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(base_dir, 'results')
        archive_dir = os.path.join(results_dir, 'archive')
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(archive_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f'experiment_results_{timestamp}.csv')
        archive_file = os.path.join(archive_dir, f'experiment_results_{timestamp}.csv')
        
        # Load existing results if they exist
        existing_results = []
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    existing_results = list(reader)
                logging.info(f"Loaded {len(existing_results)} existing results")
            except Exception as e:
                logging.warning(f"Could not load existing results: {e}")
        
        # Combine existing and new results
        all_results = existing_results + self.results
        logging.info(f"Total results to save: {len(all_results)}")
        
        # Get all unique keys from all results
        fieldnames = set()
        for result in all_results:
            fieldnames.update(result.keys())
        
        # Sort fieldnames to ensure consistent column order
        sorted_fieldnames = sorted(fieldnames, key=lambda x: (
            # Primary sort for main metrics and metadata
            0 if x in ['timestamp', 'dataset', 'classifier', 'mode', 'iteration', 
                      'total_images', 'num_poisoned', 'poisoned_classes',
                      'modification_method', 'flip_type', 'latency'] else
            # Secondary sort for main performance metrics
            1 if x in ['accuracy', 'precision', 'recall', 'f1'] else
            # Tertiary sort for attack-specific metrics
            2 if x in ['attack_type', 'target_class', 'source_class', 'poison_rate', 'num_poisoned_samples'] else
            # Quaternary sort for per-class metrics
            3 if any(x.startswith(prefix) for prefix in ['precision_class_', 'recall_class_', 'f1_class_', 'accuracy_class_']) else
            # Quinary sort for confusion matrix
            4 if x.startswith('confusion_matrix_') else
            # Final sort alphabetically
            5,
            # Secondary sort alphabetically within each group
            x
        ))
        
        # Archive existing file if it exists
        if os.path.exists(results_file):
            os.replace(results_file, archive_file)
            logging.info(f"Archived existing results to {archive_file}")
        
        # Write all results to new file
        with open(results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted_fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        logging.info(f"Saved {len(all_results)} results to {results_file}")


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
        attack_mode = attack_params.get('mode', 'random_to_random')
        target_class = attack_params.get('target_class', None)
        source_class = attack_params.get('source_class', None)
        
        # Initialize model once
        model = ModelFactory.create_model(self.classifier_name)
        
        # For each poison rate
        for poison_rate in poison_rates:
            rate_start = time.time()
            logging.info(f"\nProcessing poison rate: {poison_rate}")
            
            # Apply poisoning if rate > 0
            if poison_rate > 0:
                t0 = time.time()
                poisoned_labels = self.dataset_handler.label_flipping(
                    labels=train_labels.copy(),
                    mode=attack_mode,
                    target_class=target_class,
                    source_class=source_class,
                    poison_rate=poison_rate
                )[0]
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
                self.model = self.train_model(model, train_features, poisoned_labels, train_weights, timeout=600)  # Increased to 10 minutes
            except TimeoutError:
                logging.warning("Model training timed out, using simpler configuration")
                # Create a new model instance with the same class
                model = ModelFactory.create_model(self.classifier_name)
                self.model = self.train_model(model, train_features, poisoned_labels, train_weights, timeout=600)  # Increased to 10 minutes
            except Exception as e:
                logging.error(f"Error during model training: {str(e)}")
                raise
            logging.info(f"Model training completed in {time.time() - t0:.2f}s")
            
            # Make predictions
            t0 = time.time()
            predictions, latency = self.predict(self.model, val_features)
            logging.info(f"Prediction completed in {time.time() - t0:.2f}s")
            
            # Compute and log metrics
            metrics = calculate_metrics(val_labels, predictions)
            metrics['poison_rate'] = poison_rate  # Add poison rate to metrics
            metrics['num_poisoned_samples'] = int(len(train_labels) * poison_rate) if poison_rate > 0 else 0
            self.log_results(
                metrics=metrics,
                dataset_name=self.config.name,
                modification_method=attack_type,
                num_poisoned=int(len(train_labels) * poison_rate) if poison_rate > 0 else 0,
                poisoned_classes=[],  # Will be populated by apply_label_flipping
                flip_type=attack_mode if poison_rate > 0 else 'none',
                latency=latency,
                iteration=iteration,
                classifier_name=self.classifier_name,
                total_images=len(train_labels),  # Use actual training set size
                timestamp=timestamp
            )
            
            logging.info(f"Poison rate {poison_rate} completed in {time.time() - rate_start:.2f}s")
        
        # Save results
        self.save_results(timestamp)
        logging.info(f"Total evaluation time: {time.time() - start_time:.2f}s")
