"""Model implementations for DynaDetect v2."""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SoftKNN(nn.Module):
    """Soft KNN classifier that can be used with gradient-based attacks."""

    def __init__(self, n_neighbors: int = 5, temperature: float = 1.0):
        """Initialize the Soft KNN classifier.

        Args:
            n_neighbors: Number of neighbors to use
            temperature: Temperature parameter for soft voting
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.temperature = temperature
        self.train_features: Optional[torch.Tensor] = None
        self.train_labels: Optional[torch.Tensor] = None
        self.device = device

    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor) -> None:
        """Store training data.

        Args:
            train_features: Training features
            train_labels: Training labels
        """
        logging.info(f"Training {self.__class__.__name__} on {len(train_features)} samples...")
        start_time = time.time()
        self.train_features = train_features.to(self.device)
        self.train_labels = train_labels.to(self.device)
        logging.info(f"Training completed in {time.time() - start_time:.2f}s")

    def compute_distances(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances between x and y.

        Args:
            x: First set of features
            y: Second set of features

        Returns:
            Tensor of pairwise distances
        """
        x_norm = (x**2).sum(1).view(-1, 1)
        y_norm = (y**2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return torch.clamp(dist, min=0.0).sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute soft KNN predictions.

        Args:
            x: Input features

        Returns:
            Predicted class probabilities
        """
        if self.train_features is None or self.train_labels is None:
            raise RuntimeError("Model must be fitted before making predictions")

        x = x.to(self.device)
        distances = self.compute_distances(x, self.train_features)

        # Get k nearest neighbors
        _, indices = torch.topk(distances, k=self.n_neighbors, dim=1, largest=False)
        neighbors_labels = self.train_labels[indices]

        # One-hot encode labels
        n_classes = torch.max(self.train_labels) + 1
        neighbors_one_hot = torch.zeros(
            neighbors_labels.size(0),
            neighbors_labels.size(1),
            n_classes,
            device=self.device
        )
        neighbors_one_hot.scatter_(2, neighbors_labels.unsqueeze(2), 1)

        # Weight by distance
        weights = torch.softmax(-distances[..., None] / self.temperature, dim=1)
        weighted_votes = weights * neighbors_one_hot

        return weighted_votes.sum(1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions.

        Args:
            x: Input features

        Returns:
            Predicted class labels
        """
        probs = self.forward(x)
        return torch.argmax(probs, dim=1)

    def get_loss_and_grad(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss and gradients for adversarial attacks.

        Args:
            x: Input features
            labels: Target labels

        Returns:
            Tuple of (loss, gradients)
        """
        x.requires_grad_(True)
        logits = self.forward(x)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        return loss.detach(), x.grad.detach()

    def evaluate_metrics(self, true_labels, predicted_labels, num_classes):
        """Evaluate model performance metrics."""
        metrics = {}
        metrics['accuracy'] = accuracy_score(true_labels, predicted_labels)
        metrics['precision'] = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
        metrics['recall'] = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
        metrics['f1'] = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
        metrics['confusion_matrix'] = confusion_matrix(true_labels, predicted_labels).tolist()

        # Add class-specific metrics
        metrics['precision_per_class'] = precision_score(true_labels, predicted_labels, average=None, zero_division=0).tolist()
        metrics['recall_per_class'] = recall_score(true_labels, predicted_labels, average=None, zero_division=0).tolist()
        metrics['f1_per_class'] = f1_score(true_labels, predicted_labels, average=None, zero_division=0).tolist()

        # Calculate class-specific accuracies
        cm = metrics['confusion_matrix']
        class_accuracies = []
        for i in range(num_classes):
            if sum(cm[i]) > 0:
                class_accuracies.append(cm[i][i] / sum(cm[i]))
            else:
                class_accuracies.append(0.0)
        metrics['class_accuracies'] = class_accuracies

        # Add attack-specific metrics
        if hasattr(self, 'attack_params'):
            metrics.update({
                'attack_type': self.attack_params.get('type', 'none'),
                'target_class': self.attack_params.get('target_class', None),
                'source_class': self.attack_params.get('source_class', None),
                'poison_rate': self.attack_params.get('poison_rate', 0.0),
                'num_poisoned_samples': self.attack_params.get('num_poisoned', 0),
                'poisoned_classes': self.attack_params.get('poisoned_classes', [])
            })

        return metrics


class ClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Base wrapper for classifiers."""
    
    def __init__(self, model_class, model_name, **kwargs):
        """Initialize the wrapper."""
        self.model = model_class(**kwargs)
        self.model_name = model_name
    
    def fit(self, X, y):
        """Fit the model."""
        try:
            self.model.fit(X, y)
            return self
        except Exception as e:
            logging.error(f"Error in {self.model_name} training: {str(e)}")
            raise
    
    def predict(self, X):
        """Predict using the model."""
        return self.model.predict(X)


class LogisticRegressionWrapper(ClassifierWrapper):
    """Wrapper for LogisticRegression with logging."""
    
    def __init__(self, **kwargs):
        """Initialize the wrapper."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        # Set reasonable defaults for high-dimensional data
        lr_params = {
            'max_iter': 100,
            'solver': 'saga',
            'C': 0.1,
            'class_weight': 'balanced',
            'tol': 0.01
        }
        lr_params.update(kwargs)
        
        # Create pipeline with PCA
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=100)),  # Reduce to 100 components
            ('classifier', LogisticRegression(**lr_params))
        ])
        self.model_name = 'LogisticRegression'
    
    def fit(self, X, y):
        """Fit the model with logging."""
        import time
        start_time = time.time()
        logging.info(f"Starting LogisticRegression pipeline on {X.shape} data...")
        try:
            self.model.fit(X, y)
            train_time = time.time() - start_time
            n_iter = self.model.named_steps['classifier'].n_iter_
            explained_var = self.model.named_steps['pca'].explained_variance_ratio_.sum()
            logging.info(f"LogisticRegression pipeline completed in {train_time:.2f}s ({n_iter} iterations)")
            logging.info(f"PCA explained variance ratio: {explained_var:.4f}")
            return self
        except Exception as e:
            logging.error(f"Error in LogisticRegression pipeline: {str(e)}")
            raise
            
    def predict(self, X):
        """Predict using the model."""
        return self.model.predict(X)


class SVMWrapper(ClassifierWrapper):
    """Wrapper for SVM with logging."""
    
    def __init__(self, **kwargs):
        """Initialize the wrapper."""
        from sklearn.svm import SVC
        super().__init__(SVC, 'SVM', **kwargs)


class RandomForestWrapper(ClassifierWrapper):
    """Wrapper for RandomForest with logging."""
    
    def __init__(self, **kwargs):
        """Initialize the wrapper."""
        from sklearn.ensemble import RandomForestClassifier
        super().__init__(RandomForestClassifier, 'RandomForest', **kwargs)


class KNeighborsWrapper(ClassifierWrapper):
    """Wrapper for KNeighbors with logging."""
    
    def __init__(self, **kwargs):
        """Initialize the wrapper."""
        from sklearn.neighbors import KNeighborsClassifier
        super().__init__(KNeighborsClassifier, 'KNeighbors', **kwargs)


class DecisionTreeWrapper(ClassifierWrapper):
    """Wrapper for DecisionTree with logging."""
    
    def __init__(self, **kwargs):
        """Initialize the wrapper."""
        from sklearn.tree import DecisionTreeClassifier
        super().__init__(DecisionTreeClassifier, 'DecisionTree', **kwargs)


class ModelFactory:
    """Factory for creating models."""
    
    @staticmethod
    def create_model(model_name: str, **kwargs) -> BaseEstimator:
        """Create model based on name.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Additional arguments for model creation
            
        Returns:
            Created model instance
        """
        models = {
            'SVM': SVMWrapper,
            'RF': RandomForestWrapper,
            'DT': DecisionTreeWrapper,
            'KNN': KNeighborsWrapper,
            'LogisticRegression': LogisticRegressionWrapper,
            'KNeighborsClassifier': KNeighborsWrapper,
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
            
        return models[model_name](**kwargs)

    @staticmethod
    def get_classifier(model_name: str, **kwargs):
        """Get a classifier based on name.
        
        Args:
            model_name: Name of the classifier to create
            **kwargs: Additional arguments for classifier creation
            
        Returns:
            Created classifier instance
        """
        model_map = {
            'LogisticRegression': LogisticRegressionWrapper,
            'SVM': SVMWrapper,
            'RandomForest': RandomForestWrapper,
            'KNeighbors': KNeighborsWrapper,
            'DecisionTree': DecisionTreeWrapper
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")
            
        return model_map[model_name](**kwargs)
