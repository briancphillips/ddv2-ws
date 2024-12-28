"""Model implementations for DynaDetect v2."""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging
import time
import psutil
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseModel:
    """Base class for all models."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError
        
    def predict(self, X):
        raise NotImplementedError
        
    def predict_proba(self, X):
        raise NotImplementedError


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


class ClassifierWrapper:
    """Base wrapper for scikit-learn classifiers."""
    
    def __init__(self, model_class, name: str, **kwargs):
        """Initialize the wrapper."""
        logging.info(f"Initializing {name} wrapper with params: {kwargs}")
        self.model = model_class(**kwargs)
        self.name = name
        
    def fit(self, X, y, sample_weight=None):
        """Fit the model."""
        logging.info(f"Fitting {self.name} model with data shape: {X.shape}")
        start_time = time.time()
        if sample_weight is not None and hasattr(self.model, 'fit') and 'sample_weight' in self.model.fit.__code__.co_varnames:
            result = self.model.fit(X, y, sample_weight=sample_weight)
        else:
            result = self.model.fit(X, y)
        logging.info(f"{self.name} fit completed in {time.time() - start_time:.2f}s")
        return result
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)


class LogisticRegressionWrapper(BaseModel):
    """GPU-accelerated LogisticRegression using PyTorch."""
    
    def __init__(self, learning_rate=0.001, max_iter=1000, batch_size=128, weight_decay=0.01,
                 validation_fraction=0.1, early_stopping=True, n_iter_no_change=5,
                 tol=0.001, verbose=True):
        super().__init__()
        self.learning_rate = learning_rate  # Reduced learning rate
        self.max_iter = max_iter  # Increased iterations
        self.batch_size = batch_size
        self.weight_decay = weight_decay  # Increased regularization
        self.validation_fraction = validation_fraction
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change  # More patience
        self.tol = tol  # Tighter tolerance
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")

    def _init_model(self, input_dim, n_classes):
        """Initialize the PyTorch model."""
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.model = nn.Linear(input_dim, n_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), 
                                  lr=self.learning_rate, 
                                  weight_decay=self.weight_decay)

    def fit(self, X, y, sample_weight=None):
        """Fit the model using PyTorch."""
        start_time = time.time()
        self.logger.info(f"Memory usage at start_fit: CPU {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB, GPU {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
        self.logger.info(f"Input data shape: {X.shape}, labels shape: {y.shape}")
        
        try:
            # Convert data to PyTorch tensors
            X = torch.FloatTensor(X)
            y = torch.LongTensor(y)
            if sample_weight is not None:
                sample_weight = torch.FloatTensor(sample_weight)

            # Initialize model
            n_classes = len(torch.unique(y))
            self.logger.info(f"Number of unique classes: {n_classes}")
            self._init_model(X.shape[1], n_classes)

            # Scale features
            start_scaling = time.time()
            self.scaler = StandardScaler()
            X = torch.FloatTensor(self.scaler.fit_transform(X))
            self.logger.info("Fitted scaler and transformed data")
            self.logger.info(f"Scaling time: {time.time() - start_scaling:.2f}s")

            # Create data loaders
            dataset = TensorDataset(X, y)
            train_size = int((1 - self.validation_fraction) * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

            # Training loop
            best_val_loss = float('inf')
            no_improvement_count = 0
            
            for epoch in range(self.max_iter):
                self.model.train()
                total_loss = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                
                # Validation
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        val_loss += self.criterion(outputs, batch_y).item()
                
                avg_train_loss = total_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                if self.verbose:
                    self.logger.info(f"Epoch {epoch + 1}/{self.max_iter}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                # Early stopping
                if self.early_stopping:
                    if avg_val_loss < best_val_loss - self.tol:
                        best_val_loss = avg_val_loss
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        
                    if no_improvement_count >= self.n_iter_no_change:
                        if self.verbose:
                            self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break
            
            self.logger.info(f"Training completed in {time.time() - start_time:.2f}s")
            self.logger.info(f"Final memory usage: CPU {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB, GPU {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
            return self
            
        except Exception as e:
            self.logger.error(f"Error during model fitting: {str(e)}")
            raise

    def predict(self, X):
        """Predict using the model."""
        try:
            X = torch.FloatTensor(self.scaler.transform(X)).to(self.device)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X)
                predictions = torch.argmax(outputs, dim=1)
            return predictions.cpu().numpy()
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_proba(self, X):
        """Predict class probabilities."""
        try:
            X = torch.FloatTensor(self.scaler.transform(X)).to(self.device)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X)
                probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
        except Exception as e:
            self.logger.error(f"Error during probability prediction: {str(e)}")
            raise


class SVMWrapper(BaseModel):
    def __init__(self, learning_rate=0.001, max_iter=500, batch_size=128, weight_decay=0.01,
                 margin=1.0, validation_fraction=0.1, early_stopping=True, n_iter_no_change=5,
                 tol=0.001, verbose=True):
        super().__init__()
        self.learning_rate = learning_rate  # Reduced learning rate
        self.max_iter = max_iter  # Increased iterations
        self.batch_size = batch_size
        self.weight_decay = weight_decay  # Increased regularization
        self.margin = margin
        self.validation_fraction = validation_fraction
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change  # More patience
        self.tol = tol  # Tighter tolerance
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")

    def _init_model(self, input_dim, n_classes):
        """Initialize the PyTorch model."""
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.model = nn.Linear(input_dim, n_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                  lr=self.learning_rate, 
                                  weight_decay=self.weight_decay)

    def _hinge_loss(self, outputs, targets):
        """Compute multi-class hinge loss."""
        batch_size = outputs.size(0)
        correct_scores = outputs[torch.arange(batch_size), targets].view(-1, 1)
        margin_diff = outputs - correct_scores + self.margin
        margin_diff[torch.arange(batch_size), targets] = 0
        loss = torch.sum(torch.clamp(margin_diff, min=0)) / batch_size
        return loss

    def fit(self, X, y, sample_weight=None):
        """Fit the SVM model using PyTorch."""
        try:
            start_time = time.time()
            self.logger.info(f"Memory usage at start_fit: CPU {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB, GPU {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
            self.logger.info(f"Input data shape: {X.shape}, labels shape: {y.shape}")
            
            # Convert data to PyTorch tensors
            X = torch.FloatTensor(X)
            y = torch.LongTensor(y)
            if sample_weight is not None:
                sample_weight = torch.FloatTensor(sample_weight)

            # Initialize model
            n_classes = len(torch.unique(y))
            self.logger.info(f"Number of unique classes: {n_classes}")
            self._init_model(X.shape[1], n_classes)

            # Scale features
            start_scaling = time.time()
            self.scaler = StandardScaler()
            X = torch.FloatTensor(self.scaler.fit_transform(X))
            self.logger.info("Fitted scaler and transformed data")
            self.logger.info(f"Scaling time: {time.time() - start_scaling:.2f}s")

            # Create data loaders
            dataset = TensorDataset(X, y)
            train_size = int((1 - self.validation_fraction) * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

            # Training loop
            best_val_loss = float('inf')
            no_improvement_count = 0
            
            for epoch in range(self.max_iter):
                self.model.train()
                total_loss = 0
                n_batches = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self._hinge_loss(outputs, batch_y)
                    
                    if sample_weight is not None:
                        batch_weights = sample_weight[train_dataset.indices][n_batches*self.batch_size:(n_batches+1)*self.batch_size]
                        loss = loss * batch_weights.mean()
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    n_batches += 1
                
                avg_train_loss = total_loss / n_batches
                
                # Validation
                self.model.eval()
                val_loss = 0
                n_val_batches = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = self._hinge_loss(outputs, batch_y)
                        val_loss += loss.item()
                        n_val_batches += 1
                
                avg_val_loss = val_loss / n_val_batches
                
                if self.verbose:
                    self.logger.info(f"Epoch {epoch+1}/{self.max_iter}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                # Early stopping
                if self.early_stopping:
                    if avg_val_loss < best_val_loss - self.tol:
                        best_val_loss = avg_val_loss
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    if no_improvement_count >= self.n_iter_no_change:
                        self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
            
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f}s")
            self.logger.info(f"Final memory usage: CPU {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB, GPU {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise

    def predict(self, X):
        """Predict using the SVM model."""
        try:
            X = torch.FloatTensor(self.scaler.transform(X)).to(self.device)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X)
                predictions = torch.argmax(outputs, dim=1)
            return predictions.cpu().numpy()
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_proba(self, X):
        """Predict class probabilities using the SVM model."""
        try:
            X = torch.FloatTensor(self.scaler.transform(X)).to(self.device)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X)
                probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
        except Exception as e:
            self.logger.error(f"Error during probability prediction: {str(e)}")
            raise


class DecisionTreeModule(nn.Module):
    """GPU-accelerated Decision Tree using PyTorch."""
    def __init__(self, input_dim, n_classes, max_depth=10):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.max_depth = max_depth
        
        # Initialize tree parameters with smaller depth for efficiency
        n_nodes = 2**max_depth - 1
        # Use float tensors for parameters that require gradients
        self.split_features = nn.Parameter(torch.rand(n_nodes) * input_dim)
        self.split_thresholds = nn.Parameter(torch.randn(n_nodes) / np.sqrt(input_dim))  # Better initialization
        self.leaf_probabilities = nn.Parameter(torch.zeros(2**max_depth, n_classes))
        nn.init.xavier_uniform_(self.leaf_probabilities, gain=1/np.sqrt(n_classes))  # Better initialization

    def forward(self, x):
        batch_size = x.size(0)
        node_indices = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        # Pre-compute feature indices for all nodes
        node_features = torch.floor(self.split_features).long()
        node_features = torch.clamp(node_features, 0, self.input_dim - 1)
        
        for depth in range(self.max_depth):
            # Get current node features and thresholds for the entire batch
            current_features = node_features[node_indices]
            current_thresholds = self.split_thresholds[node_indices]
            
            # Compute split decisions
            decisions = x[torch.arange(batch_size), current_features] > current_thresholds
            
            # Update node indices
            node_indices = node_indices * 2 + 1 + decisions.long()
        
        # Get leaf probabilities
        leaf_indices = node_indices - (2**self.max_depth - 1)
        probabilities = self.leaf_probabilities[leaf_indices]
        return torch.softmax(probabilities, dim=1)


class RandomForestWrapper(BaseModel):
    """GPU-accelerated Random Forest using PyTorch."""
    
    def __init__(self, n_estimators=50, max_depth=8, learning_rate=0.001, 
                 batch_size=512, n_epochs=10, weight_decay=0.01,
                 validation_fraction=0.05, early_stopping=True,
                 n_iter_no_change=3, tol=0.005, verbose=True):
        super().__init__()
        self.n_estimators = n_estimators  # Reduced number of trees
        self.max_depth = max_depth  # Reduced depth
        self.learning_rate = learning_rate
        self.batch_size = batch_size  # Increased batch size
        self.n_epochs = n_epochs  # Reduced epochs
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction  # Reduced validation fraction
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change  # More aggressive early stopping
        self.tol = tol  # Looser tolerance
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
        self.chunk_size = 10  # Process trees in chunks

    def _init_forest(self, input_dim, n_classes):
        """Initialize the forest of decision trees."""
        # Initialize all trees at once
        self.trees = nn.ModuleList([
            DecisionTreeModule(input_dim, n_classes, self.max_depth)
            for _ in range(self.n_estimators)
        ]).to(self.device)
        
        # Use AdamW optimizer for better regularization
        self.optimizer = optim.AdamW(
            self.trees.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def fit(self, X, y, sample_weight=None):
        """Train the random forest using PyTorch."""
        try:
            start_time = time.time()
            self.logger.info(f"Memory usage at start_fit: CPU {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB, GPU {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
            self.logger.info(f"Input data shape: {X.shape}, labels shape: {y.shape}")
            
            # Scale features
            start_scaling = time.time()
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            self.logger.info("Fitted scaler and transformed data")
            self.logger.info(f"Scaling time: {time.time() - start_scaling:.2f}s")

            # Convert to tensors and move to GPU
            X = torch.FloatTensor(X).to(self.device)
            y = torch.LongTensor(y).to(self.device)
            if sample_weight is not None:
                sample_weight = torch.FloatTensor(sample_weight).to(self.device)

            # Initialize forest
            n_classes = len(torch.unique(y))
            self.logger.info(f"Number of unique classes: {n_classes}")
            self._init_forest(X.shape[1], n_classes)

            # Create data loaders
            dataset = TensorDataset(X, y)
            train_size = int((1 - self.validation_fraction) * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, pin_memory=False)

            # Training loop
            criterion = nn.CrossEntropyLoss()
            best_val_loss = float('inf')
            no_improvement_count = 0
            
            for epoch in range(self.n_epochs):
                # Training
                for tree in self.trees:
                    tree.train()
                total_loss = 0
                n_batches = 0
                
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    
                    # Process trees in chunks to reduce memory usage
                    chunk_outputs = []
                    for i in range(0, len(self.trees), self.chunk_size):
                        chunk_trees = self.trees[i:i+self.chunk_size]
                        outputs = torch.stack([tree(batch_X) for tree in chunk_trees])
                        chunk_outputs.append(outputs.mean(dim=0))
                    
                    # Average predictions across all chunks
                    ensemble_output = torch.stack(chunk_outputs).mean(dim=0)
                    
                    # Compute loss
                    loss = criterion(ensemble_output, batch_y)
                    if sample_weight is not None:
                        batch_weights = sample_weight[train_dataset.indices][n_batches*self.batch_size:(n_batches+1)*self.batch_size]
                        loss = loss * batch_weights.mean()
                    
                    # Backward pass and optimization
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.trees.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    n_batches += 1

                avg_train_loss = total_loss / n_batches
                
                # Validation
                for tree in self.trees:
                    tree.eval()
                val_loss = 0
                n_val_batches = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        # Process trees in chunks during validation
                        chunk_outputs = []
                        for i in range(0, len(self.trees), self.chunk_size):
                            chunk_trees = self.trees[i:i+self.chunk_size]
                            outputs = torch.stack([tree(batch_X) for tree in chunk_trees])
                            chunk_outputs.append(outputs.mean(dim=0))
                        
                        ensemble_output = torch.stack(chunk_outputs).mean(dim=0)
                        loss = criterion(ensemble_output, batch_y)
                        val_loss += loss.item()
                        n_val_batches += 1
                
                avg_val_loss = val_loss / n_val_batches
                
                if self.verbose:
                    self.logger.info(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                # Early stopping with looser tolerance
                if self.early_stopping:
                    if avg_val_loss < best_val_loss - self.tol:
                        best_val_loss = avg_val_loss
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    if no_improvement_count >= self.n_iter_no_change:
                        self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break

            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f}s")
            self.logger.info(f"Final memory usage: CPU {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB, GPU {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
        
    def predict(self, X):
        """Predict class labels for samples in X."""
        try:
            # Scale and convert to tensor
            X = torch.FloatTensor(self.scaler.transform(X)).to(self.device)
            
            # Get predictions from all trees in parallel
            with torch.no_grad():
                outputs = torch.stack([tree(X) for tree in self.trees])
                ensemble_output = outputs.mean(dim=0)
                predictions = torch.argmax(ensemble_output, dim=1)
            
            return predictions.cpu().numpy()
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_proba(self, X):
        """Predict class probabilities."""
        try:
            # Scale and convert to tensor
            X = torch.FloatTensor(self.scaler.transform(X)).to(self.device)
            
            # Get probabilities from all trees in parallel
            with torch.no_grad():
                outputs = torch.stack([tree(X) for tree in self.trees])
                ensemble_output = outputs.mean(dim=0)
            
            return ensemble_output.cpu().numpy()
            
        except Exception as e:
            self.logger.error(f"Error during probability prediction: {str(e)}")
            raise


class DecisionTreeWrapper(ClassifierWrapper):
    """Wrapper for DecisionTree with logging."""
    
    def __init__(self, **kwargs):
        """Initialize the wrapper."""
        from sklearn.tree import DecisionTreeClassifier
        super().__init__(DecisionTreeClassifier, 'DecisionTree', **kwargs)


class KNeighborsWrapper(BaseModel):
    """GPU-accelerated KNN using PyTorch."""
    
    def __init__(self, n_neighbors=5, batch_size=128, verbose=True):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")

    def fit(self, X, y, sample_weight=None):
        """Store training data and scale features."""
        try:
            start_time = time.time()
            self.logger.info(f"Memory usage at start_fit: CPU {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB, GPU {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
            self.logger.info(f"Input data shape: {X.shape}, labels shape: {y.shape}")
            
            # Scale features
            start_scaling = time.time()
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            self.logger.info("Fitted scaler and transformed data")
            self.logger.info(f"Scaling time: {time.time() - start_scaling:.2f}s")

            # Store training data as tensors
            self.X_train = torch.FloatTensor(X).to(self.device)
            self.y_train = torch.LongTensor(y).to(self.device)
            self.n_classes = len(torch.unique(self.y_train))
            
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f}s")
            self.logger.info(f"Final memory usage: CPU {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB, GPU {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def _compute_distances(self, X_batch):
        """Compute pairwise distances between batch and training data."""
        # Compute squared L2 norm of each vector
        X_norm = (X_batch ** 2).sum(1).view(-1, 1)
        train_norm = (self.X_train ** 2).sum(1).view(1, -1)
        
        # Compute distances using matrix multiplication
        # dist^2 = ||x||^2 + ||y||^2 - 2<x,y>
        distances = X_norm + train_norm - 2.0 * torch.mm(X_batch, self.X_train.t())
        return torch.sqrt(torch.clamp(distances, min=0.0))

    def _get_probabilities(self, distances, k):
        """Convert distances to probabilities using k-nearest neighbors."""
        # Get indices of k nearest neighbors
        _, indices = torch.topk(distances, k=k, dim=1, largest=False)
        
        # Get labels of k nearest neighbors
        neighbor_labels = self.y_train[indices]
        
        # Convert to one-hot encoding
        one_hot = torch.zeros(indices.shape[0], indices.shape[1], self.n_classes, device=self.device)
        one_hot.scatter_(2, neighbor_labels.unsqueeze(2), 1)
        
        # Average over neighbors to get probabilities
        return one_hot.mean(dim=1)

    def predict(self, X):
        """Predict class labels for samples in X."""
        try:
            # Scale and convert to tensor
            X = torch.FloatTensor(self.scaler.transform(X))
            
            # Process in batches
            predictions = []
            n_samples = X.shape[0]
            
            for i in range(0, n_samples, self.batch_size):
                batch = X[i:i+self.batch_size].to(self.device)
                distances = self._compute_distances(batch)
                probs = self._get_probabilities(distances, self.n_neighbors)
                predictions.append(torch.argmax(probs, dim=1))
            
            # Concatenate results
            predictions = torch.cat(predictions)
            return predictions.cpu().numpy()
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_proba(self, X):
        """Predict probability estimates."""
        try:
            # Scale and convert to tensor
            X = torch.FloatTensor(self.scaler.transform(X))
            
            # Process in batches
            probabilities = []
            n_samples = X.shape[0]
            
            for i in range(0, n_samples, self.batch_size):
                batch = X[i:i+self.batch_size].to(self.device)
                distances = self._compute_distances(batch)
                probs = self._get_probabilities(distances, self.n_neighbors)
                probabilities.append(probs)
            
            # Concatenate results
            probabilities = torch.cat(probabilities)
            return probabilities.cpu().numpy()
            
        except Exception as e:
            self.logger.error(f"Error during probability prediction: {str(e)}")
            raise


class ModelFactory:
    """Factory class for creating model instances."""
    
    @staticmethod
    def create_model(classifier_name: str) -> ClassifierWrapper:
        """Create a model instance based on the classifier name."""
        logging.info(f"Creating model for classifier: {classifier_name}")
        if classifier_name == 'SVM':
            model = SVMWrapper()
            logging.info(f"Created SVMWrapper instance: {model.__class__.__name__}")
            return model
        elif classifier_name == 'LogisticRegression':
            return LogisticRegressionWrapper()
        elif classifier_name == 'RandomForest':
            return RandomForestWrapper()
        elif classifier_name == 'KNeighbors':
            return KNeighborsWrapper()
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")

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
