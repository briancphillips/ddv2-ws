"""Advanced adaptive features for DynaDetect.

This module implements adaptive techniques for improving model performance:
1. Adaptive neighborhood sizing based on local density
2. Dynamic feature weighting based on importance
3. Uncertainty quantification
"""

import torch
import numpy as np
from typing import Tuple, Optional
from sklearn.neighbors import KernelDensity
from dynadetectv2.core.experimental.monitoring import PerformanceMonitor

class AdaptiveNeighborhood:
    """Adaptive neighborhood sizing based on local density estimation."""
    
    def __init__(
        self,
        base_k: int = 5,
        min_k: int = 3,
        max_k: int = 15,
        bandwidth: float = 0.1
    ):
        """Initialize adaptive neighborhood.
        
        Args:
            base_k: Base number of neighbors
            min_k: Minimum number of neighbors
            max_k: Maximum number of neighbors
            bandwidth: Bandwidth for kernel density estimation
        """
        self.base_k = base_k
        self.min_k = min_k
        self.max_k = max_k
        self.bandwidth = bandwidth
        self.kde = None
        
    def fit(self, X: torch.Tensor) -> None:
        """Fit density estimator on training data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
        """
        # Convert to CPU numpy for sklearn
        X_np = X.cpu().numpy()
        self.kde = KernelDensity(bandwidth=self.bandwidth)
        self.kde.fit(X_np)
        
    def get_adaptive_k(self, X: torch.Tensor) -> torch.Tensor:
        """Get adaptive k for each query point based on local density.
        
        Args:
            X: Query points of shape (n_samples, n_features)
            
        Returns:
            Tensor of adaptive k values for each point
        """
        X_np = X.cpu().numpy()
        log_density = self.kde.score_samples(X_np)
        density = np.exp(log_density)
        
        # Scale k based on relative density
        density_norm = (density - density.min()) / (density.max() - density.min())
        k_values = self.base_k + (density_norm * (self.max_k - self.base_k)).astype(int)
        k_values = np.clip(k_values, self.min_k, self.max_k)
        
        return torch.from_numpy(k_values).to(X.device)

class DynamicFeatureWeighting:
    """Dynamic feature weighting based on importance scores."""
    
    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        regularization: float = 0.1
    ):
        """Initialize feature weighting.
        
        Args:
            n_features: Number of input features
            learning_rate: Learning rate for weight updates
            regularization: L2 regularization strength
        """
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.weights = None
        
    def initialize_weights(self, device: torch.device) -> None:
        """Initialize feature weights."""
        self.weights = torch.ones(self.n_features, device=device)
        self.weights.requires_grad = True
        
    def update_weights(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        distances: torch.Tensor
    ) -> None:
        """Update feature weights based on prediction errors.
        
        Args:
            X: Input features
            y: True labels
            distances: Pairwise distances
        """
        if self.weights is None:
            self.initialize_weights(X.device)
            
        # Compute weighted features
        weighted_features = X * self.weights.unsqueeze(0)
        
        # Compute pairwise distances with weighted features
        weighted_dist = torch.cdist(weighted_features, weighted_features)
        
        # Simple gradient update based on prediction errors
        loss = torch.mean(weighted_dist)
        loss.backward()
        
        with torch.no_grad():
            self.weights -= self.learning_rate * self.weights.grad
            self.weights += self.regularization * (1 - self.weights)
            self.weights.clamp_(min=0.1, max=2.0)
            self.weights.grad.zero_()
            
    def get_weighted_features(self, X: torch.Tensor) -> torch.Tensor:
        """Apply feature weights to input data.
        
        Args:
            X: Input features
            
        Returns:
            Weighted features
        """
        if self.weights is None:
            self.initialize_weights(X.device)
        return X * self.weights.unsqueeze(0)

class UncertaintyQuantification:
    """Uncertainty quantification for predictions."""
    
    def __init__(self, n_classes: int):
        """Initialize uncertainty quantification.
        
        Args:
            n_classes: Number of classes
        """
        self.n_classes = n_classes
        
    def compute_uncertainty(
        self,
        probabilities: torch.Tensor,
        distances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute aleatoric and epistemic uncertainty.
        
        Args:
            probabilities: Predicted class probabilities
            distances: Distances to k-nearest neighbors
            
        Returns:
            Tuple of (aleatoric_uncertainty, epistemic_uncertainty)
        """
        # Aleatoric uncertainty - prediction entropy
        aleatoric = -torch.sum(
            probabilities * torch.log(probabilities + 1e-10),
            dim=1
        )
        
        # Epistemic uncertainty - distance-based uncertainty
        epistemic = torch.mean(distances, dim=1)
        
        return aleatoric, epistemic 