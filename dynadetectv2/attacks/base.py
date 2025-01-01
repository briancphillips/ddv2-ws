"""Base classes for attack implementations."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Any
import torch
import numpy as np


class BaseAttack(ABC):
    """Base class for all attacks."""
    
    def __init__(self, poison_rate: float = 0.01):
        """Initialize attack.
        
        Args:
            poison_rate: Fraction of samples to poison
        """
        self.poison_rate = poison_rate
        
    @abstractmethod
    def attack(self, features: torch.Tensor, labels: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Apply attack to the data.
        
        Args:
            features: Input features
            labels: Input labels
            **kwargs: Additional attack parameters
            
        Returns:
            Tuple of (poisoned features, poisoned labels, indices of poisoned samples)
        """
        pass
    
    def select_poison_subset(self, num_samples: int) -> List[int]:
        """Select indices of samples to poison.
        
        Args:
            num_samples: Total number of samples
            
        Returns:
            List of indices to poison
        """
        num_poison = int(num_samples * self.poison_rate)
        indices = list(range(num_samples))
        return np.random.choice(indices, num_poison, replace=False).tolist()


class GradientBasedAttack(BaseAttack):
    """Base class for gradient-based attacks."""
    
    def __init__(
        self,
        poison_rate: float = 0.01,
        eps: float = 0.1,
        alpha: float = 0.01,
        iters: int = 40
    ):
        """Initialize gradient-based attack.
        
        Args:
            poison_rate: Fraction of samples to poison
            eps: Maximum perturbation
            alpha: Step size
            iters: Number of iterations
        """
        super().__init__(poison_rate)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        
    @abstractmethod
    def compute_perturbation(
        self,
        model: torch.nn.Module,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute perturbation for features.
        
        Args:
            model: Model to attack
            features: Input features
            labels: Target labels
            
        Returns:
            Perturbation to apply to features
        """
        pass
