"""Calibrated DDKNN model implementation."""

import torch
import torch.nn as nn
from ..models import DDKNN
from .monitoring import monitor
import logging

class CalibratedDDKNN(DDKNN):
    """DDKNN with confidence calibration capabilities."""

    def __init__(self, n_neighbors: int = 5, temperature: float = 1.0, calibration_method: str = "temperature"):
        """Initialize the calibrated DDKNN classifier.

        Args:
            n_neighbors: Number of neighbors to use
            temperature: Initial temperature parameter for soft voting
            calibration_method: Calibration method to use ("temperature", "isotonic", or "ensemble")
        """
        super(CalibratedDDKNN, self).__init__(n_neighbors=n_neighbors, temperature=temperature)
        self.calibration_method = calibration_method
        self.calibration_temp = nn.Parameter(torch.ones(1))
        self.logger = logging.getLogger(__name__)
        
    def _get_base_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get base logits from DDKNN model.
        
        This method handles the dimension mismatch in the base model's forward pass.
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
            self.n_neighbors,
            n_classes,
            device=self.device
        )
        neighbors_one_hot.scatter_(2, neighbors_labels.unsqueeze(2), 1)

        # Weight by distance
        weights = torch.softmax(-distances[:, :self.n_neighbors].unsqueeze(-1) / self.temperature, dim=1)
        weighted_votes = weights * neighbors_one_hot

        return weighted_votes.sum(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute calibrated DDKNN predictions.

        Args:
            x: Input features

        Returns:
            Calibrated class probabilities
        """
        with monitor.measure_time("forward_pass"):
            # Get base predictions using our modified method
            base_logits = self._get_base_logits(x)
            
            # Apply calibration
            if self.calibration_method == "temperature":
                calibrated_logits = base_logits / self.calibration_temp
            else:
                calibrated_logits = base_logits
                
            return torch.softmax(calibrated_logits, dim=1)
            
    def fit_calibration(self, val_features: torch.Tensor, val_labels: torch.Tensor, 
                       val_poison_mask: torch.Tensor = None) -> None:
        """Fit calibration parameters on validation data.

        Args:
            val_features: Validation features
            val_labels: Validation labels
            val_poison_mask: Optional mask indicating poisoned samples
        """
        with monitor.measure_time("calibration_fit"):
            if self.calibration_method == "temperature":
                self._fit_temperature(val_features, val_labels, val_poison_mask)
            # Add other calibration methods here
                
    def _fit_temperature(self, val_features: torch.Tensor, val_labels: torch.Tensor,
                        val_poison_mask: torch.Tensor = None) -> None:
        """Fit temperature scaling parameter.

        Args:
            val_features: Validation features
            val_labels: Validation labels
            val_poison_mask: Optional mask indicating poisoned samples
        """
        self.calibration_temp.requires_grad_(True)
        optimizer = torch.optim.LBFGS([self.calibration_temp], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()
        
        def eval():
            optimizer.zero_grad()
            with torch.no_grad():
                logits = self._get_base_logits(val_features)
            scaled_logits = logits / self.calibration_temp
            loss = criterion(scaled_logits, val_labels)
            loss.backward()
            return loss
            
        optimizer.step(eval)
        self.calibration_temp.requires_grad_(False)
        
        self.logger.info(f"Fitted temperature scaling parameter: {self.calibration_temp.item():.4f}")
        
    def get_calibration_metrics(self, probs: torch.Tensor, labels: torch.Tensor,
                              poison_mask: torch.Tensor = None) -> dict:
        """Compute calibration metrics.

        Args:
            probs: Predicted probabilities
            labels: True labels
            poison_mask: Optional mask indicating poisoned samples

        Returns:
            Dictionary of calibration metrics
        """
        with monitor.measure_time("compute_metrics"):
            metrics = {}
            
            # Expected Calibration Error (ECE)
            n_bins = 10
            bin_boundaries = torch.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            confidences, predictions = torch.max(probs, dim=1)
            accuracies = predictions.eq(labels)
            
            ece = torch.zeros(1, device=self.device)
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = accuracies[in_bin].float().mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    
            metrics['ece'] = ece.item()
            metrics['accuracy'] = accuracies.float().mean().item()
            metrics['avg_confidence'] = confidences.mean().item()
            
            if poison_mask is not None:
                clean_ece = self._compute_ece(probs[~poison_mask], labels[~poison_mask])
                poison_ece = self._compute_ece(probs[poison_mask], labels[poison_mask])
                metrics['clean_ece'] = clean_ece
                metrics['poison_ece'] = poison_ece
                
            return metrics
            
    def _compute_ece(self, probs: torch.Tensor, labels: torch.Tensor) -> float:
        """Helper method to compute ECE for a subset of data."""
        n_bins = 10
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)
        
        ece = torch.zeros(1, device=self.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece.item() 