"""Confidence calibration system for DynaDetect v2."""

from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from .monitoring import monitor
import logging
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

@dataclass
class CalibrationConfig:
    """Configuration for confidence calibration."""
    method: str = "temperature"  # Options: "temperature", "isotonic", "ensemble"
    n_bins: int = 10
    cv_folds: int = 5
    max_iter: int = 100
    tolerance: float = 1e-4

class TemperatureScaling(nn.Module):
    """Temperature scaling for confidence calibration."""
    
    def __init__(self):
        """Initialize temperature scaling."""
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature
        
    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100, lr: float = 0.01):
        """Fit temperature scaling using validation data."""
        self.train()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def eval():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss
            
        optimizer.step(eval)
        self.eval()

class ConfidenceCalibration:
    """Confidence calibration for improving clean data performance."""
    
    def __init__(self, config: CalibrationConfig):
        """Initialize calibration system.
        
        Args:
            config: Calibration configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.calibrators: Dict[str, Any] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        poison_mask: Optional[torch.Tensor] = None
    ) -> None:
        """Fit calibration model.
        
        Args:
            logits: Raw model outputs
            labels: True labels
            poison_mask: Optional mask indicating poisoned samples
        """
        with monitor.measure_time("calibration_fit"):
            if self.config.method == "temperature":
                self._fit_temperature(logits, labels, poison_mask)
            elif self.config.method == "isotonic":
                self._fit_isotonic(logits, labels, poison_mask)
            elif self.config.method == "ensemble":
                self._fit_ensemble(logits, labels, poison_mask)
            else:
                raise ValueError(f"Unknown calibration method: {self.config.method}")
                
    def _fit_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        poison_mask: Optional[torch.Tensor]
    ) -> None:
        """Fit temperature scaling."""
        # Use clean data only if poison mask is provided
        if poison_mask is not None:
            clean_mask = ~poison_mask
            logits = logits[clean_mask]
            labels = labels[clean_mask]
            
        # Split data for cross-validation
        skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True)
        temperatures = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(logits, labels)):
            temp_scaler = TemperatureScaling().to(self.device)
            val_logits = logits[val_idx].to(self.device)
            val_labels = labels[val_idx].to(self.device)
            
            temp_scaler.fit(
                val_logits,
                val_labels,
                max_iter=self.config.max_iter
            )
            
            temperatures.append(temp_scaler.temperature.item())
            
        # Use median temperature
        final_temperature = np.median(temperatures)
        self.calibrators['temperature'] = TemperatureScaling().to(self.device)
        self.calibrators['temperature'].temperature.data = torch.tensor([final_temperature])
        
    def _fit_isotonic(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        poison_mask: Optional[torch.Tensor]
    ) -> None:
        """Fit isotonic regression per class."""
        probs = torch.softmax(logits, dim=1)
        n_classes = probs.size(1)
        
        if poison_mask is not None:
            clean_mask = ~poison_mask
            probs = probs[clean_mask]
            labels = labels[clean_mask]
            
        self.calibrators['isotonic'] = []
        
        for c in range(n_classes):
            class_probs = probs[:, c].cpu().numpy()
            class_labels = (labels == c).cpu().numpy().astype(float)
            
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(class_probs, class_labels)
            self.calibrators['isotonic'].append(ir)
            
    def _fit_ensemble(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        poison_mask: Optional[torch.Tensor]
    ) -> None:
        """Fit both temperature and isotonic calibration."""
        self._fit_temperature(logits, labels, poison_mask)
        self._fit_isotonic(logits, labels, poison_mask)
        
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply calibration to model outputs.
        
        Args:
            logits: Raw model outputs
            
        Returns:
            Calibrated probabilities
        """
        with monitor.measure_time("calibration_predict"):
            if self.config.method == "temperature":
                return self._calibrate_temperature(logits)
            elif self.config.method == "isotonic":
                return self._calibrate_isotonic(logits)
            elif self.config.method == "ensemble":
                return self._calibrate_ensemble(logits)
            else:
                raise ValueError(f"Unknown calibration method: {self.config.method}")
                
    def _calibrate_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling."""
        if 'temperature' not in self.calibrators:
            return torch.softmax(logits, dim=1)
            
        with torch.no_grad():
            calibrated_logits = self.calibrators['temperature'](logits)
            return torch.softmax(calibrated_logits, dim=1)
            
    def _calibrate_isotonic(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply isotonic regression calibration."""
        if 'isotonic' not in self.calibrators:
            return torch.softmax(logits, dim=1)
            
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        calibrated_probs = np.zeros_like(probs)
        
        for c, calibrator in enumerate(self.calibrators['isotonic']):
            calibrated_probs[:, c] = calibrator.predict(probs[:, c])
            
        # Normalize to ensure valid probabilities
        calibrated_probs /= calibrated_probs.sum(axis=1, keepdims=True)
        return torch.from_numpy(calibrated_probs).to(self.device)
        
    def _calibrate_ensemble(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply ensemble of calibration methods."""
        temp_probs = self._calibrate_temperature(logits)
        iso_probs = self._calibrate_isotonic(logits)
        return (temp_probs + iso_probs) / 2
        
    def get_calibration_metrics(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        poison_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute calibration metrics.
        
        Args:
            probs: Predicted probabilities
            labels: True labels
            poison_mask: Optional mask indicating poisoned samples
            
        Returns:
            Dictionary of calibration metrics
        """
        if poison_mask is not None:
            clean_mask = ~poison_mask
            probs = probs[clean_mask]
            labels = labels[clean_mask]
            
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = (predictions == labels).float()
        
        # Expected Calibration Error
        confidences = confidences.cpu().numpy()
        accuracies = accuracies.cpu().numpy()
        
        bins = np.linspace(0, 1, self.config.n_bins + 1)
        bin_indices = np.digitize(confidences, bins[1:-1])
        
        ece = 0.0
        for bin_idx in range(self.config.n_bins):
            mask = bin_indices == bin_idx
            if mask.any():
                bin_conf = confidences[mask].mean()
                bin_acc = accuracies[mask].mean()
                bin_size = mask.mean()
                ece += bin_size * abs(bin_acc - bin_conf)
                
        return {
            'ece': float(ece),
            'avg_confidence': float(confidences.mean()),
            'avg_accuracy': float(accuracies.mean())
        } 