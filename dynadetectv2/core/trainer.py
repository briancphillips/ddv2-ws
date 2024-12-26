"""Training implementation for DynaDetect v2."""

from typing import Tuple, Optional
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import IsolationForest
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DynaDetectTrainer:
    """DynaDetect training implementation with robust feature selection and anomaly detection."""
    
    def __init__(self, n_components: int = 100, contamination: float = 0.1):
        """Initialize DynaDetect trainer.
        
        Args:
            n_components: Number of components for dynamic feature selection
            contamination: Expected proportion of outliers in the dataset
        """
        self.n_components = n_components
        self.contamination = contamination
        self.feature_selector: Optional[SelectKBest] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.device = device
        logging.info(f"Initialized DynaDetect trainer on {device}")
    
    def fit_transform(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the DynaDetect components and transform the features.
        
        1. Feature Selection:
           - Use Mutual Information to select most relevant features
           
        2. Robust Training:
           - Scale features robustly
           - Detect and handle outliers
           - Apply sample weights based on confidence scores
           
        Args:
            features: Input features
            labels: Target labels
            
        Returns:
            Tuple of (transformed features, sample weights)
        """
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Select features
        self.feature_selector = SelectKBest(
            score_func=mutual_info_classif,
            k=min(self.n_components, features.shape[1])
        )
        selected_features = self.feature_selector.fit_transform(scaled_features, labels)
        
        # Detect anomalies
        self.anomaly_detector = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        anomaly_scores = self.anomaly_detector.fit_predict(selected_features)
        
        # Convert anomaly scores to sample weights
        # 1 for inliers, reduced weight for outliers
        sample_weights = np.ones(len(anomaly_scores))
        sample_weights[anomaly_scores == -1] = 0.5
        
        return selected_features, sample_weights
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform new features using fitted components.
        
        Args:
            features: Input features
            
        Returns:
            Transformed features
        """
        if self.feature_selector is None:
            raise RuntimeError("Trainer must be fitted before transforming features")
            
        scaled_features = self.scaler.transform(features)
        return self.feature_selector.transform(scaled_features)
    
    def predict_anomaly_scores(self, features: np.ndarray) -> np.ndarray:
        """Predict anomaly scores for new features.
        
        Args:
            features: Input features
            
        Returns:
            Array of anomaly scores (-1 for outliers, 1 for inliers)
        """
        if self.anomaly_detector is None:
            raise RuntimeError("Trainer must be fitted before predicting anomalies")
            
        transformed_features = self.transform(features)
        return self.anomaly_detector.predict(transformed_features)
