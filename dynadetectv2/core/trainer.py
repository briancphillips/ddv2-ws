"""Training implementation for DynaDetect v2."""

from typing import Tuple, Optional
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import IsolationForest
import logging
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import LocalOutlierFactor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DynaDetectTrainer:
    """DynaDetect training implementation with robust feature selection and anomaly detection."""
    
    def __init__(self, n_components: int = 50, contamination: float = 0.1):
        """Initialize DynaDetect trainer."""
        self.n_components = min(n_components, 50)  # Limit components for speed
        self.contamination = contamination
        self.feature_selector: Optional[SelectKBest] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.scaler = RobustScaler()
        self.device = device
        logging.info(f"Initialized DynaDetect trainer on {device}")
        
    def fit_transform(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the DynaDetect components and transform the features."""
        start_time = time.time()
        
        # Scale features
        logging.info(f"Starting feature scaling on shape {features.shape}")
        t0 = time.time()
        scaled_features = self.scaler.fit_transform(features)
        logging.info(f"Feature scaling completed in {time.time() - t0:.2f}s")
        
        # Select features
        selected_features = self.select_features(scaled_features, self.n_components)
        
        # Initialize weights with ones
        sample_weights = np.ones(len(features))
        
        # Detect anomalies and adjust weights
        outliers = self.detect_anomalies(selected_features, self.contamination)
        sample_weights[outliers] = 0.5
        
        total_time = time.time() - start_time
        logging.info(f"Total fit_transform time: {total_time:.2f}s")
        return selected_features, sample_weights
        
    def compute_sample_weights(self) -> np.ndarray:
        """Compute sample weights for the dataset."""
        start_time = time.time()
        
        # Check cache first
        cache_key = (id(self.dataset), len(self.dataset))
        if cache_key in self._cache:
            logging.info("Using cached sample weights")
            return self._cache[cache_key]
            
        # Process data in batches
        logging.info(f"Processing dataset of size {len(self.dataset)} in batches")
        batch_size = 128
        features_list = []
        labels_list = []
        
        t0 = time.time()
        dataloader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True  # Enable faster data transfer to GPU
        )
        
        with torch.cuda.amp.autocast():  # Enable automatic mixed precision
            for i, (data, label) in enumerate(dataloader):
                if i == 0:
                    logging.info(f"Processing first batch. Data shape: {data.shape}")
                data = data.to(self.device)
                if isinstance(data, torch.Tensor):
                    features_list.append(data.reshape(data.size(0), -1))
                else:
                    features_list.append(torch.from_numpy(data).reshape(data.size(0), -1).to(self.device))
                labels_list.append(label.to(self.device))
        
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        logging.info(f"Data loading completed in {time.time() - t0:.2f}s. Final feature shape: {features.shape}")
        
        # Transform features and get weights
        transformed_features, weights = self.fit_transform(features.cpu().numpy(), labels.cpu().numpy())
        
        # Ensure weights match the number of samples
        if len(weights) != len(features):
            logging.warning(f"Weights length ({len(weights)}) does not match features length ({len(features)}). Adjusting...")
            # Initialize default weights
            adjusted_weights = np.ones(len(features))
            # If we have some weights, use them as anomaly indicators
            if len(weights) > 0:
                anomaly_threshold = np.percentile(transformed_features.std(axis=1), 90)
                anomaly_scores = transformed_features.std(axis=1)
                adjusted_weights[anomaly_scores > anomaly_threshold] = 0.5
            weights = adjusted_weights
        
        # Cache the results
        self._cache[cache_key] = weights
        
        total_time = time.time() - start_time
        logging.info(f"Total compute_sample_weights time: {total_time:.2f}s")
        return weights
        
    def select_features(self, features: np.ndarray, k: int = 50) -> np.ndarray:
        """Select top k features using GPU-accelerated clustering."""
        logging.info(f"Starting feature selection (k={k})")
        t0 = time.time()
        
        # Convert to torch tensor and move to GPU
        features_tensor = torch.from_numpy(features).float().to(self.device)
        
        # Compute feature importance scores on GPU
        with torch.cuda.amp.autocast():
            # Compute feature variance
            var_scores = torch.var(features_tensor, dim=0)
            
            # Compute feature correlations with target (simplified mutual information)
            n_samples = features_tensor.size(0)
            n_features = features_tensor.size(1)
            
            # Use top-k variances for initial feature selection
            _, top_indices = torch.topk(var_scores, min(k * 2, n_features))
            selected_features = features_tensor[:, top_indices]
            
            # Further reduce using correlation-based selection
            corr_matrix = torch.corrcoef(selected_features.T)
            redundancy_scores = torch.sum(torch.abs(corr_matrix), dim=1)
            _, final_indices = torch.topk(redundancy_scores, k, largest=False)
            
            final_selected = selected_features[:, final_indices]
        
        selected_features = final_selected.cpu().numpy()
        logging.info(f"Feature selection completed in {time.time() - t0:.2f}s. Output shape: {selected_features.shape}")
        
        return selected_features
        
    def detect_anomalies(self, features: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        """Detect anomalies using GPU acceleration."""
        logging.info("Starting anomaly detection")
        t0 = time.time()
        
        # Convert to torch tensor and move to GPU
        features_tensor = torch.from_numpy(features).float().to(self.device)
        
        with torch.cuda.amp.autocast():
            # Compute pairwise distances
            dist_matrix = torch.cdist(features_tensor, features_tensor)
            
            # Compute local outlier scores
            k = 10  # number of neighbors
            knn_dists, _ = torch.topk(dist_matrix, k + 1, largest=False)
            knn_dists = knn_dists[:, 1:]  # exclude self-distance
            
            # Compute local reachability density
            lrd = 1.0 / torch.mean(knn_dists, dim=1)
            
            # Compute LOF scores
            lof_scores = torch.zeros(len(features_tensor), device=self.device)
            for i in range(len(features_tensor)):
                neighbors = torch.argsort(dist_matrix[i])[1:k+1]
                lof_scores[i] = torch.mean(lrd[neighbors]) / lrd[i]
            
            # Determine outliers
            threshold = torch.quantile(lof_scores, 1 - contamination)
            outliers = lof_scores > threshold
        
        outliers_np = outliers.cpu().numpy()
        num_outliers = np.sum(outliers_np)
        
        logging.info(f"Anomaly detection completed in {time.time() - t0:.2f}s")
        logging.info(f"Found {num_outliers} outliers ({num_outliers/len(features)*100:.1f}%)")
        
        return outliers_np
