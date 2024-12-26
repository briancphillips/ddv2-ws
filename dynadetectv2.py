"""
DynaDetect v2: A Framework for Handling Image and Numerical Datasets
with Data Poisoning Detection and Robustness Evaluation
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                           precision_score, recall_score)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest

from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass
import json
import csv
import os
import random
import copy
import datetime

# Data processing
import pandas as pd
from sklearn.datasets import load_diabetes

# ML models and metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier

# Visualization
import matplotlib.pyplot as plt

# Combined Code: Handling Both Image and Numerical Datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Utility Functions

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging():
    """Set up logging configuration."""
    # Don't set up logging here as it's handled by run_full_evaluation.py
    pass

def get_targets(dataset: Dataset) -> np.ndarray:
    """Get targets from dataset in a consistent format."""
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)
    elif isinstance(dataset, datasets.GTSRB):
        targets = []
        for _, target in dataset:
            targets.append(target)
        return np.array(targets)
    else:
        # For ImageFolder datasets like ImageNette
        return np.array([y for _, y in dataset])

def get_transform(dataset_name: str) -> Optional[transforms.Compose]:
    if dataset_name == 'CIFAR100':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
    elif dataset_name == 'GTSRB':
        return transforms.Compose([
            transforms.Resize((32, 32)),  # Resize all images to 32x32
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3337, 0.3064, 0.3171],  # GTSRB mean values
                std=[0.2672, 0.2564, 0.2629]   # GTSRB std values
            )
        ])
    elif "ImageNette" in dataset_name:
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to standard ImageNet size
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean values
                std=[0.229, 0.224, 0.225]    # ImageNet std values
            )
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def load_dataset(dataset_name: str, transform: Optional[transforms.Compose] = None) -> Dataset:
    """Load dataset based on name.
    
    Args:
        dataset_name: Name of the dataset to load
        transform: Optional transform to apply
        
    Returns:
        Dataset object
    """
    if transform is None:
        transform = get_transform(dataset_name)
    
    if dataset_name == 'CIFAR100':
        dataset = datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        logging.info(f"Loaded CIFAR-100 dataset with {len(dataset)} samples")
        logging.info(f"Number of classes: {100}")
        return dataset
    elif dataset_name == 'GTSRB':
        dataset = datasets.GTSRB(
            root='./data',
            split='train',
            download=True,
            transform=transform
        )
        logging.info(f"Loaded GTSRB dataset with {len(dataset)} samples")
        logging.info(f"Number of classes: {43}")
        return dataset
    elif "ImageNette" in dataset_name:
        dataset = datasets.ImageFolder(
            root=os.path.join(os.path.dirname(__file__), '.datasets/imagenette/train'),
            transform=transform
        )
        logging.info(f"Loaded ImageNette dataset with {len(dataset)} samples")
        logging.info(f"Number of classes: {10}")
        return dataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

class SoftKNN(nn.Module):
    """Soft KNN classifier that can be used with gradient-based attacks."""
    
    def __init__(self, n_neighbors=5, temperature=1.0):
        """Initialize the Soft KNN classifier."""
        super().__init__()
        self.n_neighbors = n_neighbors
        self.temperature = temperature
        self.train_features = None
        self.train_labels = None
        self.device = device
        logging.info(f"Initialized SoftKNN on {self.device}")
    
    def fit(self, train_features, train_labels):
        """Store training data."""
        if isinstance(train_features, np.ndarray):
            train_features = torch.tensor(train_features, dtype=torch.float32)
        if isinstance(train_labels, np.ndarray):
            train_labels = torch.tensor(train_labels, dtype=torch.long)
            
        self.train_features = train_features.to(self.device)
        self.train_labels = train_labels.to(self.device)
        return self
    
    def compute_distances(self, x: torch.Tensor, y: torch.Tensor):
        """Compute pairwise Euclidean distances between x and y."""
        x = x.to(self.device)
        y = y.to(self.device)
        x_norm = (x**2).sum(1).view(-1, 1)
        y_norm = (y**2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.transpose(0, 1))
        return torch.clamp(dist, min=0.0).sqrt()
    
    def forward(self, x: torch.Tensor):
        """Compute soft KNN predictions."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        
        # Compute distances
        distances = self.compute_distances(x, self.train_features)
        
        # Get k nearest neighbors
        _, indices = torch.topk(distances, k=self.n_neighbors, dim=1, largest=False)
        neighbor_labels = self.train_labels[indices]
        
        # One-hot encode labels
        n_classes = torch.max(self.train_labels) + 1
        one_hot = torch.zeros(neighbor_labels.size(0), neighbor_labels.size(1), n_classes, device=self.device)
        one_hot.scatter_(2, neighbor_labels.unsqueeze(2), 1)
        
        # Compute softmax weights
        weights = torch.softmax(-distances[torch.arange(len(x)).unsqueeze(1), indices] / self.temperature, dim=1)
        weights = weights.unsqueeze(2)
        
        # Weighted sum of one-hot vectors
        output = (weights * one_hot).sum(1)
        return output
    
    def predict(self, x: torch.Tensor):
        """Make predictions."""
        with torch.no_grad():
            output = self.forward(x)
            return output.argmax(1).cpu().numpy()
    
    def get_loss_and_grad(self, x: torch.Tensor, labels: torch.Tensor):
        """Compute loss and gradients for adversarial attacks."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels, dtype=torch.long)
            
        x = x.to(self.device)
        labels = labels.to(self.device)
        
        output = self.forward(x)
        loss = torch.nn.functional.cross_entropy(output, labels)
        
        # Handle the case where loss might be a float
        loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
        
        # Only compute gradients if loss is a tensor
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()
            # Keep gradients on the same device
            grad = x.grad.clone() if x.grad is not None else None
        else:
            grad = None
            
        return loss_val, grad

class DatasetHandler:
    """Handles dataset loading and preprocessing."""
    
    def __init__(self, dataset_name: str, dataset_type: str = 'image', sample_size: Optional[int] = None,
                 batch_size: int = 32, device: str = 'cuda'):
        """Initialize DatasetHandler."""
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.device = device
        
        # Create necessary directories
        os.makedirs('results', exist_ok=True)
        
        logging.info(f"Initialized DatasetHandler for {dataset_name}")
        logging.info(f"Dataset type: {dataset_type}")
        logging.info(f"Sample size: {sample_size}")
        logging.info(f"Initial GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        logging.info("")
    
    def load_data(self):
        """Load and preprocess dataset."""
        transform = self._get_transform()
        
        if self.dataset_name == 'CIFAR100':
            train_dataset = datasets.CIFAR100(
                root=os.path.join(os.path.dirname(__file__), '.datasets'),
                train=True,
                download=True,
                transform=transform
            )
            val_dataset = datasets.CIFAR100(
                root=os.path.join(os.path.dirname(__file__), '.datasets'),
                train=False,
                download=True,
                transform=transform
            )
        elif self.dataset_name == 'GTSRB':
            train_dataset = datasets.GTSRB(
                root=os.path.join(os.path.dirname(__file__), '.datasets'),
                split='train',
                download=True,
                transform=transform
            )
            val_dataset = datasets.GTSRB(
                root=os.path.join(os.path.dirname(__file__), '.datasets'),
                split='test',
                download=True,
                transform=transform
            )
        elif "ImageNette" in self.dataset_name:
            train_dataset = datasets.ImageFolder(
                root=os.path.join(os.path.dirname(__file__), '.datasets/imagenette/train'),
                transform=transform
            )
            val_dataset = datasets.ImageFolder(
                root=os.path.join(os.path.dirname(__file__), '.datasets/imagenette/val'),
                transform=transform
            )
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        # Apply subsetting if sample size is specified
        if self.sample_size is not None:
            train_dataset = self._subset_dataset(train_dataset)
            # Also subset validation set to 20% of training size
            val_size = max(int(self.sample_size * 0.2), 100)  # At least 100 validation samples
            val_dataset = self._subset_dataset(val_dataset, sample_size=val_size)
        
        logging.info(f"Loaded {len(train_dataset)} training samples")
        logging.info(f"Loaded {len(val_dataset)} validation samples")
        
        return train_dataset, val_dataset
    
    def _get_transform(self) -> transforms.Compose:
        """Get dataset-specific transforms."""
        if self.dataset_type == 'image':
            if self.dataset_name == 'CIFAR100':
                return transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5071, 0.4867, 0.4408],
                        std=[0.2675, 0.2565, 0.2761]
                    )
                ])
            elif self.dataset_name == 'GTSRB':
                return transforms.Compose([
                    transforms.Resize((32, 32)),  # Resize all images to 32x32
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.3337, 0.3064, 0.3171],  # GTSRB mean values
                        std=[0.2672, 0.2564, 0.2629]   # GTSRB std values
                    )
                ])
            elif "ImageNette" in self.dataset_name:
                return transforms.Compose([
                    transforms.Resize((224, 224)),  # Resize to standard ImageNet size
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],  # ImageNet mean values
                        std=[0.229, 0.224, 0.225]    # ImageNet std values
                    )
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                
                if "Dynadetect" in self.dataset_name:
                    # Add data augmentation for robustness
                    transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transform
                    ])
                
                return transform
        else:
            raise ValueError(f"Dataset type {self.dataset_type} not supported")
    
    def _subset_dataset(self, dataset: Dataset, sample_size: Optional[int] = None) -> Dataset:
        """Create a balanced subset of the dataset."""
        actual_size = sample_size if sample_size is not None else self.sample_size
        if actual_size is None:
            return dataset
        return balance_dataset(dataset, sample_size=actual_size)

class PoisoningMethods:
    """Implements various poisoning methods."""
    
    @staticmethod
    def apply_poisoning(
        dataset: Dataset,
        method: str,
        poison_rate: float,
        **kwargs
    ) -> Tuple[Dataset, List[int], List[int], str]:
        """Apply poisoning to dataset.
        
        Args:
            dataset: Dataset to poison
            method: Poisoning method
            poison_rate: Rate of poisoning
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Tuple of (poisoned dataset, indices of poisoned samples, source classes, flip type)
        """
        num_samples = len(dataset)
        num_poison = int(num_samples * poison_rate)
        
        if method == 'label_flipping':
            mode = kwargs.get('mode', 'random_to_random')
            source_class = kwargs.get('source_class', None)
            target_class = kwargs.get('target_class', None)
            
            # Get indices to poison
            indices = torch.randperm(num_samples)[:num_poison].tolist()
            
            # Get original labels
            targets = get_targets(dataset)
            source_classes = [targets[i] for i in indices]
            
            # Create modified dataset
            poisoned_dataset = copy.deepcopy(dataset)
            if isinstance(dataset, Subset):
                dataset_targets = poisoned_dataset.dataset.targets
                for idx in indices:
                    current_label = dataset_targets[poisoned_dataset.indices[idx]]
                    # Get number of classes based on dataset
                    if isinstance(dataset.dataset, datasets.CIFAR100):
                        num_classes = 100  # CIFAR-100 has 100 classes
                    elif isinstance(dataset.dataset, datasets.GTSRB):
                        num_classes = 43  # GTSRB has 43 classes
                    else:  # ImageNette has 10 classes
                        num_classes = 10
                    possible_labels = list(range(num_classes))
                    possible_labels.remove(current_label)
                    new_label = random.choice(possible_labels)
                    dataset_targets[poisoned_dataset.indices[idx]] = new_label
            else:
                for idx in indices:
                    current_label = poisoned_dataset.targets[idx]
                    # Get number of classes based on dataset
                    if isinstance(dataset, datasets.CIFAR100):
                        num_classes = 100  # CIFAR-100 has 100 classes
                    elif isinstance(dataset, datasets.GTSRB):
                        num_classes = 43  # GTSRB has 43 classes
                    else:  # ImageNette has 10 classes
                        num_classes = 10
                    possible_labels = list(range(num_classes))
                    possible_labels.remove(current_label)
                    new_label = random.choice(possible_labels)
                    poisoned_dataset.targets[idx] = new_label
            
            return poisoned_dataset, indices, source_classes, mode
        else:
            raise ValueError(f"Unknown poisoning method: {method}")

def pgd_attack(model, images, labels, eps=0.1, alpha=0.01, iters=40, poison_rate=0.01):
    """
    Perform PGD attack on the given images.
    
    Args:
        model: Model to attack
        images: Input images
        labels: True labels
        eps: Maximum perturbation
        alpha: Step size
        iters: Number of iterations
        poison_rate: Percentage of images to poison
        
    Returns:
        Tuple of (perturbed images, indices of poisoned samples)
    """
    device = model.device if hasattr(model, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if isinstance(images, np.ndarray):
        images = torch.tensor(images, dtype=torch.float32)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels, dtype=torch.long)
    
    # Ensure inputs are on the correct device
    images = images.to(device)
    labels = labels.to(device)
        
    # Select indices to poison
    num_samples = len(images)
    num_poison = max(1, int(poison_rate * num_samples))
    
    # Initialize perturbed images
    perturbed = images.clone().detach()
    
    # Track successful perturbations
    successful_indices = []
    
    for idx in range(num_samples):
        logging.debug(f"\nProcessing sample {idx}")
        # Get single sample and make it require gradients
        sample = perturbed[idx:idx+1].clone().detach().requires_grad_(True)
        target = labels[idx:idx+1]
        
        success = False
        for i in range(iters):
            loss_val, grad = model.get_loss_and_grad(sample, target)
            logging.debug(f"Iteration {i}, Loss: {loss_val}")
            
            if grad is None:
                logging.debug("No gradient available")
                break
                
            # Update the sample
            with torch.no_grad():
                # Ensure gradient is on the same device as sample
                grad = grad.to(device)
                sample = sample + alpha * grad.sign()
                
                # Project onto epsilon ball
                sample = torch.min(torch.max(sample, images[idx:idx+1] - eps), 
                                 images[idx:idx+1] + eps)
                
                # Ensure the sample requires gradients for next iteration
                sample.requires_grad_(True)
                success = True
        
        if success:
            perturbed[idx] = sample.detach()[0]
            successful_indices.append(idx)
            
    logging.info(f"Successfully poisoned {len(successful_indices)}/{num_poison} samples")
    return perturbed, np.array(successful_indices)

def _apply_pgd_attack(
    self,
    dataset: Dataset,
    poison_rate: float,
    eps: float = 0.1,  # Reduced from 0.3 to be more subtle
    alpha: float = 0.01,  # Increased from 2/255 for stronger updates
    iters: int = 40,
    **kwargs
) -> Dataset:
    """Apply PGD attack."""
    # Convert dataset to tensors
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    images, labels = next(iter(data_loader))
    
    # Convert images to float
    images = images.float()
    
    # Initialize and fit KNN model with clean data
    n_neighbors = min(5, len(labels))  # Ensure n_neighbors doesn't exceed dataset size
    model = SoftKNN(n_neighbors=n_neighbors, temperature=1.0)  # Increased temperature for softer predictions
    
    # Extract features if needed
    if len(images.shape) == 4:  # Image data
        features = images.view(images.shape[0], -1)  # Flatten images
    else:
        features = images
    
    # Fit model on clean data
    model.fit(features, labels)
    
    # Apply PGD attack
    poisoned_images, poison_indices = pgd_attack(
        model=model,
        images=images,
        labels=labels,
        eps=eps,
        alpha=alpha,
        iters=iters,
        poison_rate=poison_rate
    )
    
    # Create new dataset with poisoned images
    poisoned_dataset = copy.deepcopy(dataset)
    if isinstance(dataset, Subset):
        original_dataset = dataset.dataset
        indices = dataset.indices
        for i, idx in enumerate(indices):
            if i in poison_indices:
                if hasattr(original_dataset, 'data'):
                    original_dataset.data[idx] = poisoned_images[i].detach().cpu().numpy()
                else:
                    original_dataset.tensors = (poisoned_images.detach().cpu(), original_dataset.tensors[1])
        return Subset(poisoned_dataset, indices)
    else:
        if hasattr(poisoned_dataset, 'data'):
            for idx in poison_indices:
                poisoned_dataset.data[idx] = poisoned_images[idx].detach().cpu().numpy()
        else:
            poisoned_dataset.tensors = (poisoned_images.detach().cpu(), poisoned_dataset.tensors[1])
        return poisoned_dataset

def evaluate_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray, num_classes: int) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    cm = confusion_matrix(true_labels, predicted_labels)
    
    class_accuracies = np.zeros(num_classes)
    for i in range(num_classes):
        if cm.sum(axis=1)[i] > 0:
            class_accuracies[i] = cm[i, i] / cm.sum(axis=1)[i]
    
    logging.info(f"Confusion Matrix:\n{cm}")
    logging.info(f"Class-specific accuracies: {class_accuracies}")

    return accuracy, precision, recall, f1, cm, class_accuracies

def balance_dataset(dataset: Dataset, sample_size: Optional[int] = None, min_samples_per_class: int = 50, num_classes: int = 100) -> Subset:
    labels = get_targets(dataset)
    class_indices = {i: [] for i in range(num_classes)}
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    balanced_indices = []
    if sample_size is not None:
        samples_per_class = sample_size // num_classes
    else:
        samples_per_class = min_samples_per_class

    for class_id, indices in class_indices.items():
        if len(indices) < min_samples_per_class:
            raise ValueError(f"Class {class_id} has fewer than {min_samples_per_class} samples.")
        balanced_indices.extend(
            np.random.choice(indices, min(samples_per_class, len(indices)), 
                           replace=False))
    
    return torch.utils.data.Subset(dataset, balanced_indices)

def extract_features(data_loader: DataLoader, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from dataset.
    
    For image datasets, we use a pre-trained ResNet18 as feature extractor.
    The model is more efficient than ResNet50 while maintaining good performance.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize feature extractor
    if "Dynadetect" in dataset_name:
        model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()
        
        logging.info(f"Using ResNet18 feature extractor for {dataset_name}")
        logging.info(f"Running on device: {device}")
        if torch.cuda.is_available():
            logging.info(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        all_features = []
        all_labels = []
        
        batch_size = data_loader.batch_size
        total_batches = len(data_loader)
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx % 10 == 0:
                    logging.info(f"Processing batch {batch_idx}/{total_batches}")
                    if torch.cuda.is_available():
                        logging.info(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
                
                # Move data to GPU
                data = data.to(device)
                
                # Extract features
                features = feature_extractor(data)
                features = features.view(features.size(0), -1)
                
                # Move results back to CPU efficiently
                all_features.append(features.cpu())
                all_labels.append(targets)
                
                if batch_idx % 10 == 0:
                    logging.info(f"Batch feature shape: {features.shape}")
        
        # Concatenate all features and convert to numpy
        features = torch.cat(all_features, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        
        logging.info(f"Final feature shape: {features.shape}")
        
    else:
        # For non-Dynadetect datasets, use simpler feature extraction
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for data, targets in data_loader:
                # Move data to GPU if it's an image dataset
                if len(data.shape) == 4:  # Image data
                    data = data.to(device)
                    features = data.view(data.size(0), -1)
                    features = features.cpu()
                else:  # Numerical data
                    features = data.view(data.size(0), -1)
                
                all_features.append(features)
                all_labels.append(targets)
        
        features = torch.cat(all_features, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
    
    return features, labels

def preprocess_and_subset(dataset_name: str, sample_size: Optional[int] = None, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, Dataset, Subset]:
    logging.info(f"Preprocessing dataset: {dataset_name}")

    transform = get_transform(dataset_name)
    dataset = load_dataset(dataset_name, transform)
    
    # Adjust min_samples_per_class if sample_size is too small
    if sample_size is not None:
        num_classes = len(dataset.classes) if hasattr(dataset, 'classes') else len(np.unique(get_targets(dataset)))
        samples_per_class = sample_size // num_classes
        min_samples_per_class = min(50, samples_per_class)
    else:
        min_samples_per_class = 50
    
    balanced_dataset = balance_dataset(dataset, sample_size, min_samples_per_class, num_classes)
    balanced_loader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=False)
    features, balanced_labels = extract_features(balanced_loader, dataset_name)
    
    unique, counts = np.unique(balanced_labels, return_counts=True)
    logging.info(f"Class distribution after balancing: {dict(zip(unique, counts))}")
    
    logging.info(f"Feature and label shapes for {dataset_name}: {features.shape}, {balanced_labels.shape}")

    return features, balanced_labels, dataset, balanced_dataset

def process_validation_set(val_dataset: Dataset, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Process validation dataset."""
    if dataset_name == 'CIFAR100':
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
        val_dataset = datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=val_transform
        )
    elif dataset_name == 'GTSRB':
        val_transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize all images to 32x32
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3337, 0.3064, 0.3171],  # GTSRB mean values
                std=[0.2672, 0.2564, 0.2629]   # GTSRB std values
            )
        ])
        val_dataset = datasets.GTSRB(
            root='./data',
            split='test',
            download=True,
            transform=val_transform
        )
    elif "ImageNette" in dataset_name:
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to standard ImageNet size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean values
                std=[0.229, 0.224, 0.225]    # ImageNet std values
            )
        ])
        val_dataset = datasets.ImageFolder(
            root=os.path.join(os.path.dirname(__file__), '.datasets/imagenette/val'),
            transform=val_transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    val_features, val_labels = extract_features(val_loader, dataset_name)
    
    # Flatten the features if they are 4D (image data)
    if val_features.ndim == 4:
        val_features = val_features.reshape(val_features.shape[0], -1)
        logging.info(f"Validation features flattened. New shape: {val_features.shape}")
    
    return val_features, val_labels

class ModelFactory:
    """Factory for creating models."""
    
    @staticmethod
    def create_model(model_name: str, **kwargs) -> BaseEstimator:
        """Create model based on name."""
        mode = kwargs.get('mode', 'regular')
        
        # Create base classifier
        if model_name == 'LogisticRegression':
            model = LogisticRegression(max_iter=1000)
        elif model_name == 'KNeighborsClassifier':
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_name == 'SVM':
            model = SVC(probability=True)
        elif model_name == 'RF':
            model = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Wrap with DynaDetect if needed
        if mode == 'dynadetect':
            model = DynaDetectWrapper(model)
        
        return model

class DynaDetectWrapper(BaseEstimator):
    """Wrapper class that adds DynaDetect functionality to any classifier."""
    
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier
        self.trainer = DynaDetectTrainer(n_components=100, contamination=0.1)
        
    def fit(self, X, y):
        # Apply DynaDetect's robust feature selection and anomaly detection
        X_transformed, sample_weights = self.trainer.fit_transform(X, y)
        self.base_classifier.fit(X_transformed, y, sample_weight=sample_weights)
        return self
        
    def predict(self, X):
        X_transformed = self.trainer.transform(X)
        return self.base_classifier.predict(X_transformed)
        
    def predict_proba(self, X):
        X_transformed = self.trainer.transform(X)
        if hasattr(self.base_classifier, 'predict_proba'):
            return self.base_classifier.predict_proba(X_transformed)
        else:
            # For models like SVM that don't have predict_proba by default
            return self.base_classifier.decision_function(X_transformed)

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
        
    def _train_knn(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray = None) -> None:
        """Train K-Nearest Neighbors classifier."""
        if sample_weights is not None:
            logging.warning("KNN classifier does not support sample weights. Ignoring weights.")
        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(X, y)
    
    def train_model(
        self,
        model: BaseEstimator,
        features: np.ndarray,
        labels: np.ndarray
    ) -> BaseEstimator:
        """Train the model with given features and labels."""
        if self.classifier_name == 'KNN':
            self._train_knn(features, labels)
        else:
            model.fit(features, labels)
        return model
        
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

    def compute_metrics(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        num_classes = len(np.unique(true_labels))
        accuracy, precision, recall, f1, cm, class_accuracies = evaluate_metrics(
            true_labels, predicted_labels, num_classes)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'class_accuracies': class_accuracies
        }

    def log_results(self, metrics: Dict[str, Any], dataset_name: str, modification_method: str, num_poisoned: int, poisoned_classes: List[Any], flip_type: str, latency: float, iteration: int, classifier_name: str, total_images: int, timestamp: str):
        """Log evaluation results."""
        # Log to console
        logging.info(f"Evaluation results for {dataset_name}:")
        logging.info(f"Modification method: {modification_method}")
        logging.info(f"Number of samples poisoned: {num_poisoned}")
        logging.info(f"Poisoned classes: {poisoned_classes}")
        logging.info(f"Flip type: {flip_type}")
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Precision: {metrics['precision']:.4f}")
        logging.info(f"Recall: {metrics['recall']:.4f}")
        logging.info(f"F1 Score: {metrics['f1']:.4f}")
        logging.info(f"Inference Latency: {latency:.4f} seconds")
        
        # Create results directory if it doesn't exist
        base_dir = os.getcwd()  # Use current working directory
        results_dir = os.path.join(base_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Prepare CSV data
        csv_file = os.path.join(results_dir, f'experiment_results_{timestamp}.csv')
        file_exists = os.path.exists(csv_file)
        
        # Format class accuracies
        class_accuracies = metrics['class_accuracies']
        class_acc_dict = {f'Class_{i}_Accuracy': acc for i, acc in enumerate(class_accuracies)}
        
        # Prepare row data
        row_data = {
            'Iteration': iteration,
            'Dataset': dataset_name,
            'Classifier': classifier_name,
            'Modification_Method': modification_method,
            'Total_Images': total_images,
            'Num_Poisoned': num_poisoned,
            'Poisoned_Classes': ','.join(map(str, poisoned_classes)),
            'Flip_Type': flip_type,
            'Flip_Counts': num_poisoned,  # This could be modified if we track individual flip counts
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'Latency': latency,
            **class_acc_dict  # Add all class accuracies
        }
        
        # Write to CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
            
        logging.info(f"Results saved to: {os.path.abspath(csv_file)}")

class DatasetEvaluator:
    """Evaluates datasets with different models and poisoning methods."""
    
    def __init__(self, config: 'DatasetConfig', classifier_name: str):
        """Initialize evaluator with configuration."""
        self.dataset_name = config.name
        self.dataset_type = config.dataset_type
        self.sample_size = config.sample_size
        self.modification_method = config.modification_method
        self.attack_params = config.attack_params
        self.poison_rate = config.poison_rate
        self.metric = config.metric
        self.classifier_name = classifier_name
        self.mode = config.mode
        self.device = device
        self.num_poisoned = 0
        
        logging.info(f"Initializing DatasetEvaluator for {self.dataset_name}")
        logging.info(f"Dataset type: {self.dataset_type}")
        logging.info(f"Sample size: {self.sample_size}")
        logging.info(f"Modification method: {self.modification_method}")
        logging.info(f"Poison rate: {self.poison_rate}")
        logging.info(f"Classifier: {classifier_name}")
        logging.info(f"Mode: {self.mode}")
        logging.info(f"Using device: {self.device}")
        
        # Initialize components
        self.dataset_handler = DatasetHandler(
            dataset_name=self.dataset_name,
            dataset_type=self.dataset_type,
            sample_size=self.sample_size
        )
        
        self.model_factory = ModelFactory()
        self.poisoning_methods = PoisoningMethods()
        
        # Initialize DynaDetect trainer if needed
        if "Dynadetect" in self.dataset_name:
            self.trainer = DynaDetectTrainer(
                n_components=100,  # Adjusted for GPU memory
                contamination=0.1
            )
        else:
            self.trainer = None
        
        if torch.cuda.is_available():
            logging.info(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    def run_evaluation(self, iteration: int = 0, timestamp: str = None) -> Tuple[Dict[str, Any], int, List[int], str, float]:
        """Run evaluation pipeline."""
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
        logging.info(f"\nStarting evaluation iteration {iteration}")
        
        # Load and preprocess dataset
        train_dataset, val_dataset = self.dataset_handler.load_data()
        
        # Get total number of images
        total_images = len(train_dataset) + len(val_dataset)
        
        # Process training set
        train_features, train_labels = self.process_training_set(train_dataset)
        
        # Create and train model
        model = self.model_factory.create_model(self.classifier_name, mode=self.mode)
        
        # Initialize latency
        start_time = time.time()
        
        if "Dynadetect" in self.dataset_name:
            metrics = self._train_with_dynadetect(model, train_features, train_labels)
        else:
            # Standard training
            model.fit(train_features, train_labels)
            
            # Process validation set
            val_features, val_labels = process_validation_set(val_dataset, self.dataset_name)
            
            # Make predictions
            predictions = model.predict(val_features)
            
            # Compute metrics
            accuracy, precision, recall, f1, cm, class_accuracies = evaluate_metrics(val_labels, predictions, len(np.unique(train_labels)))
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm,
                'class_accuracies': class_accuracies
            }
        
        # Calculate total latency
        latency = time.time() - start_time
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info(f"Final GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        # Log results
        evaluator = ModelEvaluator(classifier_name=self.classifier_name, mode=self.mode)
        evaluator.log_results(
            metrics=metrics,
            dataset_name=self.dataset_name,
            modification_method=self.modification_method,
            num_poisoned=self.num_poisoned,
            poisoned_classes=[],
            flip_type='none',
            latency=latency,
            iteration=iteration,
            classifier_name=self.classifier_name,
            total_images=total_images,
            timestamp=timestamp
        )
        
        return metrics, self.num_poisoned, [], self.modification_method, latency
    
    def _train_with_dynadetect(self, model, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """DynaDetect specific training logic."""
        logging.info("Training with DynaDetect")
        
        # Apply DynaDetect feature transformation
        transformed_features, sample_weights = self.trainer.fit_transform(features, labels)
        
        # Train model with sample weights
        if hasattr(model, 'fit_transform'):
            model.fit(transformed_features, labels, sample_weight=sample_weights)
        else:
            model.fit(transformed_features, labels)
        
        # Get anomaly scores
        anomaly_scores = self.trainer.predict_anomaly_scores(features)
        
        metrics = {
            'anomaly_scores': anomaly_scores,
            'sample_weights': sample_weights
        }
        
        return metrics
    
    def process_training_set(self, train_dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Process training dataset."""
        batch_size = min(32, len(train_dataset))  # Adjust batch size based on dataset size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        features, labels = extract_features(train_loader, self.dataset_name)
        
        if self.modification_method:
            logging.info(f"Applying {self.modification_method} modification")
            if self.modification_method == 'label_flipping':
                labels, self.num_poisoned = label_flipping(
                    labels=labels,
                    mode=self.attack_params.get('mode', 'random_to_random'),
                    target_class=self.attack_params.get('target_class'),
                    source_class=self.attack_params.get('source_class'),
                    poison_rate=self.poison_rate
                )
            elif self.modification_method in ['pgd', 'gradient_ascent']:
                # Convert to torch tensors for GPU operations
                features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
                labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
                
                # Create temporary SoftKNN model for attack
                temp_model = SoftKNN(n_neighbors=5).to(self.device)
                temp_model.fit(features_tensor, labels_tensor)
                
                if self.modification_method == 'pgd':
                    features_tensor, indices = pgd_attack(
                        temp_model, features_tensor, labels_tensor,
                        eps=self.attack_params.get('eps', 0.1),
                        alpha=self.attack_params.get('alpha', 0.01),
                        iters=self.attack_params.get('iters', 40),
                        poison_rate=self.poison_rate
                    )
                else:  # gradient_ascent
                    features_tensor, indices = gradient_ascent(
                        temp_model, features_tensor, labels_tensor,
                        lr=self.attack_params.get('lr', 0.1),
                        iters=self.attack_params.get('iters', 100),
                        poison_rate=self.poison_rate
                    )
                
                # Move back to CPU
                features = features_tensor.cpu().numpy()
                del temp_model
                torch.cuda.empty_cache()
        
        return features, labels
    
    def process_validation_set(self, val_dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Process validation dataset."""
        val_dataset = process_validation_set(val_dataset, self.dataset_name)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        return extract_features(val_loader, self.dataset_name)

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
        self.feature_selector = None
        self.anomaly_detector = None
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
        logging.info("Starting DynaDetect fit_transform")
        if torch.cuda.is_available():
            logging.info(f"GPU Memory before processing: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        # Convert to torch tensors for GPU operations
        features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # 1. Feature Selection
        # Use CPU for feature selection as sklearn doesn't support GPU
        features_cpu = features_tensor.cpu().numpy()
        labels_cpu = labels_tensor.cpu().numpy()
        
        # Select features using mutual information
        self.feature_selector = SelectKBest(mutual_info_classif, k=min(self.n_components, features.shape[1]))
        features_selected = self.feature_selector.fit_transform(features_cpu, labels_cpu)
        
        # 2. Robust Scaling
        features_scaled = self.scaler.fit_transform(features_selected)
        
        # 3. Anomaly Detection
        self.anomaly_detector = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        # Fit anomaly detector and get decision scores
        self.anomaly_detector.fit(features_scaled)
        decision_scores = self.anomaly_detector.score_samples(features_scaled)
        
        # Convert scores to sample weights (higher weight = more likely to be clean)
        sample_weights = np.clip(decision_scores - decision_scores.min(), 0, None)
        sample_weights = sample_weights / sample_weights.max()  # Normalize to [0,1]
        
        if torch.cuda.is_available():
            logging.info(f"GPU Memory after processing: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            torch.cuda.empty_cache()
        
        return features_scaled, sample_weights
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform new features using fitted components."""
        if self.feature_selector is None or self.scaler is None:
            raise RuntimeError("DynaDetect components not fitted. Call fit_transform first.")
        
        # Select features and scale
        features_selected = self.feature_selector.transform(features)
        features_scaled = self.scaler.transform(features_selected)
        
        return features_scaled
    
    def predict_anomaly_scores(self, features: np.ndarray) -> np.ndarray:
        """Predict anomaly scores for new features."""
        if self.anomaly_detector is None:
            raise RuntimeError("Anomaly detector not fitted. Call fit_transform first.")
        
        # Transform features first
        features_transformed = self.transform(features)
        
        # Get anomaly scores
        return self.anomaly_detector.score_samples(features_transformed)

# Define Numerical Dataset Class
class NumericalDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.targets = self.y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

# Configuration
config4 = {
    'datasets': [
        #GTSRB
        {
            'name': 'GTSRB (Dynadetect)',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': '',  # No modification for baseline
            'poison_rate': 0.0,  # Irrelevant since no modification
            'attack_params': {}  # Empty since no attack
        },
        {
            'name': 'GTSRB',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': '',  # No modification for baseline
            'poison_rate': 0.0,  # Irrelevant since no modification
            'attack_params': {}  # Empty since no attack
        },
        {
            'name': 'GTSRB',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'label_flipping',
            'poison_rate': 0.25,
            'attack_params': {'mode': 'random_to_target', 'target_class': 7}
        },
        {
            'name': 'GTSRB',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'label_flipping',
            'poison_rate': 0.15,
            'attack_params': {'mode': 'source_to_target', 'source_class': 2, 'target_class': 9}
        },
    
        # GTSRB (Dynadetect)
        {
            'name': 'GTSRB (Dynadetect)',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'label_flipping',
            'poison_rate': 0.2,
            'attack_params': {'mode': 'random_to_target', 'target_class': 5}
        },
        {
            'name': 'GTSRB (Dynadetect)',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'label_flipping',
            'poison_rate': 0.1,
            'attack_params': {'mode': 'source_to_target', 'source_class': 1, 'target_class': 8}
        },
    
        # Imagenette
        {
            'name': 'Imagenette (Dynadetect)',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': '',  # No modification for baseline
            'poison_rate': 0.0,  # Irrelevant since no modification
            'attack_params': {}  # Empty since no attack
        },
        {
            'name': 'Imagenette',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': '',  # No modification for baseline
            'poison_rate': 0.0,  # Irrelevant since no modification
            'attack_params': {}  # Empty since no attack
        },
        {
            'name': 'Imagenette',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'label_flipping',
            'poison_rate': 0.3,
            'attack_params': {'mode': 'random_to_target', 'target_class': 3}
        },
        {
            'name': 'Imagenette',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'label_flipping',
            'poison_rate': 0.18,
            'attack_params': {'mode': 'source_to_target', 'source_class': 0, 'target_class': 5}
        },
    
        # Imagenette (Dynadetect)
        {
            'name': 'Imagenette (Dynadetect)',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'label_flipping',
            'poison_rate': 0.22,
            'attack_params': {'mode': 'random_to_target', 'target_class': 2}
        },
        {
            'name': 'Imagenette (Dynadetect)',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'label_flipping',
            'poison_rate': 0.12,
            'attack_params': {'mode': 'source_to_target', 'source_class': 4, 'target_class': 7}
        },
    
        # CIFAR10
        {
            'name': 'CIFAR10',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'label_flipping',
            'poison_rate': 0.28,
            'attack_params': {'mode': 'random_to_target', 'target_class': 1}
        },
        {
            'name': 'CIFAR10',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'label_flipping',
            'poison_rate': 0.16,
            'attack_params': {'mode': 'source_to_target', 'source_class': 3, 'target_class': 6}
        },
    
        #CIFAR10 (Dynadetect)
        {
            'name': 'CIFAR10 (Dynadetect)',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': '',  # No modification for baseline
            'poison_rate': 0.0,  # Irrelevant since no modification
            'attack_params': {}  # Empty since no attack
        },
        {
            'name': 'CIFAR10 (Dynadetect)',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'label_flipping',
            'poison_rate': 0.14,
            'attack_params': {'mode': 'source_to_target', 'source_class': 5, 'target_class': 9}
        },
    
        # Diabetes
        {
            'name': 'Diabetes',
            'type': 'numerical',
            'sample_size': None,
            'metric': 'euclidean',
            'modification_method': 'label_flipping',
            'poison_rate': 0.2,
            'attack_params': {'mode': 'random_to_target', 'target_class': 1}
        },
        {
            'name': 'Diabetes',
            'type': 'numerical',
            'sample_size': None,
            'metric': 'euclidean',
            'modification_method': 'label_flipping',
            'poison_rate': 0.15,
            'attack_params': {'mode': 'source_to_target', 'source_class': 0, 'target_class': 1}
        },
    
        # WineQuality
        {
            'name': 'WineQuality',
            'type': 'numerical',
            'sample_size': None,
            'metric': 'euclidean',
            'modification_method': 'label_flipping',
            'poison_rate': 0.18,
            'attack_params': {'mode': 'random_to_target', 'target_class': 3}
        },
        {
            'name': 'WineQuality',
            'type': 'numerical',
            'sample_size': None,
            'metric': 'euclidean',
            'modification_method': 'label_flipping',
            'poison_rate': 0.12,
            'attack_params': {'mode': 'source_to_target', 'source_class': 2, 'target_class': 4}
        },
        # GTSRB
        {
            'name': 'GTSRB',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'gradient_ascent',
            'poison_rate': 0.4,
            'attack_params': {'mode': 'random_to_random', 'target_class': 3}
        },
        {
            'name': 'GTSRB',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'pgd',
            'poison_rate': 0.2,
            'attack_params': {'mode': 'random_to_targeted', 'target_class': 5}
        },
        {
            'name': 'GTSRB',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'label_flipping',
            'poison_rate': 0.1,
            'attack_params': {'mode': 'targeted_to_random', 'target_class': 1}
        },
        
        # GTSRB (Dynadetect)
        {
            'name': 'GTSRB (Dynadetect)',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'gradient_ascent',
            'poison_rate': 0.3,
            'attack_params': {'mode': 'random_to_random', 'target_class': 2}
        },
        {
            'name': 'GTSRB (Dynadetect)',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'pgd',
            'poison_rate': 0.0,
            'attack_params': {'mode': 'random_to_random', 'target_class': 4}
        },
        
        # Imagenette
        {
            'name': 'Imagenette',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'gradient_ascent',
            'poison_rate': 0.2,
            'attack_params': {'mode': 'random_to_targeted', 'target_class': 1}
        },
        {
            'name': 'Imagenette',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'label_flipping',
            'poison_rate': 0.3,
            'attack_params': {'mode': 'targeted_to_random', 'target_class': 2}
        },
        
        # Imagenette (Dynadetect)
        {
            'name': 'Imagenette (Dynadetect)',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'pgd',
            'poison_rate': 0.1,
            'attack_params': {'mode': 'random_to_random', 'target_class': 0}
        },
        {
            'name': 'Imagenette (Dynadetect)',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'label_flipping',
            'poison_rate': 0.0,
            'attack_params': {'mode': 'random_to_targeted', 'target_class': 3}
        },
        
        # CIFAR10
        {
            'name': 'CIFAR10',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'gradient_ascent',
            'poison_rate': 0.15,
            'attack_params': {'mode': 'targeted_to_random', 'target_class': 4}
        },
        {
            'name': 'CIFAR10',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'label_flipping',
            'poison_rate': 0.25,
            'attack_params': {'mode': 'random_to_targeted', 'target_class': 6}
        },
        
        # CIFAR10 (Dynadetect)
        {
            'name': 'CIFAR10 (Dynadetect)',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'pgd',
            'poison_rate': 0.35,
            'attack_params': {'mode': 'random_to_random', 'target_class': 7}
        },
        {
            'name': 'CIFAR10 (Dynadetect)',
            'type': 'image',
            'sample_size': 5000,
            'metric': 'cosine',
            'modification_method': 'label_flipping',
            'poison_rate': 0.0,
            'attack_params': {'mode': 'targeted_to_random', 'target_class': 8}
        },
        
        # Diabetes
        {
            'name': 'Diabetes',
            'type': 'numerical',
            'sample_size': None,
            'metric': 'euclidean',
            'modification_method': 'gradient_ascent',
            'poison_rate': 0.2,
            'attack_params': {'mode': 'random_to_random'}
        },
        {
            'name': 'Diabetes',
            'type': 'numerical',
            'sample_size': None,
            'metric': 'euclidean',
            'modification_method': 'pgd',
            'poison_rate': 0.1,
            'attack_params': {'mode': 'random_to_targeted'}
        },
        {
            'name': 'Diabetes',
            'type': 'numerical',
            'sample_size': None,
            'metric': 'euclidean',
            'modification_method': 'label_flipping',
            'poison_rate': 0.0,
            'attack_params': {'mode': 'targeted_to_random'}
        },
        
        # WineQuality
        {
            'name': 'WineQuality',
            'type': 'numerical',
            'sample_size': None,
            'metric': 'euclidean',
            'modification_method': 'gradient_ascent',
            'poison_rate': 0.05,
            'attack_params': {'mode': 'random_to_targeted'}
        },
        {
            'name': 'WineQuality',
            'type': 'numerical',
            'sample_size': None,
            'metric': 'euclidean',
            'modification_method': 'pgd',
            'poison_rate': 0.15,
            'attack_params': {'mode': 'targeted_to_random'}
        },
        {
            'name': 'WineQuality',
            'type': 'numerical',
            'sample_size': None,
            'metric': 'euclidean',
            'modification_method': 'label_flipping',
            'poison_rate': 0.0,
            'attack_params': {'mode': 'random_to_random'}
        },
        {
            'name': 'Diabetes',
            'type': 'numerical',
            'sample_size': None,
            'metric': 'euclidean',
            'modification_method': 'gradient_ascent',
            'poison_rate': 0.4,
            'attack_params': {'mode': 'random_to_random'}
        },
        {
            'name': 'Diabetes',
            'type': 'numerical',
            'sample_size': None,
            'metric': 'euclidean',
            'modification_method': '',
            'poison_rate': 0.0,
            'attack_params': {}
        },
        {
            'name': 'WineQuality',
            'type': 'numerical',
            'sample_size': None,
            'metric': 'euclidean',
            'modification_method': '',
            'poison_rate': 0.1,
            'attack_params': {}
        },
        {
            'name': 'WineQuality',
            'type': 'numerical',
            'sample_size': None,
            'metric': 'euclidean',
            'modification_method': 'label_flipping',
            'poison_rate': 0.1,
            'attack_params': {'mode': 'random_to_random'}
        },
        {
            'name': 'GTSRB', 
            'sample_size': 5000, 
            'metric': 'cosine',
            'modification_method': 'gradient_ascent',
            'poison_rate': 0.4,
            'attack_params': {
                'mode': 'random_to_random',
                'target_class': 3
            }
        },
        {
            'name': 'GTSRB (Dynadetect)', 
            'sample_size': 5000, 
            'metric': 'cosine',
            'modification_method': 'gradient_ascent',
            'poison_rate': 0.4,
            'attack_params': {
                'mode': 'random_to_random',
                'target_class': 3
            }
        },
        {
            'name': 'Imagenette (Dynadetect)', 
            'sample_size': 5000, 
            'metric': 'cosine',
            'modification_method': 'pgd',
            'poison_rate': 0.4,
            'attack_params': {
                'mode': 'random_to_random',
                'target_class': 3
            }
        },
        {
            'name': 'Imagenette', 
            'sample_size': 5000, 
            'metric': 'cosine',
            'modification_method': 'pgd',
            'poison_rate': 0.4,
            'attack_params': {
                'mode': 'random_to_random',
                'target_class': 3
            }
        },
        {
            'name': 'CIFAR10 (Dynadetect)', 
            'sample_size': 5000, 
            'metric': 'cosine',
            'modification_method': 'pgd',
            'poison_rate': 0.4,
            'attack_params': {
                'mode': 'random_to_random',
                'target_class': 3
            }
        },
        {
            'name': 'CIFAR10', 
            'sample_size': 5000, 
            'metric': 'cosine',
            'modification_method': 'pgd',
            'poison_rate': 0.4,
            'attack_params': {
                'mode': 'random_to_random',
                'target_class': 3
            }
        },
        #... (Other datasets from the original configuration)
    ],
    'classifiers': ['SVM', 'Random Forest', 'Decision Tree', 'Logistic Regression', 'KNN'],
    'iterations': 1,
    'results_file': 'results-testing.csv',
    'seed': 42,
}

# Initialize config2 as an empty dictionary with 'datasets' key
config2 = {'datasets': []}

# Datasets to configure
dataset_configs = [
    {'name': 'GTSRB', 'type': 'image'},
    {'name': 'GTSRB (Dynadetect)', 'type': 'image'},
    {'name': 'CIFAR10', 'type': 'image'},
    {'name': 'CIFAR10 (Dynadetect)', 'type': 'image'},
    {'name': 'Imagenette', 'type': 'image'},
    {'name': 'Imagenette (Dynadetect)', 'type': 'image'},
    {'name': 'WineQuality', 'type': 'numerical'},
    {'name': 'Diabetes', 'type': 'numerical'}
]

# Modification methods and their associated attack parameters
modification_methods = ['label_flipping', 'pgd', 'gradient_ascent']
poison_rates = [0.0, 0.01, 0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30]
attack_params_variants = [
    {'mode': 'source_to_target', 'source_class': 2, 'target_class': 9},
    {'mode': 'random_to_target', 'target_class': 7},
    {'mode': 'random_to_random', 'target_class': 3}
]

# Generate dataset configurations
for dataset in dataset_configs:
    # Add baseline configuration
    config2['datasets'].append({
        'name': dataset['name'],
        'type': dataset['type'],
        'sample_size': 5000 if dataset['type'] == 'image' else None,
        'metric': 'cosine' if dataset['type'] == 'image' else 'euclidean',
        'modification_method': '',  # No modification for baseline
        'poison_rate': 0.0,
        'attack_params': {}
    })

    # Add configurations for modification methods
    for modification_method in modification_methods:
        for poison_rate in poison_rates:
            attack_params = {}
            if modification_method == 'label_flipping':
                # Only use attack params for label flipping
                for variant in attack_params_variants:
                    config2['datasets'].append({
                        'name': dataset['name'],
                        'type': dataset['type'],
                        'sample_size': 5000 if dataset['type'] == 'image' else None,
                        'metric': 'cosine' if dataset['type'] == 'image' else 'euclidean',
                        'modification_method': modification_method,
                        'poison_rate': poison_rate,
                        'attack_params': variant
                    })
            else:
                # For pgd and gradient_ascent, use empty attack_params
                config2['datasets'].append({
                    'name': dataset['name'],
                    'type': dataset['type'],
                    'sample_size': 5000 if dataset['type'] == 'image' else None,
                    'metric': 'cosine' if dataset['type'] == 'image' else 'euclidean',
                    'modification_method': modification_method,
                    'poison_rate': poison_rate,
                    'attack_params': {}
                })

# Additional config settings
config2['classifiers'] = ['SVM', 'Random Forest', 'Decision Tree', 'Logistic Regression', 'KNN']
config2['iterations'] = 1
config2['results_file'] = 'results-testing.csv'
config2['seed'] = 42

# Assign config2 to config
config = config2 




# Main execution
if __name__ == "__main__":
    set_seed(config['seed'])
    setup_logging()

    results_file = config['results_file']
    if os.path.isfile(results_file):
        base, ext = os.path.splitext(results_file)
        new_name = f"{base}_{int(time.time())}{ext}"
        os.rename(results_file, new_name)
        logging.info(f"Renamed existing results file to {new_name}")

    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Iteration', 'Dataset', 'Classifier', 'Modification_Method', 'Num_Poisoned', 'Poisoned_Classes', 'Flip_Type', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Latency'] + [f'Class_{i}_Accuracy' for i in range(100)]  # Adjust the range as needed
        writer.writerow(header)

        for iteration in range(config['iterations']): 
            for dataset_config in config['datasets']:
                for classifier in config['classifiers']:
                    logging.info(f"Processing configuration: {dataset_config['name']}, {dataset_config['modification_method']}, {classifier}")
                    evaluator = DatasetEvaluator(dataset_config, classifier)
                    try:
                        metrics, num_poisoned, poisoned_classes, flip_type, latency = evaluator.run_evaluation(iteration)

                        row = [str(iteration), dataset_config['name'], classifier, dataset_config['modification_method'], 
                               str(num_poisoned), str(poisoned_classes), str(flip_type), 
                               str(metrics['accuracy']), str(metrics['precision']), str(metrics['recall']), str(metrics['f1']), 
                               str(latency)] + [str(acc) for acc in metrics['class_accuracies']]
                        writer.writerow(row)
                        file.flush()  # Ensure the row is written to the file immediately
                    except Exception as e:
                        logging.error(f"Error processing {dataset_config['name']} with classifier {classifier}: {e}")

    logging.info("Evaluation completed.")

@dataclass
class ExperimentConfig:
    datasets: List[Dict[str, Any]]
    classifiers: List[str]
    iterations: int
    results_file: str
    seed: int

@dataclass
class DatasetConfig:
    name: str
    dataset_type: str
    sample_size: Optional[int]
    metric: str
    modification_method: str
    poison_rate: float
    attack_params: Dict[str, Any]
    mode: str  # Add mode parameter

def main(config: ExperimentConfig, timestamp: str = None) -> None:
    """Main execution function."""
    set_seed(config.seed)
    setup_logging()
    
    results = []
    for dataset_config in config.datasets:
        for classifier in config.classifiers:
            evaluator = DatasetEvaluator(dataset_config, classifier)
            
            for iteration in range(config.iterations):
                metrics, num_poisoned, poisoned_classes, flip_type, latency = evaluator.run_evaluation(iteration, timestamp)
                
                # Convert numpy arrays to lists for JSON serialization
                if isinstance(poisoned_classes, np.ndarray):
                    poisoned_classes = poisoned_classes.tolist()
                
                # Convert all numpy types in metrics to Python native types
                metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else 
                          v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in metrics.items()}
                
                result = {
                    'dataset': dataset_config.name,
                    'classifier': classifier,
                    'iteration': iteration,
                    'modification_method': dataset_config.modification_method,
                    'poison_rate': dataset_config.poison_rate,
                    'num_poisoned': int(num_poisoned) if isinstance(num_poisoned, np.integer) else num_poisoned,
                    'poisoned_classes': poisoned_classes,
                    'flip_type': flip_type,
                    'latency': float(latency),
                    **metrics
                }
                results.append(result)
                
                # Save results after each iteration
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(config.results_file), exist_ok=True)
                    with open(config.results_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    logging.info(f"Saved results to {config.results_file}")
                except Exception as e:
                    logging.error(f"Error saving results to {config.results_file}: {e}")
    
    # Initialize or load existing results
    results = []
    if os.path.exists(config.results_file):
        try:
            with open(config.results_file, 'r') as f:
                results = json.load(f)
            logging.info(f"Loaded {len(results)} existing results from {config.results_file}")
        except json.JSONDecodeError:
            logging.warning(f"Could not load existing results from {config.results_file}. Starting fresh.")
    
    for dataset_config in config.datasets:
        for classifier in config.classifiers:
            evaluator = DatasetEvaluator(dataset_config, classifier)
            
            for iteration in range(config.iterations):
                logging.info(f"\nStarting evaluation for {dataset_config.name}, {classifier}, iteration {iteration}")
                metrics, num_poisoned, poisoned_classes, flip_type, latency = evaluator.run_evaluation(iteration, timestamp)
                
                # Convert numpy arrays to lists for JSON serialization
                if isinstance(poisoned_classes, np.ndarray):
                    poisoned_classes = poisoned_classes.tolist()
                
                # Convert all numpy types in metrics to Python native types
                metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else 
                          v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in metrics.items()}
                
                result = {
                    'dataset': dataset_config.name,
                    'classifier': classifier,
                    'iteration': iteration,
                    'modification_method': dataset_config.modification_method,
                    'poison_rate': dataset_config.poison_rate,
                    'num_poisoned': int(num_poisoned) if isinstance(num_poisoned, np.integer) else num_poisoned,
                    'poisoned_classes': poisoned_classes,
                    'flip_type': flip_type,
                    'latency': float(latency),
                    **metrics
                }
                results.append(result)
                
                # Save results after each iteration
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(config.results_file), exist_ok=True)
                    with open(config.results_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    logging.info(f"Saved results to {config.results_file}")
                except Exception as e:
                    logging.error(f"Error saving results to {config.results_file}: {e}")

if __name__ == "__main__":
    # Example configuration
    config = ExperimentConfig(
        datasets=[
            DatasetConfig(
                name='CIFAR10',
                dataset_type='image',
                sample_size=1000,
                metric='accuracy',
                modification_method='label_flipping',
                poison_rate=0.1,
                attack_params={'mode': 'random_to_random'},
                mode='dynadetect'
            ),
            DatasetConfig(
                name='Diabetes',
                dataset_type='numerical',
                sample_size=None,
                metric='accuracy',
                modification_method='gradient_ascent',
                poison_rate=0.2,
                attack_params={'lr': 0.1, 'iters': 100},
                mode='standard'
            )
        ],
        classifiers=['SVM', 'RF', 'SoftKNN'],
        iterations=3,
        results_file='experiment_results.json',
        seed=42
    )
    
    main(config)

def label_flipping(labels: np.ndarray, mode: str, target_class: Optional[int] = None, 
                  source_class: Optional[int] = None, poison_rate: float = 0.0) -> Tuple[np.ndarray, int]:
    """Apply label flipping attack to the dataset.
    
    Args:
        labels: Original labels
        mode: Type of label flipping ('random_to_random', 'random_to_target', 'source_to_target')
        target_class: Target class for flipping (required for target modes)
        source_class: Source class for flipping (required for source_to_target mode)
        poison_rate: Fraction of labels to flip
        
    Returns:
        Tuple of (modified labels, number of poisoned samples)
    """
    if poison_rate == 0.0:
        return labels, 0
        
    num_samples = len(labels)
    num_to_poison = int(num_samples * poison_rate)
    
    # Create a mask for samples to poison
    poison_mask = np.zeros(num_samples, dtype=bool)
    poison_indices = np.random.choice(num_samples, num_to_poison, replace=False)
    poison_mask[poison_indices] = True
    
    modified_labels = labels.copy()
    
    if mode == 'random_to_random':
        unique_labels = np.unique(labels)
        for idx in poison_indices:
            current_label = labels[idx]
            possible_labels = unique_labels[unique_labels != current_label]
            modified_labels[idx] = np.random.choice(possible_labels)
            
    elif mode == 'random_to_target':
        if target_class is None:
            raise ValueError("Target class must be specified for random_to_target mode")
        modified_labels[poison_mask] = target_class
        
    elif mode == 'source_to_target':
        if target_class is None or source_class is None:
            raise ValueError("Both source and target classes must be specified for source_to_target mode")
        source_samples = labels == source_class
        num_source = min(np.sum(source_samples), num_to_poison)
        source_indices = np.where(source_samples)[0]
        if num_source > 0:
            poison_indices = np.random.choice(source_indices, num_source, replace=False)
            modified_labels[poison_indices] = target_class
            
    else:
        raise ValueError(f"Unknown label flipping mode: {mode}")
    
    return modified_labels, num_to_poison

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
        
    def _train_knn(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray = None) -> None:
        """Train K-Nearest Neighbors classifier."""
        if sample_weights is not None:
            logging.warning("KNN classifier does not support sample weights. Ignoring weights.")
        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(X, y)
    
    def train_model(
        self,
        model: BaseEstimator,
        features: np.ndarray,
        labels: np.ndarray
    ) -> BaseEstimator:
        """Train the model with given features and labels."""
        if self.classifier_name == 'KNN':
            self._train_knn(features, labels)
        else:
            model.fit(features, labels)
        return model
        
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

    def compute_metrics(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        num_classes = len(np.unique(true_labels))
        accuracy, precision, recall, f1, cm, class_accuracies = evaluate_metrics(
            true_labels, predicted_labels, num_classes)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'class_accuracies': class_accuracies
        }

    def log_results(self, metrics: Dict[str, Any], dataset_name: str, modification_method: str, num_poisoned: int, poisoned_classes: List[Any], flip_type: str, latency: float, iteration: int, classifier_name: str, total_images: int, timestamp: str):
        """Log evaluation results."""
        # Log to console
        logging.info(f"Evaluation results for {dataset_name}:")
        logging.info(f"Modification method: {modification_method}")
        logging.info(f"Number of samples poisoned: {num_poisoned}")
        logging.info(f"Poisoned classes: {poisoned_classes}")
        logging.info(f"Flip type: {flip_type}")
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Precision: {metrics['precision']:.4f}")
        logging.info(f"Recall: {metrics['recall']:.4f}")
        logging.info(f"F1 Score: {metrics['f1']:.4f}")
        logging.info(f"Inference Latency: {latency:.4f} seconds")
        
        # Create results directory if it doesn't exist
        base_dir = os.getcwd()  # Use current working directory
        results_dir = os.path.join(base_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Prepare CSV data
        csv_file = os.path.join(results_dir, f'experiment_results_{timestamp}.csv')
        file_exists = os.path.exists(csv_file)
        
        # Format class accuracies
        class_accuracies = metrics['class_accuracies']
        class_acc_dict = {f'Class_{i}_Accuracy': acc for i, acc in enumerate(class_accuracies)}
        
        # Prepare row data
        row_data = {
            'Iteration': iteration,
            'Dataset': dataset_name,
            'Classifier': classifier_name,
            'Modification_Method': modification_method,
            'Total_Images': total_images,
            'Num_Poisoned': num_poisoned,
            'Poisoned_Classes': ','.join(map(str, poisoned_classes)),
            'Flip_Type': flip_type,
            'Flip_Counts': num_poisoned,  # This could be modified if we track individual flip counts
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'Latency': latency,
            **class_acc_dict  # Add all class accuracies
        }
        
        # Write to CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
            
        logging.info(f"Results saved to: {os.path.abspath(csv_file)}")

class DatasetEvaluator:
    """Evaluates datasets with different models and poisoning methods."""
    
    def __init__(self, config: 'DatasetConfig', classifier_name: str):
        """Initialize evaluator with configuration."""
        self.dataset_name = config.name
        self.dataset_type = config.dataset_type
        self.sample_size = config.sample_size
        self.modification_method = config.modification_method
        self.attack_params = config.attack_params
        self.poison_rate = config.poison_rate
        self.metric = config.metric
        self.classifier_name = classifier_name
        self.mode = config.mode
        self.device = device
        self.num_poisoned = 0
        
        logging.info(f"Initializing DatasetEvaluator for {self.dataset_name}")
        logging.info(f"Dataset type: {self.dataset_type}")
        logging.info(f"Sample size: {self.sample_size}")
        logging.info(f"Modification method: {self.modification_method}")
        logging.info(f"Poison rate: {self.poison_rate}")
        logging.info(f"Classifier: {classifier_name}")
        logging.info(f"Mode: {self.mode}")
        logging.info(f"Using device: {self.device}")
        
        # Initialize components
        self.dataset_handler = DatasetHandler(
            dataset_name=self.dataset_name,
            dataset_type=self.dataset_type,
            sample_size=self.sample_size
        )
        
        self.model_factory = ModelFactory()
        self.poisoning_methods = PoisoningMethods()
        
        # Initialize DynaDetect trainer if needed
        if "Dynadetect" in self.dataset_name:
            self.trainer = DynaDetectTrainer(
                n_components=100,  # Adjusted for GPU memory
                contamination=0.1
            )
        else:
            self.trainer = None
        
        if torch.cuda.is_available():
            logging.info(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    def run_evaluation(self, iteration: int = 0, timestamp: str = None) -> Tuple[Dict[str, Any], int, List[int], str, float]:
        """Run evaluation pipeline."""
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
        logging.info(f"\nStarting evaluation iteration {iteration}")
        
        # Load and preprocess dataset
        train_dataset, val_dataset = self.dataset_handler.load_data()
        
        # Get total number of images
        total_images = len(train_dataset) + len(val_dataset)
        
        # Process training set
        train_features, train_labels = self.process_training_set(train_dataset)
        
        # Create and train model
        model = self.model_factory.create_model(self.classifier_name, mode=self.mode)
        
        # Initialize latency
        start_time = time.time()
        
        if "Dynadetect" in self.dataset_name:
            metrics = self._train_with_dynadetect(model, train_features, train_labels)
        else:
            # Standard training
            model.fit(train_features, train_labels)
            
            # Process validation set
            val_features, val_labels = process_validation_set(val_dataset, self.dataset_name)
            
            # Make predictions
            predictions = model.predict(val_features)
            
            # Compute metrics
            accuracy, precision, recall, f1, cm, class_accuracies = evaluate_metrics(val_labels, predictions, len(np.unique(train_labels)))
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm,
                'class_accuracies': class_accuracies
            }
        
        # Calculate total latency
        latency = time.time() - start_time
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info(f"Final GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        # Log results
        evaluator = ModelEvaluator(classifier_name=self.classifier_name, mode=self.mode)
        evaluator.log_results(
            metrics=metrics,
            dataset_name=self.dataset_name,
            modification_method=self.modification_method,
            num_poisoned=self.num_poisoned,
            poisoned_classes=[],
            flip_type='none',
            latency=latency,
            iteration=iteration,
            classifier_name=self.classifier_name,
            total_images=total_images,
            timestamp=timestamp
        )
        
        return metrics, self.num_poisoned, [], self.modification_method, latency
    
    def _train_with_dynadetect(self, model, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """DynaDetect specific training logic."""
        logging.info("Training with DynaDetect")
        
        # Apply DynaDetect feature transformation
        transformed_features, sample_weights = self.trainer.fit_transform(features, labels)
        
        # Train model with sample weights
        if hasattr(model, 'fit_transform'):
            model.fit(transformed_features, labels, sample_weight=sample_weights)
        else:
            model.fit(transformed_features, labels)
        
        # Get anomaly scores
        anomaly_scores = self.trainer.predict_anomaly_scores(features)
        
        metrics = {
            'anomaly_scores': anomaly_scores,
            'sample_weights': sample_weights
        }
        
        return metrics
    
    def process_training_set(self, train_dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Process training dataset."""
        batch_size = min(32, len(train_dataset))  # Adjust batch size based on dataset size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        features, labels = extract_features(train_loader, self.dataset_name)
        
        if self.modification_method:
            logging.info(f"Applying {self.modification_method} modification")
            if self.modification_method == 'label_flipping':
                labels, self.num_poisoned = label_flipping(
                    labels=labels,
                    mode=self.attack_params.get('mode', 'random_to_random'),
                    target_class=self.attack_params.get('target_class'),
                    source_class=self.attack_params.get('source_class'),
                    poison_rate=self.poison_rate
                )
            elif self.modification_method in ['pgd', 'gradient_ascent']:
                # Convert to torch tensors for GPU operations
                features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
                labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
                
                # Create temporary SoftKNN model for attack
                temp_model = SoftKNN(n_neighbors=5).to(self.device)
                temp_model.fit(features_tensor, labels_tensor)
                
                if self.modification_method == 'pgd':
                    features_tensor, indices = pgd_attack(
                        temp_model, features_tensor, labels_tensor,
                        eps=self.attack_params.get('eps', 0.1),
                        alpha=self.attack_params.get('alpha', 0.01),
                        iters=self.attack_params.get('iters', 40),
                        poison_rate=self.poison_rate
                    )
                else:  # gradient_ascent
                    features_tensor, indices = gradient_ascent(
                        temp_model, features_tensor, labels_tensor,
                        lr=self.attack_params.get('lr', 0.1),
                        iters=self.attack_params.get('iters', 100),
                        poison_rate=self.poison_rate
                    )
                
                # Move back to CPU
                features = features_tensor.cpu().numpy()
                del temp_model
                torch.cuda.empty_cache()
        
        return features, labels
    
    def process_validation_set(self, val_dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Process validation dataset."""
        val_dataset = process_validation_set(val_dataset, self.dataset_name)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        return extract_features(val_loader, self.dataset_name)

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
        self.feature_selector = None
        self.anomaly_detector = None
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
        logging.info("Starting DynaDetect fit_transform")
        if torch.cuda.is_available():
            logging.info(f"GPU Memory before processing: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        # Convert to torch tensors for GPU operations
        features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # 1. Feature Selection
        # Use CPU for feature selection as sklearn doesn't support GPU
        features_cpu = features_tensor.cpu().numpy()
        labels_cpu = labels_tensor.cpu().numpy()
        
        # Select features using mutual information
        self.feature_selector = SelectKBest(mutual_info_classif, k=min(self.n_components, features.shape[1]))
        features_selected = self.feature_selector.fit_transform(features_cpu, labels_cpu)
        
        # 2. Robust Scaling
        features_scaled = self.scaler.fit_transform(features_selected)
        
        # 3. Anomaly Detection
        self.anomaly_detector = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        # Fit anomaly detector and get decision scores
        self.anomaly_detector.fit(features_scaled)
        decision_scores = self.anomaly_detector.score_samples(features_scaled)
        
        # Convert scores to sample weights (higher weight = more likely to be clean)
        sample_weights = np.clip(decision_scores - decision_scores.min(), 0, None)
        sample_weights = sample_weights / sample_weights.max()  # Normalize to [0,1]
        
        if torch.cuda.is_available():
            logging.info(f"GPU Memory after processing: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            torch.cuda.empty_cache()
        
        return features_scaled, sample_weights
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform new features using fitted components."""
        if self.feature_selector is None or self.scaler is None:
            raise RuntimeError("DynaDetect components not fitted. Call fit_transform first.")
        
        # Select features and scale
        features_selected = self.feature_selector.transform(features)
        features_scaled = self.scaler.transform(features_selected)
        
        return features_scaled
    
    def predict_anomaly_scores(self, features: np.ndarray) -> np.ndarray:
        """Predict anomaly scores for new features."""
        if self.anomaly_detector is None:
            raise RuntimeError("Anomaly detector not fitted. Call fit_transform first.")
        
        # Transform features first
        features_transformed = self.transform(features)
        
        # Get anomaly scores
        return self.anomaly_detector.score_samples(features_transformed)

# Define Numerical Dataset Class
class NumericalDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.targets = self.y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
