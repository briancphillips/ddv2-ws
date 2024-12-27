"""Dataset handling for DynaDetect v2."""

from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
import logging
from dataclasses import dataclass
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import shutil
import requests
import zipfile
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class DatasetSpecs:
    """Specifications for a dataset."""
    name: str
    dataset_type: str
    default_sample_size: int
    min_sample_size: int
    max_sample_size: int
    min_samples_per_class: int
    num_classes: int


class NumericalDataset(Dataset):
    """Dataset class for numerical data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """Initialize dataset.
        
        Args:
            X: Input features
            y: Labels
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.targets = self.y
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class GTSRBDataset(Dataset):
    """GTSRB dataset class using torchvision."""
    
    def __init__(self, root_dir, train=True, transform=None, val_split=0.2):
        """Initialize GTSRB dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            train: Whether to use training or validation split
            transform: Optional transform to be applied to images
            val_split: Fraction of training data to use for validation
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        
        # Use torchvision's GTSRB dataset
        base_dataset = datasets.GTSRB(
            root=root_dir,
            split='train',  # Always load training data for train/val split
            transform=transform,
            download=True
        )
        
        if train is not None:  # Handle train/val split
            # Get all targets
            all_targets = []
            for _, target in base_dataset:
                all_targets.append(target)
            all_targets = torch.tensor(all_targets)
            
            # Split into train and validation sets
            train_size = int((1 - val_split) * len(base_dataset))
            val_size = len(base_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                base_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Set the dataset based on train parameter
            self.dataset = train_dataset if train else val_dataset
            
            # Store targets for the split
            indices = self.dataset.indices
            self.targets = [all_targets[i].item() for i in indices]
        else:  # Use test set
            test_dataset = datasets.GTSRB(
                root=root_dir,
                split='test',
                transform=transform,
                download=True
            )
            self.dataset = test_dataset
            # Get all targets for test set
            all_targets = []
            for _, target in test_dataset:
                all_targets.append(target)
            self.targets = all_targets
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        return img, target


class DatasetHandler:
    """Handler for dataset operations."""
    
    def __init__(self, dataset_config):
        """Initialize dataset handler."""
        self.config = dataset_config
        self.dataset_name = dataset_config.name
        self.sample_size = dataset_config.sample_size
        self.root_dir = '/home/brian/Notebooks/ddv2-ws'
        self.transform = self.get_transform()
        self.pca = None  # Store PCA object for consistent dimensionality
        self._feature_cache = {}  # Cache for extracted features
        self._label_flip_cache = {}  # Cache for flipped labels
        logging.info(f"Initialized DatasetHandler for {self.dataset_name}")
        logging.info(f"Dataset type: {self.get_dataset_type()}")
        logging.info(f"Sample size: {self.sample_size}")
        
        # Set up GTSRB dataset if needed
        if self.dataset_name == "GTSRB":
            self.setup_gtsrb()

    def setup_gtsrb(self):
        """Set up GTSRB dataset using torchvision."""
        # Create dataset directory
        dataset_dir = os.path.join(self.root_dir, '.datasets/gtsrb')
        os.makedirs(dataset_dir, exist_ok=True)
        
        # The dataset will be downloaded automatically when creating GTSRBDataset
        logging.info("GTSRB dataset will be downloaded through torchvision if needed")

    def get_dataset_type(self):
        """Get dataset type."""
        if self.dataset_name in ['Diabetes', 'WineQuality']:
            return 'numerical'
        return 'image'
    
    def get_transform(self):
        """Get transform for dataset."""
        if self.get_dataset_type() == 'numerical':
            return None
            
        if "ImageNette" in self.dataset_name:
            return transforms.Compose([
                transforms.Resize((32, 32)),  # Smaller size for testing
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif "CIFAR" in self.dataset_name:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        elif "GTSRB" in self.dataset_name:
            return transforms.Compose([
                transforms.Resize((32, 32)),  # Standardize GTSRB size
                transforms.ToTensor(),
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def get_stratified_indices(self, dataset, sample_size):
        """Get stratified sample indices ensuring balanced class representation."""
        targets = self.get_targets(dataset)
        unique_labels = np.unique(targets)
        indices = []
        samples_per_class = sample_size // len(unique_labels)
        remaining_samples = sample_size % len(unique_labels)
        
        for label in unique_labels:
            label_indices = np.where(targets == label)[0]
            if len(label_indices) > 0:
                selected_indices = np.random.choice(label_indices, min(samples_per_class, len(label_indices)), replace=False)
                indices.extend(selected_indices)
        
        # Add remaining samples randomly
        if remaining_samples > 0:
            remaining_indices = np.setdiff1d(np.arange(len(targets)), indices)
            if len(remaining_indices) > 0:
                additional_indices = np.random.choice(remaining_indices, min(remaining_samples, len(remaining_indices)), replace=False)
                indices.extend(additional_indices)
        
        return np.array(indices, dtype=np.int64)

    def get_train_dataset(self):
        """Get training dataset."""
        return self._get_dataset('train', self.sample_size)
        
    def get_val_dataset(self):
        """Get validation dataset."""
        if self.dataset_name == 'GTSRB':
            # For GTSRB, use 20% of train size or minimum 1000 samples
            val_size = max(self.sample_size // 5, 1000)
        else:
            # For other datasets, use 20% of train size
            val_size = self.sample_size // 5 if self.sample_size else None
        
        return self._get_dataset('val', val_size)
        
    def _get_dataset(self, split: str, sample_size: Optional[int] = None) -> Dataset:
        """Internal method to get dataset with sample size."""
        if self.get_dataset_type() == 'numerical':
            return self.load_numerical_dataset(split, sample_size)
            
        if self.dataset_name == 'CIFAR100':
            # Create a subset of indices first
            is_train = (split == 'train')
            total_samples = 50000 if is_train else 10000
            
            dataset = datasets.CIFAR100(
                root=os.path.join(self.root_dir, '.datasets/cifar-100'),
                train=is_train,
                download=True,
                transform=self.transform
            )
            
            if sample_size is not None and sample_size > 0:
                indices = self.get_stratified_indices(dataset, sample_size)
                dataset = Subset(dataset, indices)
                # Store targets for subset
                dataset.targets = np.array(dataset.dataset.targets)[dataset.indices]
                
        elif self.dataset_name == "ImageNette":
            split_dir = 'train' if split == 'train' else 'val'
            dataset = datasets.ImageFolder(
                root=os.path.join(self.root_dir, f'.datasets/imagenette/{split_dir}'),
                transform=self.transform
            )
            
            if sample_size is not None and sample_size > 0 and sample_size < len(dataset):
                indices = self.get_stratified_indices(dataset, sample_size)
                dataset = Subset(dataset, indices)
                # Store targets for subset
                dataset.targets = np.array([dataset.dataset.targets[i] for i in indices])
                
        elif self.dataset_name == "GTSRB":
            # Use our custom GTSRB dataset implementation
            dataset = GTSRBDataset(
                root_dir=os.path.join(self.root_dir, '.datasets/gtsrb'),
                train=(split == 'train'),
                transform=self.transform
            )
            
            if sample_size is not None and sample_size > 0:
                if split != 'train':
                    # For test set, ensure at least 1000 samples
                    sample_size = max(sample_size, 1000)
                indices = self.get_stratified_indices(dataset, sample_size)
                dataset = Subset(dataset, indices)
                # Store targets for subset
                dataset.targets = np.array([dataset.dataset.targets[i] for i in indices])
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
        return dataset

    def load_numerical_dataset(self, split='train', sample_size=None):
        """Load numerical dataset."""
        if self.dataset_name == 'Diabetes':
            data = pd.read_csv(os.path.join(self.root_dir, "diabetes.csv"))
            X = data.drop("Outcome", axis=1).values.astype(np.float32)
            y = data["Outcome"].values.astype(np.int64)
        elif self.dataset_name == 'WineQuality':
            data = pd.read_csv(os.path.join(self.root_dir, "winequality-red.csv"), sep=";")
            X = data.drop("quality", axis=1).values.astype(np.float32)
            y = data["quality"].values.astype(np.int64)
            y = y - y.min()  # Shift labels to start from 0
        else:
            raise ValueError(f"Unsupported numerical dataset: {self.dataset_name}")
        
        dataset = NumericalDataset(X, y)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        if split == 'train':
            dataset = train_dataset
        else:
            dataset = val_dataset
        
        if sample_size is not None:
            indices = self.get_stratified_indices(dataset, sample_size)
            dataset = Subset(dataset, indices)
            
        return dataset
    
    def extract_features(self, dataset: torch.utils.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from dataset.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Tuple of (features, labels)
        """
        # Generate cache key based on dataset object id and length
        cache_key = (id(dataset), len(dataset))
        if cache_key in self._feature_cache:
            logging.info("Using cached features")
            return self._feature_cache[cache_key]
            
        features = []
        labels = []
        total_samples = len(dataset)
        
        logging.info(f"Extracting features from {total_samples} samples...")
        
        # Extract raw features
        for data, label in dataset:
            if isinstance(data, torch.Tensor):
                features.append(data.cpu().numpy().flatten())
            else:
                features.append(data.flatten())
            labels.append(label)
            
        features = np.array(features)
        labels = np.array(labels)
        
        # Apply PCA if needed
        if self.dataset_name == 'image' and features.shape[1] > 100:
            logging.info(f"Fitting PCA to reduce dimensions from {features.shape[1]} to 100...")
            pca = PCA(n_components=100)
            features = pca.fit_transform(features)
            
        logging.info(f"Feature extraction completed. Final feature shape: {features.shape}")
        
        # Cache the results
        self._feature_cache[cache_key] = (features, labels)
        return features, labels

    def get_targets(self, dataset):
        """Get targets from dataset."""
        if isinstance(dataset, Subset):
            if hasattr(dataset, 'targets'):
                return dataset.targets
            else:
                base_targets = np.array(dataset.dataset.targets)
                return base_targets[dataset.indices]
        elif isinstance(dataset, ConcatDataset):
            all_targets = []
            for d in dataset.datasets:
                all_targets.extend(self.get_targets(d))
            return np.array(all_targets)
        elif hasattr(dataset, 'targets'):
            return np.array(dataset.targets)
        elif hasattr(dataset, 'tensors'):  # For TensorDataset
            return dataset.tensors[1].numpy()
        else:
            raise ValueError("Dataset type not supported for extracting targets")
            
    def label_flipping(self, labels, mode, target_class=None, source_class=None, poison_rate=0.0):
        """Apply label flipping attack to the dataset."""
        # Generate cache key
        cache_key = (hash(str(labels.tolist())), mode, target_class, source_class, poison_rate)
        if cache_key in self._label_flip_cache:
            logging.info("Using cached flipped labels")
            return self._label_flip_cache[cache_key]
            
        labels = np.array(labels)  # Ensure labels is a numpy array
        unique_labels = np.unique(labels)
        original_labels = labels.copy()
        num_to_poison = int(len(labels) * poison_rate)
        new_labels = labels.copy()
        poisoned_classes = set()

        if mode == 'random_to_random':
            poison_indices = np.random.choice(len(labels), num_to_poison, replace=False)
            for idx in poison_indices:
                new_label = np.random.choice([l for l in unique_labels if l != labels[idx]])
                new_labels[idx] = new_label
                poisoned_classes.add(int(labels[idx]))
                poisoned_classes.add(int(new_label))
        elif mode == 'random_to_target':
            if target_class is None:
                raise ValueError("target_class must be specified for 'random_to_target' mode.")
            poison_indices = np.random.choice(len(labels), num_to_poison, replace=False)
            for idx in poison_indices:
                if labels[idx] != target_class:
                    new_labels[idx] = target_class
                    poisoned_classes.add(int(labels[idx]))
                    poisoned_classes.add(int(target_class))
        elif mode == 'source_to_target':
            if source_class is None or target_class is None:
                raise ValueError("Both source_class and target_class must be specified for 'source_to_target' mode.")
            if source_class == target_class:
                raise ValueError("source_class and target_class must be different.")
                
            # Get indices of samples in source class
            source_indices = np.where(labels == source_class)[0]
            num_source_samples = len(source_indices)
            
            # Calculate the number of samples to flip, limited by available samples
            # Always keep at least one sample in the source class to prevent class elimination
            num_to_flip = min(num_source_samples - 1, num_to_poison) if num_source_samples > 0 else 0
            
            # Log detailed information about the flipping operation
            logging.info(f"Label flipping details:")
            logging.info(f"- Source class: {source_class}")
            logging.info(f"- Target class: {target_class}")
            logging.info(f"- Available samples in source class: {num_source_samples}")
            logging.info(f"- Requested samples to poison: {num_to_poison}")
            logging.info(f"- Actual samples to flip: {num_to_flip}")
            logging.info(f"- Samples remaining in source class: {num_source_samples - num_to_flip}")
            
            # Only proceed if we have samples to flip
            if num_to_flip > 0:
                # Randomly select indices to flip
                flip_indices = np.random.choice(source_indices, num_to_flip, replace=False)
                new_labels[flip_indices] = target_class
                poisoned_classes.add(int(source_class))
                poisoned_classes.add(int(target_class))
                logging.info(f"Successfully flipped {num_to_flip} labels from class {source_class} to {target_class}")
            else:
                logging.warning(f"No labels were flipped: insufficient samples in source class {source_class}")
        else:
            raise ValueError("Invalid mode specified for label flipping.")

        num_poisoned = np.sum(original_labels != new_labels)
        logging.info(f"Total number of labels flipped: {num_poisoned}")

        attack_params = {
            'type': mode,
            'target_class': target_class,
            'source_class': source_class,
            'poison_rate': poison_rate,
            'num_poisoned': int(num_poisoned),
            'poisoned_classes': list(poisoned_classes)
        }
        
        # Cache the results
        self._label_flip_cache[cache_key] = (new_labels.astype(np.int64), attack_params)
        return new_labels.astype(np.int64), attack_params

    def apply_label_flipping(self, dataset, poison_rate, flip_type='random_to_random'):
        """Apply label flipping attack to the dataset.
        
        Args:
            dataset: Dataset to apply label flipping to
            poison_rate: Percentage of labels to flip (0.0 to 1.0)
            flip_type: Type of flipping - 'random_to_random', 'random_to_target', or 'source_to_target'
        """
        # Get original targets
        targets = self.get_targets(dataset)
        
        # Set up parameters for label flipping
        target_class = None
        source_class = None
        
        # For random_to_target and source_to_target, we'll use class 0 as target
        # and class 1 as source for simplicity in testing
        if flip_type == 'random_to_target':
            target_class = 0
        elif flip_type == 'source_to_target':
            target_class = 0
            source_class = 1
        
        # Apply label flipping
        new_targets, _ = self.label_flipping(
            labels=targets,
            mode=flip_type,
            target_class=target_class,
            source_class=source_class,
            poison_rate=poison_rate
        )
        
        # Create new dataset with flipped labels
        if isinstance(dataset, Subset):
            # Update both the subset and base dataset targets
            dataset.targets = new_targets
            base_targets = np.array(dataset.dataset.targets)
            base_targets[dataset.indices] = new_targets
            dataset.dataset.targets = base_targets.tolist()  # Convert back to list for CIFAR100
        else:
            dataset.targets = new_targets.tolist()  # Convert back to list for CIFAR100
            
        return dataset

    def get_train_data(self):
        """Get training data and labels."""
        train_dataset = self.get_train_dataset(sample_size=self.config.sample_size)
        return self.extract_features(train_dataset)

    def get_test_data(self):
        """Get test data and labels."""
        test_dataset = self.get_val_dataset(sample_size=self.config.sample_size)
        return self.extract_features(test_dataset)

    def apply_label_flipping_to_labels(self, labels: np.ndarray, poison_rate: float, flip_type: str = 'random_to_random') -> np.ndarray:
        """Apply label flipping directly to labels.
        
        Args:
            labels: Array of labels to modify
            poison_rate: Proportion of labels to flip
            flip_type: Type of flipping to apply
            
        Returns:
            Modified labels array
        """
        num_samples = len(labels)
        num_classes = len(np.unique(labels))
        num_to_poison = int(num_samples * poison_rate)
        
        if num_to_poison == 0:
            return labels
            
        # Select indices to poison
        indices_to_poison = np.random.choice(
            num_samples, 
            size=num_to_poison, 
            replace=False
        )
        
        # Create copy of labels
        poisoned_labels = labels.copy()
        
        if flip_type == 'random_to_random':
            # For each selected index, randomly change its label
            for idx in indices_to_poison:
                current_label = poisoned_labels[idx]
                # Get all labels except current
                possible_labels = [l for l in range(num_classes) if l != current_label]
                # Randomly select new label
                new_label = np.random.choice(possible_labels)
                poisoned_labels[idx] = new_label
                
        elif flip_type == 'next_class':
            # Change each selected label to next class (cycling back to 0)
            poisoned_labels[indices_to_poison] = (
                poisoned_labels[indices_to_poison] + 1
            ) % num_classes
            
        else:
            raise ValueError(f"Unknown flip type: {flip_type}")
            
        logging.info(f"Number of labels flipped: {num_to_poison}")
        return poisoned_labels
