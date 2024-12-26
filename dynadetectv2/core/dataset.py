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


class DatasetHandler:
    """Handler for dataset operations."""
    
    def __init__(self, dataset_name):
        """Initialize dataset handler."""
        self.dataset_name = dataset_name
        self.root_dir = '/home/brian/Notebooks/ddv2-ws'
        self.transform = self.get_transform()
        self.pca = None  # Store PCA object for consistent dimensionality
        logging.info(f"Initialized DatasetHandler for {dataset_name}")
        logging.info(f"Dataset type: {self.get_dataset_type()}")
        
        # Set up GTSRB test set structure if needed
        if dataset_name == "GTSRB":
            self.setup_gtsrb_test()

    def setup_gtsrb_test(self):
        """Create class subdirectories for GTSRB test set if they don't exist."""
        test_dir = os.path.join(self.root_dir, '.datasets/gtsrb/GTSRB/Final_Test/Images')
        if not os.path.exists(test_dir):
            return
        
        # Check if subdirectories already exist
        subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        if subdirs:
            return
        
        # Create subdirectories for each class (0-42)
        for i in range(43):
            class_dir = os.path.join(test_dir, f"{i:05d}")
            os.makedirs(class_dir, exist_ok=True)
        
        # Move images to appropriate subdirectories based on filename
        for filename in os.listdir(test_dir):
            if not filename.endswith('.ppm'):
                continue
            
            # Extract class from filename (first 5 digits)
            class_id = filename[:5]
            src = os.path.join(test_dir, filename)
            dst = os.path.join(test_dir, class_id, filename)
        
            try:
                os.rename(src, dst)
            except OSError:
                continue

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

    def get_train_dataset(self, sample_size=None):
        """Get training dataset."""
        if self.get_dataset_type() == 'numerical':
            dataset = self.load_numerical_dataset('train', sample_size)
            
        if self.dataset_name == 'CIFAR100':
            dataset = datasets.CIFAR100(root=os.path.join(self.root_dir, '.datasets/cifar-100'), train=True, download=True, transform=self.transform)
            if sample_size is not None and sample_size > 0 and sample_size < len(dataset):
                indices = self.get_stratified_indices(dataset, sample_size)
                dataset = Subset(dataset, indices)
                # Store targets for subset
                dataset.targets = np.array(dataset.dataset.targets)[dataset.indices]
        elif self.dataset_name == "ImageNette":
            dataset = datasets.ImageFolder(root=os.path.join(self.root_dir, '.datasets/imagenette/train'), transform=self.transform)
            if sample_size is not None and sample_size > 0 and sample_size < len(dataset):
                indices = self.get_stratified_indices(dataset, sample_size)
                dataset = Subset(dataset, indices)
                # Store targets for subset
                dataset.targets = np.array([dataset.dataset.targets[i] for i in indices])
        elif self.dataset_name == "GTSRB":
            # For GTSRB training, each class has its own folder (00000, 00001, etc.)
            dataset = datasets.ImageFolder(root=os.path.join(self.root_dir, '.datasets/gtsrb/GTSRB/Training'), transform=self.transform)
            if sample_size is not None and sample_size > 0 and sample_size < len(dataset):
                indices = self.get_stratified_indices(dataset, sample_size)
                dataset = Subset(dataset, indices)
                # Store targets for subset
                dataset.targets = np.array([dataset.dataset.targets[i] for i in indices])
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        return dataset
    
    def get_val_dataset(self, sample_size=None):
        """Get validation dataset."""
        if self.get_dataset_type() == 'numerical':
            dataset = self.load_numerical_dataset('val', sample_size)
            
        if self.dataset_name == 'CIFAR100':
            dataset = datasets.CIFAR100(root=os.path.join(self.root_dir, '.datasets/cifar-100'), train=False, download=True, transform=self.transform)
            if sample_size is not None and sample_size > 0 and sample_size < len(dataset):
                indices = self.get_stratified_indices(dataset, sample_size)
                dataset = Subset(dataset, indices)
                # Store targets for subset
                dataset.targets = np.array(dataset.dataset.targets)[dataset.indices]
        elif self.dataset_name == "ImageNette":
            dataset = datasets.ImageFolder(root=os.path.join(self.root_dir, '.datasets/imagenette/val'), transform=self.transform)
            if sample_size is not None and sample_size > 0 and sample_size < len(dataset):
                indices = self.get_stratified_indices(dataset, sample_size)
                dataset = Subset(dataset, indices)
                # Store targets for subset
                dataset.targets = np.array([dataset.dataset.targets[i] for i in indices])
        elif self.dataset_name == "GTSRB":
            # For GTSRB test, all images are in a flat directory with a CSV mapping
            test_dir = os.path.join(self.root_dir, '.datasets/gtsrb/GTSRB/Final_Test/Images')
            dataset = datasets.ImageFolder(root=test_dir, transform=self.transform)
            if sample_size is not None and sample_size > 0 and sample_size < len(dataset):
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
    
    def extract_features(self, dataset):
        """Extract features from dataset using PCA."""
        # Convert dataset to DataLoader
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Extract raw features
        all_features = []
        all_labels = []
        
        for images, labels in data_loader:
            # Flatten the images
            features = images.view(images.size(0), -1).numpy()
            all_features.append(features)
            all_labels.append(labels.numpy())
        
        features = np.concatenate(all_features)
        labels = np.concatenate(all_labels)
        
        logging.info(f"Extracting features from {len(features)} samples...")
        
        # Initialize PCA if not already done
        if self.pca is None:
            n_components = min(100, len(features) - 1)  # Ensure n_components is less than n_samples
            logging.info(f"Fitting PCA to reduce dimensions from {features.shape[1]} to {n_components}...")
            self.pca = PCA(n_components=n_components)
            features = self.pca.fit_transform(features)
        else:
            # Use the same PCA transformation for validation set
            features = self.pca.transform(features)
        
        logging.info(f"Feature extraction completed. Final feature shape: {features.shape}")
        
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
        labels = np.array(labels)  # Ensure labels is a numpy array
        unique_labels = np.unique(labels)
        original_labels = labels.copy()
        num_to_poison = int(len(labels) * poison_rate)
        poison_indices = np.random.choice(len(labels), num_to_poison, replace=False)

        new_labels = labels.copy()
        poisoned_classes = set()

        if mode == 'random_to_random':
            for idx in poison_indices:
                new_label = np.random.choice([l for l in unique_labels if l != labels[idx]])
                new_labels[idx] = new_label
                poisoned_classes.add(int(labels[idx]))
                poisoned_classes.add(int(new_label))
        elif mode == 'random_to_target':
            if target_class is None:
                raise ValueError("target_class must be specified for 'random_to_target' mode.")
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
            source_indices = np.where(labels == source_class)[0]
            num_to_flip = min(len(source_indices), num_to_poison)
            flip_indices = np.random.choice(source_indices, num_to_flip, replace=False)
            new_labels[flip_indices] = target_class
            poisoned_classes.add(int(source_class))
            poisoned_classes.add(int(target_class))

            logging.info(f"Source class: {source_class}, Target class: {target_class}")
            logging.info(f"Number of samples in source class: {len(source_indices)}")
            logging.info(f"Number of samples to poison based on rate: {num_to_poison}")
            logging.info(f"Actual number of labels flipped: {num_to_flip}")
        else:
            raise ValueError("Invalid mode specified for label flipping.")

        num_poisoned = np.sum(original_labels != new_labels)
        logging.info(f"Number of labels flipped: {num_poisoned}")

        attack_params = {
            'type': mode,
            'target_class': target_class,
            'source_class': source_class,
            'poison_rate': poison_rate,
            'num_poisoned': int(num_poisoned),
            'poisoned_classes': list(poisoned_classes)
        }

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
