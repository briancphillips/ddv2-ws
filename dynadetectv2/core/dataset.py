"""Dataset handling and feature extraction."""

import os
import logging
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
from typing import Tuple, Optional, Dict
from ..config import DatasetConfig
import torchvision
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader, Subset
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor(nn.Module):
    """Feature extractor using pretrained models."""
    def __init__(self, dataset_name: str):
        super(FeatureExtractor, self).__init__()
        self.dataset_name = dataset_name
        self.logger = logging.getLogger(__name__)
        
        # Enable autocast for mixed precision
        self.use_amp = torch.cuda.is_available()
        
        # Initialize model based on dataset
        if dataset_name == "CIFAR100":
            import torchvision.models as models
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            # Remove the final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        elif dataset_name == "GTSRB":
            import torchvision.models as models
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            # Remove the final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        else:  # ImageNette
            import torchvision.models as models
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            # Remove the final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Move model to device and set to eval mode
        self.model = self.model.to(device)
        self.model.eval()
        
        # Log model information
        self.logger.info(f"Initialized feature extractor for {dataset_name}")
        if torch.cuda.is_available():
            self.logger.info(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Features tensor of shape (batch_size, feature_dim)
        """
        if self.use_amp:
            with torch.cuda.amp.autocast():
                features = self.model(x)
                if isinstance(features, tuple):
                    features = features[0]
                features = features.reshape(features.size(0), -1)
                return features
        else:
            features = self.model(x)
            if isinstance(features, tuple):
                features = features[0]
            features = features.reshape(features.size(0), -1)
            return features

class DatasetHandler:
    """Handles dataset loading and feature extraction."""
    
    def __init__(self, config: DatasetConfig):
        """Initialize dataset handler.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.dataset_name = config.name
        self.batch_size = config.batch_size
        self.feature_extractor = None
        self._feature_cache = {}
        self._label_flip_cache = {}  # Initialize label flip cache
        self.logger = logging.getLogger(__name__)
        self.data_dir = ".datasets"  # Default data directory
        os.makedirs(self.data_dir, exist_ok=True)  # Ensure data directory exists
        
        # Initialize feature extractor
        self._init_feature_extractor()

    def _init_feature_extractor(self):
        """Initialize feature extractor."""
        if self.dataset_name == "CIFAR100":
            self.feature_extractor = FeatureExtractor("CIFAR100").to(device)
        elif self.dataset_name == "GTSRB":
            self.feature_extractor = FeatureExtractor("GTSRB").to(device)
        else:  # ImageNette
            self.feature_extractor = FeatureExtractor("ImageNette").to(device)

    def get_transform(self) -> transforms.Compose:
        """Get transforms for dataset."""
        if self.dataset_name == "CIFAR100":
            if self.is_train:
                return transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])
            else:
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])
        elif self.dataset_name == "GTSRB":
            if self.is_train:
                return transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation(15),
                    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                ])
            else:
                return transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                ])
        else:  # ImageNette
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def get_train_dataset(self) -> Dataset:
        """Get training dataset."""
        self.is_train = True
        dataset = None
        
        if self.dataset_name == "CIFAR100":
            try:
                dataset = datasets.CIFAR100(root=os.path.join(self.data_dir, "cifar100"), 
                                       train=True, 
                                       download=True,
                                       transform=self.get_transform())
            except Exception as e:
                # If loading fails, try to clean up and redownload
                shutil.rmtree(os.path.join(self.data_dir, "cifar100"), ignore_errors=True)
                dataset = datasets.CIFAR100(root=os.path.join(self.data_dir, "cifar100"), 
                                       train=True, 
                                       download=True,
                                       transform=self.get_transform())
        elif self.dataset_name == "GTSRB":
            try:
                dataset = datasets.GTSRB(root=os.path.join(self.data_dir, "gtsrb"), 
                                    split="train", 
                                    download=True,
                                    transform=self.get_transform())
            except Exception as e:
                # If loading fails, try to clean up and redownload
                shutil.rmtree(os.path.join(self.data_dir, "gtsrb"), ignore_errors=True)
                dataset = datasets.GTSRB(root=os.path.join(self.data_dir, "gtsrb"), 
                                    split="train", 
                                    download=True,
                                    transform=self.get_transform())
        else:  # ImageNette
            try:
                dataset = datasets.ImageFolder(root=os.path.join(self.data_dir, "imagenette/train"),
                                         transform=self.get_transform())
            except Exception as e:
                # If loading fails, try to clean up and redownload
                shutil.rmtree(os.path.join(self.data_dir, "imagenette"), ignore_errors=True)
                dataset = datasets.ImageFolder(root=os.path.join(self.data_dir, "imagenette/train"),
                                         transform=self.get_transform())
                                     
        # Apply sample size if specified
        if hasattr(self.config, 'sample_size') and self.config.sample_size is not None:
            total_samples = len(dataset)
            sample_size = min(self.config.sample_size, total_samples)
            indices = torch.randperm(total_samples)[:sample_size]
            dataset = Subset(dataset, indices)
            self.logger.info(f"Using {sample_size} samples from training dataset")
            
        return dataset

    def get_val_dataset(self) -> Dataset:
        """Get validation dataset."""
        self.is_train = False
        dataset = None
        
        if self.dataset_name == "CIFAR100":
            try:
                dataset = datasets.CIFAR100(root=os.path.join(self.data_dir, "cifar100"), 
                                       train=False, 
                                       download=True,
                                       transform=self.get_transform())
            except Exception as e:
                # If loading fails, try to clean up and redownload
                shutil.rmtree(os.path.join(self.data_dir, "cifar100"), ignore_errors=True)
                dataset = datasets.CIFAR100(root=os.path.join(self.data_dir, "cifar100"), 
                                       train=False, 
                                       download=True,
                                       transform=self.get_transform())
        elif self.dataset_name == "GTSRB":
            try:
                dataset = datasets.GTSRB(root=os.path.join(self.data_dir, "gtsrb"), 
                                    split="test", 
                                    download=True,
                                    transform=self.get_transform())
            except Exception as e:
                # If loading fails, try to clean up and redownload
                shutil.rmtree(os.path.join(self.data_dir, "gtsrb"), ignore_errors=True)
                dataset = datasets.GTSRB(root=os.path.join(self.data_dir, "gtsrb"), 
                                    split="test", 
                                    download=True,
                                    transform=self.get_transform())
        else:  # ImageNette
            try:
                dataset = datasets.ImageFolder(root=os.path.join(self.data_dir, "imagenette/val"),
                                         transform=self.get_transform())
            except Exception as e:
                # If loading fails, try to clean up and redownload
                shutil.rmtree(os.path.join(self.data_dir, "imagenette"), ignore_errors=True)
                dataset = datasets.ImageFolder(root=os.path.join(self.data_dir, "imagenette/val"),
                                         transform=self.get_transform())
                                     
        # Apply sample size if specified
        if hasattr(self.config, 'sample_size') and self.config.sample_size is not None:
            total_samples = len(dataset)
            sample_size = min(self.config.sample_size, total_samples)
            indices = torch.randperm(total_samples)[:sample_size]
            dataset = Subset(dataset, indices)
            self.logger.info(f"Using {sample_size} samples from validation dataset")
            
        return dataset

    def extract_features(self, dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from dataset using feature extractor.
        
        Args:
            dataset: Dataset to extract features from
            
        Returns:
            Tuple of (features, labels)
        """
        # Generate cache key based on dataset and config
        cache_key = f"{self.dataset_name}_{len(dataset)}"
        
        # Check if features are already cached
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
            
        self.logger.info(f"Extracting features from {len(dataset)} samples...")
        
        # Configure DataLoader for optimal GPU usage
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Disable multiprocessing for now
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=None
        )
        
        features_list = []
        labels_list = []
        
        # Process in batches
        with torch.cuda.amp.autocast():  # Use mixed precision
            for batch_data, batch_labels in loader:
                # Move batch to GPU
                if isinstance(batch_data, torch.Tensor):
                    batch_data = batch_data.to(device)
                
                # Extract features using ResNet
                with torch.no_grad():  # Add no_grad to reduce memory usage
                    batch_features = self.feature_extractor(batch_data)
                
                # Handle memory differently based on dataset
                if self.dataset_name == "GTSRB":
                    # Move to CPU immediately to free GPU memory for GTSRB
                    features_list.append(batch_features.cpu())
                    labels_list.append(batch_labels)
                    torch.cuda.empty_cache()
                else:
                    # Keep on GPU for other datasets
                    features_list.append(batch_features)
                    labels_list.append(batch_labels.to(device))
        
        # Combine batches
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        logging.info(f"Feature extraction completed. Final feature shape: {features.shape}")
        
        # Convert to numpy
        features_np = features.cpu().numpy() if self.dataset_name != "GTSRB" else features.numpy()
        labels_np = labels.cpu().numpy() if self.dataset_name != "GTSRB" else labels.numpy()
        
        # Cache the results
        self._feature_cache[cache_key] = (features_np, labels_np)
        return features_np, labels_np

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
        
        # Create label mapping to ensure consecutive integers from 0
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        inverse_label_map = {idx: label for label, idx in label_map.items()}
        
        # Map labels to consecutive integers
        mapped_labels = np.array([label_map[label] for label in labels])
        n_classes = len(unique_labels)
        
        original_labels = mapped_labels.copy()
        num_to_poison = int(len(labels) * poison_rate)
        new_labels = mapped_labels.copy()
        poisoned_classes = set()

        if mode == 'random_to_random':
            poison_indices = np.random.choice(len(labels), num_to_poison, replace=False)
            for idx in poison_indices:
                new_label = np.random.choice([l for l in range(n_classes) if l != mapped_labels[idx]])
                new_labels[idx] = new_label
                poisoned_classes.add(int(mapped_labels[idx]))
                poisoned_classes.add(int(new_label))
        elif mode == 'random_to_target':
            if target_class is None:
                raise ValueError("target_class must be specified for 'random_to_target' mode.")
            mapped_target = label_map.get(target_class)
            if mapped_target is None:
                raise ValueError(f"target_class {target_class} not found in labels")
            poison_indices = np.random.choice(len(labels), num_to_poison, replace=False)
            for idx in poison_indices:
                if mapped_labels[idx] != mapped_target:
                    new_labels[idx] = mapped_target
                    poisoned_classes.add(int(mapped_labels[idx]))
                    poisoned_classes.add(mapped_target)
        elif mode == 'source_to_target':
            if source_class is None or target_class is None:
                raise ValueError("Both source_class and target_class must be specified for 'source_to_target' mode.")
            if source_class == target_class:
                raise ValueError("source_class and target_class must be different.")
                
            mapped_source = label_map.get(source_class)
            mapped_target = label_map.get(target_class)
            if mapped_source is None:
                raise ValueError(f"source_class {source_class} not found in labels")
            if mapped_target is None:
                raise ValueError(f"target_class {target_class} not found in labels")
                
            # Get indices of samples in source class
            source_indices = np.where(mapped_labels == mapped_source)[0]
            num_source_samples = len(source_indices)
            
            # Calculate the number of samples to flip, limited by available samples
            # Always keep at least one sample in the source class to prevent class elimination
            num_to_flip = min(num_source_samples - 1, num_to_poison) if num_source_samples > 0 else 0
            
            # Log detailed information about the flipping operation
            logging.info(f"Label flipping details:")
            logging.info(f"- Source class: {source_class} (mapped to {mapped_source})")
            logging.info(f"- Target class: {target_class} (mapped to {mapped_target})")
            logging.info(f"- Available samples in source class: {num_source_samples}")
            logging.info(f"- Requested samples to poison: {num_to_poison}")
            logging.info(f"- Actual samples to flip: {num_to_flip}")
            logging.info(f"- Samples remaining in source class: {num_source_samples - num_to_flip}")
            
            # Only proceed if we have samples to flip
            if num_to_flip > 0:
                # Randomly select indices to flip
                flip_indices = np.random.choice(source_indices, num_to_flip, replace=False)
                new_labels[flip_indices] = mapped_target
                poisoned_classes.add(mapped_source)
                poisoned_classes.add(mapped_target)
                logging.info(f"Successfully flipped {num_to_flip} labels from class {source_class} to {target_class}")
            else:
                logging.warning(f"No labels were flipped: insufficient samples in source class {source_class}")
        else:
            raise ValueError("Invalid mode specified for label flipping.")

        # Map labels back to original space
        final_labels = np.array([inverse_label_map[label] for label in new_labels])
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
        self._label_flip_cache[cache_key] = (final_labels.astype(np.int64), attack_params)
        return final_labels.astype(np.int64), attack_params

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
