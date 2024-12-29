"""Dataset handling for DynaDetect v2."""

from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms, models
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
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicBlock(nn.Module):
    """Basic block for WideResNet."""
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    """Layer container for WideResNet."""
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    """WideResNet implementation."""
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

class FeatureExtractor(nn.Module):
    """Feature extractor using pretrained models."""
    def __init__(self, dataset_name: str):
        super(FeatureExtractor, self).__init__()
        self.dataset_name = dataset_name
        
        if dataset_name == "CIFAR100":
            # Use WRN-40-4 for CIFAR100
            self.model = WideResNet(depth=40, num_classes=100, widen_factor=4)
            self.load_pretrained_wrn()
            # Remove the final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        elif dataset_name == "GTSRB":
            # Use WRN-16-8 for GTSRB
            self.model = WideResNet(depth=16, num_classes=43, widen_factor=8)
            # Remove the final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        else:
            # Use ResNet18 for other datasets
            self.model = models.resnet18(pretrained=True)
            # Remove the final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        self.model = self.model.to(device)
        self.model.eval()

    def forward(self, x):
        with torch.cuda.amp.autocast():
            features = self.model(x)
            if isinstance(features, tuple):
                features = features[0]
            features = features.reshape(features.size(0), -1)
            return features

    def load_pretrained_wrn(self):
        """Load pretrained weights for WideResNet."""
        try:
            # Create directory for pretrained models if it doesn't exist
            os.makedirs("pretrained_models", exist_ok=True)
            weights_path = os.path.join("pretrained_models", "wrn40_4_cifar100.pth")
            
            # Download weights if they don't exist
            if not os.path.exists(weights_path):
                url = "https://huggingface.co/datasets/brianhie/wrn-weights/resolve/main/wrn40_4_cifar100.pth"
                response = requests.get(url)
                response.raise_for_status()
                with open(weights_path, "wb") as f:
                    f.write(response.content)
                logging.info(f"Downloaded pretrained weights to {weights_path}")
            
            # Load the weights
            state_dict = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state_dict)
            logging.info("Successfully loaded pretrained weights")
        except Exception as e:
            logging.warning(f"Could not load pretrained weights: {str(e)}")

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
    """Handler for dataset loading and preprocessing."""
    def __init__(self, dataset_name: str, data_dir: str = ".datasets"):
        """Initialize dataset handler."""
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.is_train = True  # Default to True for training transforms
        self.feature_extractor = FeatureExtractor(dataset_name).to(device)
        self.feature_extractor.eval()
        self._feature_cache = {}  # Cache for extracted features
        self._label_flip_cache = {}  # Cache for flipped labels

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
        if self.dataset_name == "CIFAR100":
            return datasets.CIFAR100(root=self.data_dir, train=True, download=True,
                                   transform=self.get_transform())
        elif self.dataset_name == "GTSRB":
            return datasets.GTSRB(root=self.data_dir, split="train", download=True,
                                transform=self.get_transform())
        else:  # ImageNette
            return datasets.ImageFolder(root=os.path.join(self.data_dir, "imagenette/train"),
                                     transform=self.get_transform())

    def get_val_dataset(self) -> Dataset:
        """Get validation dataset."""
        self.is_train = False
        if self.dataset_name == "CIFAR100":
            return datasets.CIFAR100(root=self.data_dir, train=False, download=True,
                                   transform=self.get_transform())
        elif self.dataset_name == "GTSRB":
            return datasets.GTSRB(root=self.data_dir, split="test", download=True,
                                transform=self.get_transform())
        else:  # ImageNette
            return datasets.ImageFolder(root=os.path.join(self.data_dir, "imagenette/val"),
                                     transform=self.get_transform())

    def extract_features(self, dataset: torch.utils.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from dataset using pretrained ResNet."""
        # Generate cache key based on dataset object id and length
        cache_key = (id(dataset), len(dataset))
        if cache_key in self._feature_cache:
            logging.info("Using cached features")
            return self._feature_cache[cache_key]
        
        total_samples = len(dataset)
        # Set batch size based on dataset
        batch_size = 32 if self.dataset_name == "GTSRB" else 128
        num_workers = 2 if self.dataset_name == "GTSRB" else 4
        
        logging.info(f"Extracting features from {total_samples} samples...")
        
        # Create DataLoader for batch processing
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
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
