"""Configuration module for DynaDetect."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import random

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    dataset_type: str = 'image'  # 'image' or 'numerical'
    sample_size: Optional[int] = None
    metric: str = 'accuracy'
    n_neighbors: int = 5
    pca_components: float = 0.95
    batch_size: int = 32
    attack_params: Optional[Dict] = None

@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    datasets: List[DatasetConfig]
    classifiers: List[str]
    modes: List[str]
    iterations: int
    seed: int
    results_file: str

# Constants
DATASET_CONFIGS = {
    'GTSRB': {'type': 'image', 'num_classes': 43},
    'ImageNette': {'type': 'image', 'num_classes': 10},
    'CIFAR100': {'type': 'image', 'num_classes': 100},
}

POISON_RATES = [0.0, 0.01, 0.03, 0.05, 0.07, 0.10, 0.20]  # Clean baseline, 1%, 3%, 5%, 7%, 10%, 20%

ATTACK_METHODS = {
    'label_flipping': {
        'modes': ['random_to_random', 'random_to_target', 'source_to_target']
    },
    'pgd': {
        'eps': 0.3,
        'alpha': 2/255,
        'iters': 40
    },
    'gradient_ascent': {
        'eps': 0.3,
        'alpha': 2/255,
        'iters': 40
    }
}

# Full dataset sizes
DATASET_SIZES = {
    'CIFAR100': 50000,  # Full training set size
    'GTSRB': 39209,     # Full training set size
    'ImageNette': 9469  # Full training set size
}

# Test sample sizes (reduced for faster testing)
TEST_SAMPLE_SIZES = {
    'CIFAR100': 5000,
    'GTSRB': 5000,
    'ImageNette': 5000
}

CLASSIFIERS = [
    'SVM',
    'LogisticRegression',
    'RandomForest',
    'KNeighbors'
]

MODES = ['standard', 'dynadetect']

class ConfigurationManager:
    """Manages experiment configurations."""
    
    def __init__(self):
        """Initialize ConfigurationManager."""
        self.seed = random.randint(1, 10000)
        
    def get_test_configs(self) -> ExperimentConfig:
        """Get test configurations with reduced dataset sizes and all attack types."""
        # Create a list to store all dataset configs
        all_datasets = []
        
        # For each dataset, create configurations for each attack type
        for name, config in DATASET_CONFIGS.items():
            # Add label flipping configurations
            for flip_mode in ATTACK_METHODS['label_flipping']['modes']:
                # Set target and source classes based on flip mode
                target_class = 0 if flip_mode in ['random_to_target', 'source_to_target'] else None
                source_class = 1 if flip_mode == 'source_to_target' else None
                
                dataset_config = DatasetConfig(
                    name=name,
                    dataset_type=config['type'],
                    sample_size=TEST_SAMPLE_SIZES[name],
                    attack_params={
                        'poison_rates': POISON_RATES,
                        'type': 'label_flipping',
                        'mode': flip_mode,
                        'target_class': target_class,
                        'source_class': source_class
                    }
                )
                all_datasets.append(dataset_config)
            
            # Add PGD configuration
            dataset_config = DatasetConfig(
                name=name,
                dataset_type=config['type'],
                sample_size=TEST_SAMPLE_SIZES[name],
                attack_params={
                    'poison_rates': POISON_RATES,
                    'type': 'pgd',
                    'eps': ATTACK_METHODS['pgd']['eps'],
                    'alpha': ATTACK_METHODS['pgd']['alpha'],
                    'iters': ATTACK_METHODS['pgd']['iters']
                }
            )
            all_datasets.append(dataset_config)

            # Add Gradient Ascent configuration
            dataset_config = DatasetConfig(
                name=name,
                dataset_type=config['type'],
                sample_size=TEST_SAMPLE_SIZES[name],
                attack_params={
                    'poison_rates': POISON_RATES,
                    'type': 'gradient_ascent',
                    'eps': ATTACK_METHODS['gradient_ascent']['eps'],
                    'alpha': ATTACK_METHODS['gradient_ascent']['alpha'],
                    'iters': ATTACK_METHODS['gradient_ascent']['iters']
                }
            )
            all_datasets.append(dataset_config)
        
        return ExperimentConfig(
            datasets=all_datasets,
            classifiers=CLASSIFIERS,
            modes=MODES,
            iterations=1,
            seed=self.seed,
            results_file='test_results.csv'
        )
        
    def get_full_configs(self) -> ExperimentConfig:
        """Get full experiment configurations."""
        # Create a list to store all dataset configs
        all_datasets = []
        
        # For each dataset, create configurations for each attack type
        for name, config in DATASET_CONFIGS.items():
            # Add label flipping configurations
            for flip_mode in ATTACK_METHODS['label_flipping']['modes']:
                # Set target and source classes based on flip mode
                target_class = 0 if flip_mode in ['random_to_target', 'source_to_target'] else None
                source_class = 1 if flip_mode == 'source_to_target' else None
                
                dataset_config = DatasetConfig(
                    name=name,
                    dataset_type=config['type'],
                    sample_size=DATASET_SIZES[name],  # Use full dataset sizes
                    attack_params={
                        'poison_rates': POISON_RATES,
                        'type': 'label_flipping',
                        'mode': flip_mode,
                        'target_class': target_class,
                        'source_class': source_class
                    }
                )
                all_datasets.append(dataset_config)
            
            # Add PGD configuration
            dataset_config = DatasetConfig(
                name=name,
                dataset_type=config['type'],
                sample_size=DATASET_SIZES[name],  # Use full dataset sizes
                attack_params={
                    'poison_rates': POISON_RATES,
                    'type': 'pgd',
                    'eps': ATTACK_METHODS['pgd']['eps'],
                    'alpha': ATTACK_METHODS['pgd']['alpha'],
                    'iters': ATTACK_METHODS['pgd']['iters']
                }
            )
            all_datasets.append(dataset_config)

            # Add Gradient Ascent configuration
            dataset_config = DatasetConfig(
                name=name,
                dataset_type=config['type'],
                sample_size=DATASET_SIZES[name],  # Use full dataset sizes
                attack_params={
                    'poison_rates': POISON_RATES,
                    'type': 'gradient_ascent',
                    'eps': ATTACK_METHODS['gradient_ascent']['eps'],
                    'alpha': ATTACK_METHODS['gradient_ascent']['alpha'],
                    'iters': ATTACK_METHODS['gradient_ascent']['iters']
                }
            )
            all_datasets.append(dataset_config)
        
        return ExperimentConfig(
            datasets=all_datasets,
            classifiers=CLASSIFIERS,
            modes=MODES,
            iterations=1,
            seed=self.seed,
            results_file='experiment_results.csv'
        )
