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
    'CIFAR100': {'type': 'image', 'num_classes': 100},
    'GTSRB': {'type': 'image', 'num_classes': 43},
    'ImageNette': {'type': 'image', 'num_classes': 10},
}

POISON_RATES = [0.01, 0.03, 0.05, 0.07, 0.10, 0.20]  # 1%, 3%, 5%, 7%, 10%, 20%

ATTACK_METHODS = {
    'label_flipping': {
        'modes': ['random_to_random', 'random_to_target', 'source_to_target']
    },
    'pgd': {
        'eps': 0.3,
        'alpha': 2/255,
        'iters': 40
    }
}

TEST_SAMPLE_SIZES = {
    'CIFAR100': 1000,
    'GTSRB': 1000,
    'ImageNette': 1000
}

CLASSIFIERS = [
    'LogisticRegression',
    'RandomForest',
    'SVM',
    'KNeighbors'
]

MODES = ['standard', 'dynadetect']

class ConfigurationManager:
    """Manages experiment configurations."""
    
    def __init__(self):
        """Initialize ConfigurationManager."""
        self.seed = random.randint(1, 10000)
        
    def get_test_configs(self) -> ExperimentConfig:
        """Get test configurations with reduced dataset sizes but full configuration options."""
        datasets = [
            DatasetConfig(
                name=name,
                dataset_type=config['type'],
                sample_size=TEST_SAMPLE_SIZES[name],
                attack_params={
                    'poison_rates': POISON_RATES,
                    'type': 'label_flipping',
                    'modes': ['random_to_random']
                }
            )
            for name, config in DATASET_CONFIGS.items()
        ]
        
        return ExperimentConfig(
            datasets=datasets,
            classifiers=CLASSIFIERS,  # Use all classifiers
            modes=MODES,  # Use all modes
            iterations=1,  # Use 1 iteration for test mode
            seed=self.seed,
            results_file='test_results.csv'
        )
        
    def get_full_configs(self) -> ExperimentConfig:
        """Get full experiment configurations."""
        datasets = [
            DatasetConfig(
                name=name,
                dataset_type=config['type'],
                attack_params={'poison_rates': POISON_RATES}
            )
            for name, config in DATASET_CONFIGS.items()
        ]
        
        return ExperimentConfig(
            datasets=datasets,
            classifiers=CLASSIFIERS,
            modes=MODES,
            iterations=5,
            seed=self.seed,
            results_file='experiment_results.csv'
        )
