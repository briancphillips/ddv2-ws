"""Configuration module for DynaDetect."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    type: str = 'image'  # 'image' or 'numerical'
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

POISON_RATES = [0.05, 0.10, 0.15, 0.20]

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
    'CIFAR100': 100,
    'GTSRB': 200,
    'ImageNette': 200
}
