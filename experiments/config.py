"""
Experiment configuration for DynaDetect v2 evaluation
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Constants
POISON_RATES = [0.01, 0.03, 0.05, 0.07, 0.10, 0.20]  # Full range of poison rates
TEST_POISON_RATES = POISON_RATES  # Use all poison rates for testing

# Test sample sizes for quick evaluation
TEST_SAMPLE_SIZES = {
    'CIFAR100': 1000,  # Small sample for quick testing
    'GTSRB': 1000,     # Small sample for quick testing
    'ImageNette': 1000  # Small sample for quick testing
}

# Experiment modes
EXPERIMENT_MODES = ['standard', 'dynadetect']

# Attack methods
ATTACK_METHODS = [
    'random_to_random',
    'random_to_target',
    'source_to_target',
    'pgd'
]

# Label flipping types
LABEL_FLIP_TYPES = [
    'random_to_random',
    'random_to_target',
    'source_to_target'
]

# Attack methods for testing - use all methods
TEST_ATTACK_METHODS = ATTACK_METHODS
TEST_LABEL_FLIP_TYPES = LABEL_FLIP_TYPES

# Dataset configurations
@dataclass
class DatasetConfig:
    name: str
    dataset_type: str
    sample_size: int
    metric: str
    modification_method: str
    poison_rate: float
    attack_params: Dict[str, Any]
    mode: str = 'standard'
    seed: int = 42

@dataclass
class DatasetSpecs:
    name: str
    dataset_type: str
    default_sample_size: int
    min_sample_size: int
    max_sample_size: int
    min_samples_per_class: int
    num_classes: int

DATASET_CONFIGS = {
    'CIFAR100': DatasetSpecs(
        name='CIFAR100',
        dataset_type='image',
        default_sample_size=50000,
        min_sample_size=1000,  # Ensure at least 10 samples per class
        max_sample_size=50000,
        min_samples_per_class=10,
        num_classes=100
    ),
    'GTSRB': DatasetSpecs(
        name='GTSRB',
        dataset_type='image',
        default_sample_size=39209,  # Full training set
        min_sample_size=1000,
        max_sample_size=39209,
        min_samples_per_class=30,
        num_classes=43
    ),
    'ImageNette': DatasetSpecs(
        name='ImageNette',
        dataset_type='image',
        default_sample_size=9469,  # Full training set
        min_sample_size=1000,
        max_sample_size=9469,
        min_samples_per_class=100,
        num_classes=10
    )
}

DATASETS = {
    'image': ['CIFAR100', 'GTSRB', 'ImageNette']
}

# Available classifiers
CLASSIFIERS = [
    'LogisticRegression',
    'KNeighbors',
    'SVM',
    'RandomForest'
]

# Test subset configurations
TEST_DATASETS = {
    'image': ['CIFAR100', 'GTSRB', 'ImageNette']
}

# Test classifiers - use all classifiers
TEST_CLASSIFIERS = CLASSIFIERS

# Label Flipping Configurations
LABEL_FLIP_MODES = ['random_to_random', 'random_to_target', 'source_to_target']
LABEL_FLIP_CONFIGS = [
    {
        'mode': mode,
        'target_class': 1 if 'target' in mode else None,
        'source_class': 0 if mode == 'source_to_target' else None
    }
    for mode in LABEL_FLIP_MODES
]

# Test subset of label flipping modes
TEST_LABEL_FLIP_CONFIGS = [
    {'mode': 'random_to_random'}
]

# Gradient Attack Configurations
GRADIENT_CONFIGS = {
    'pgd': {
        'eps': 0.3,
        'alpha': 2/255,
        'iters': 40
    },
    'gradient_ascent': {
        'lr': 0.1,
        'iters': 100
    }
}

# Test subset of gradient configs
TEST_GRADIENT_CONFIGS = {
    'pgd': {
        'eps': 0.3,
        'alpha': 2/255,
        'iters': 40
    }
}

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    datasets: List[DatasetConfig]
    classifiers: List[str]
    modes: List[str]
    iterations: int = 1
    results_file: str = 'results/experiment_results_{timestamp}.json'  # timestamp placeholder
    seed: int = 42

    def __len__(self):
        return len(self.datasets)

def validate_sample_size(dataset: str, sample_size: Optional[int]) -> Optional[int]:
    """Validate and adjust sample size for dataset."""
    config = DATASET_CONFIGS[dataset]
    
    # If sample_size is None, use default
    if sample_size is None:
        return config.default_sample_size
    
    # Enforce min/max bounds
    sample_size = max(sample_size, config.min_sample_size)
    sample_size = min(sample_size, config.max_sample_size)
    
    return sample_size

def generate_experiment_configs(
    test_mode: bool = False,
    sample_size_override: Optional[int] = None,
    seed: int = 42
) -> List[ExperimentConfig]:
    """Generate experiment configurations.
    
    Args:
        test_mode: If True, use reduced test configurations
        sample_size_override: If provided, override default sample sizes
        seed: Random seed for reproducibility
    """
    configs = []
    
    # Use reduced settings for testing
    if test_mode:
        datasets = ['CIFAR100']  # Only one dataset
        poison_rates = TEST_POISON_RATES
    else:
        datasets = ['CIFAR100', 'GTSRB', 'ImageNette']
        poison_rates = POISON_RATES
    
    for dataset in datasets:
        specs = DATASET_CONFIGS[dataset]
        sample_size = sample_size_override or specs.default_sample_size
        sample_size = validate_sample_size(dataset, sample_size)
        
        # Standard mode configurations
        for poison_rate in poison_rates:
            # Label flipping attacks
            for flip_type in ['random_to_random', 'random_to_target', 'source_to_target']:
                configs.append(ExperimentConfig(
                    datasets=[DatasetConfig(
                        name=dataset,
                        dataset_type=specs.dataset_type,
                        sample_size=sample_size,
                        metric='accuracy',
                        modification_method='label_flipping',
                        attack_params={'flip_type': flip_type},
                        poison_rate=poison_rate,
                        mode='standard',
                        seed=seed
                    )],
                    classifiers=CLASSIFIERS,
                    modes=['standard'],
                    iterations=1,
                    results_file='results/experiment_results_{timestamp}.json',  # timestamp placeholder
                    seed=seed
                ))
            
            # PGD attack
            configs.append(ExperimentConfig(
                datasets=[DatasetConfig(
                    name=dataset,
                    dataset_type=specs.dataset_type,
                    sample_size=sample_size,
                    metric='accuracy',
                    modification_method='pgd',
                    attack_params={
                        'eps': 0.1,
                        'alpha': 0.01,
                        'iters': 40
                    },
                    poison_rate=poison_rate,
                    mode='standard',
                    seed=seed
                )],
                classifiers=CLASSIFIERS,
                modes=['standard'],
                iterations=1,
                results_file='results/experiment_results_{timestamp}.json',  # timestamp placeholder
                seed=seed
            ))
        
        # DynaDetect mode configurations
        for poison_rate in poison_rates:
            for attack_type in ['label_flipping', 'pgd']:
                configs.append(ExperimentConfig(
                    datasets=[DatasetConfig(
                        name=dataset,
                        dataset_type=specs.dataset_type,
                        sample_size=sample_size,
                        metric='accuracy',
                        modification_method=attack_type,
                        attack_params={
                            'eps': 0.1,
                            'alpha': 0.01,
                            'iters': 40
                        } if attack_type == 'pgd' else {'flip_type': 'random_to_random'},
                        poison_rate=poison_rate,
                        mode='dynadetect',
                        seed=seed
                    )],
                    classifiers=CLASSIFIERS,
                    modes=['dynadetect'],
                    iterations=1,
                    results_file='results/experiment_results_{timestamp}.json',  # timestamp placeholder
                    seed=seed
                ))
    
    return configs

def get_baseline_configs(
    test_mode: bool = False,
    sample_size_override: Optional[int] = None
) -> List[ExperimentConfig]:
    """Generate baseline configurations without poisoning.
    
    Args:
        test_mode: If True, use reduced test configurations
        sample_size_override: If provided, override default sample sizes
    """
    configs = []
    datasets = TEST_DATASETS if test_mode else DATASETS

    for dataset_type, dataset_list in datasets.items():
        for dataset_name in dataset_list:
            # Get validated sample size
            sample_size = validate_sample_size(dataset_name, sample_size_override)
            
            # Generate configs for both standard and dynadetect modes
            for mode in EXPERIMENT_MODES:
                configs.append(ExperimentConfig(
                    datasets=[DatasetConfig(
                        name=dataset_name,
                        dataset_type=dataset_type,
                        sample_size=sample_size,
                        metric='accuracy',
                        modification_method='none',
                        attack_params={},
                        poison_rate=0.0,
                        mode=mode,
                        seed=42
                    )],
                    classifiers=CLASSIFIERS,
                    modes=[mode],
                    iterations=1,
                    results_file='results/experiment_results_{timestamp}.json',  # timestamp placeholder
                    seed=42
                ))

    return configs

def get_classifiers(test_mode: bool = False) -> List[str]:
    """Get list of classifiers to use.
    
    Args:
        test_mode: If True, return reduced set of classifiers
    """
    return TEST_CLASSIFIERS if test_mode else CLASSIFIERS

def get_test_configs() -> List[ExperimentConfig]:
    """Generate test configurations with small sample sizes."""
    configs = []
    
    # Use all datasets
    for dataset_type, dataset_list in TEST_DATASETS.items():
        for dataset_name in dataset_list:
            # Get dataset specs
            dataset_specs = DATASET_CONFIGS[dataset_name]
            
            # Use test sample size
            sample_size = TEST_SAMPLE_SIZES[dataset_name]
            
            # For each classifier
            for classifier in CLASSIFIERS:
                # For each attack method
                for attack_method in TEST_ATTACK_METHODS:
                    # For each poison rate
                    for poison_rate in TEST_POISON_RATES:
                        # For each mode (regular and dynadetect)
                        for mode in EXPERIMENT_MODES:
                            if attack_method == 'label_flipping':
                                # For each label flipping type
                                for flip_type in TEST_LABEL_FLIP_TYPES:
                                    configs.append(
                                        DatasetConfig(
                                            name=dataset_name,
                                            dataset_type=dataset_type,
                                            sample_size=sample_size,
                                            metric='accuracy',
                                            modification_method=attack_method,
                                            poison_rate=poison_rate,
                                            attack_params={'flip_type': flip_type},
                                            mode=mode
                                        )
                                    )
                            else:  # PGD attack
                                configs.append(
                                    DatasetConfig(
                                        name=dataset_name,
                                        dataset_type=dataset_type,
                                        sample_size=sample_size,
                                        metric='accuracy',
                                        modification_method=attack_method,
                                        poison_rate=poison_rate,
                                        attack_params={},
                                        mode=mode
                                    )
                                )
    
    return ExperimentConfig(
        datasets=configs,
        classifiers=CLASSIFIERS,
        modes=EXPERIMENT_MODES,
        iterations=1
    )
