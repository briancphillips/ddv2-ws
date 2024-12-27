# DynaDetect v2

A robust framework for evaluating and improving machine learning model resilience against data poisoning attacks.

## Overview

DynaDetect v2 is a comprehensive framework designed to:

- Evaluate model performance under various poisoning scenarios
- Implement dynamic detection of poisoned data
- Support both image and numerical datasets
- Provide extensive visualization and analysis tools

## Project Structure

```
dynadetectv2/
├── core/               # Core functionality
│   ├── dataset.py     # Dataset handling and preprocessing
│   ├── models.py      # Model implementations
│   └── trainer.py     # Training utilities
├── attacks/           # Attack implementations
├── evaluation/        # Evaluation metrics and tools
├── config/           # Configuration management
└── main.py           # Main entry point

experiments/
├── results_management/  # Results handling and visualization
├── utils/              # Utility functions
└── run_full_evaluation.py  # Evaluation script
```

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run a full evaluation:

```bash
python experiments/run_full_evaluation.py
```

### Configuration

The framework supports various configuration options through `ExperimentConfig`:

- Dataset selection and parameters
- Model selection
- Attack scenarios
- Evaluation metrics

## Features

- **Dynamic Detection**: Real-time detection of poisoned data during training
- **Multiple Datasets**: Support for CIFAR100, GTSRB, ImageNette, and custom datasets
- **Visualization Tools**: Comprehensive plotting and analysis utilities
- **Memory Tracking**: Built-in memory usage monitoring
- **Extensible**: Easy to add new models, attacks, and datasets

## Development

### Testing

Run the test suite:

```bash
pytest tests/
```

### Adding New Components

- Models: Extend `BaseModel` in `core/models.py`
- Attacks: Implement in `attacks/` directory
- Datasets: Add to `core/dataset.py`

## License

[License details]

## Contributing

[Contribution guidelines]
