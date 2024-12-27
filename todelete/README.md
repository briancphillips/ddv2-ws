# DynaDetect v2

A dynamic poisoning detection framework for machine learning models, focusing on robust detection of poisoning attacks in image classification tasks.

## Features

### Attack Methods
- Label Flipping Attacks:
  - Random-to-Random
  - Random-to-Target
  - Source-to-Target
- Gradient-based Attacks:
  - PGD (Projected Gradient Descent)
  - Custom attack strength and iterations

### Supported Datasets
- CIFAR-100 (primary dataset)
- GTSRB (German Traffic Sign Recognition Benchmark)
- ImageNette

### Classifiers
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- All classifiers support both standard and DynaDetect modes

### Framework Features
- Comprehensive experiment configuration system
- Dynamic feature extraction with ResNet50
- Automatic dimensionality handling (PCA/padding)
- Robust training procedures with sample weights
- Extensive metrics collection and analysis
- GPU-accelerated computations where applicable

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ddv2-ws.git
cd ddv2-ws
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Environment setup:
- Requires Python environment with GPU support
- CUDA toolkit recommended for GPU acceleration

## Usage

### Running Experiments

Full evaluation:
```bash
python experiments/run_full_evaluation.py
```

Test mode with reduced dataset:
```bash
python experiments/run_full_evaluation.py --test
```

### Configuration

Experiment configurations in `experiments/config.py`:
- Dataset configurations (sample sizes, preprocessing)
- Attack parameters (poison rates, attack methods)
- Model configurations (classifiers, modes)
- Training parameters (batch size, learning rate)

## Project Structure

```
ddv2-ws/
├── dynadetectv2/                # Core package
│   ├── attacks/                 # Attack implementations
│   ├── config/                  # Configuration management
│   ├── core/                    # Core implementations
│   │   ├── dataset.py          # Dataset handling
│   │   ├── models.py           # Model implementations
│   │   └── trainer.py          # Training logic
│   ├── evaluation/             # Evaluation metrics and analysis
│   └── main.py                 # Main entry point
├── experiments/                 # Experiment framework
│   ├── config.py               # Experiment configuration
│   ├── results_collector.py    # Results collection utilities
│   ├── run_experiments.py      # Basic experiment runner
│   ├── run_full_evaluation.py  # Full evaluation suite
│   └── test_dynadetect.py     # Testing utilities
├── results/                    # Experiment results
├── logs/                       # Experiment logs
└── dynadetectv2.py            # Legacy implementation (to be deprecated)
```

### Key Components

- `dynadetectv2/`: Core implementation package
  - `attacks/`: Poisoning attack implementations
  - `config/`: Configuration management system
  - `core/`: Core ML components and utilities
  - `evaluation/`: Metrics and analysis tools
  - `main.py`: Main package entry point

- `experiments/`: Experiment framework
  - `config.py`: Comprehensive configuration system
  - `results_collector.py`: Results collection and analysis
  - `run_full_evaluation.py`: Main experiment runner
  - `test_dynadetect.py`: Testing and validation

- `results/` and `logs/`: Output directories
  - Timestamped experiment results
  - Detailed logging output
  - Performance metrics and analysis

## Results

Results are stored in two formats:
- CSV files: Detailed metrics per experiment
- JSON files: Complete experiment data including configurations

Key metrics tracked:
- Model accuracy, precision, recall, F1-score
- Per-class performance metrics
- Attack effectiveness measurements
- Resource utilization statistics

## License

This project is licensed under the MIT License - see the LICENSE file for details.
