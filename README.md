# DynaDetect v2

A robust framework for detecting and mitigating data poisoning attacks in machine learning models.

## Project Structure

```
dynadetectv2/
├── attacks/            # Attack implementations (label flipping, etc.)
├── config/            # Configuration management
├── core/             # Core functionality
│   ├── dataset.py    # Dataset handling and transformations
│   ├── models.py     # Model implementations
│   └── trainer.py    # Training implementations
├── evaluation/       # Evaluation metrics and tools
└── main.py          # Main entry point

experiments/
├── results_management/  # Results processing and analysis
├── utils/              # Utility functions
└── run_full_evaluation.py  # Full evaluation script

tests/                  # Test suite
.datasets/              # Dataset storage
logs/                   # Experiment logs
results/                # Experiment results
```

## Features

- Multiple attack types support:

  - Label flipping (random-to-random, random-to-target, source-to-target)
  - Future: PGD and other attack implementations

- Multiple dataset support:

  - GTSRB (German Traffic Sign Recognition Benchmark)
  - CIFAR-100
  - ImageNette
  - Support for numerical datasets

- Multiple classifier support:

  - SVM
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors

- Evaluation modes:
  - Standard (baseline)
  - DynaDetect (with dynamic detection)

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Evaluations

For test mode (reduced dataset size):

```bash
python experiments/run_full_evaluation.py --test
```

For full evaluation:

```bash
python experiments/run_full_evaluation.py
```

### Configuration

- Dataset configurations in `dynadetectv2/config/__init__.py`
- Attack parameters and poison rates can be modified in the config
- Experiment parameters (iterations, modes, etc.) are configurable

## Results

Results are stored in:

- CSV format in `results/` directory
- Logs in `logs/` directory

## Development Status

See `TODO.md` for current development status and planned features.
See `REFACTORING.md` for ongoing refactoring tasks.
