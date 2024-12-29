# DynaDetect v2

A machine learning evaluation framework focused on model robustness and security assessment, with particular emphasis on adversarial attack detection and mitigation.

## Core Features

- **Model Support**:

  - DDKNN (Dynamic Distance-based k-Nearest Neighbors)
  - GPU-accelerated LogisticRegression
  - GPU-accelerated SVM
  - RandomForest with early stopping
  - Decision Trees
  - K-Nearest Neighbors

- **Evaluation Capabilities**:

  - Label flipping attack assessment
  - Model performance metrics (accuracy, precision, recall, F1)
  - Per-class performance analysis
  - Latency measurements
  - Resource utilization tracking

- **Dataset Handling**:
  - Built-in GTSRB dataset support
  - Feature extraction caching
  - Data poisoning capabilities
  - Efficient batch processing

## Quick Start

1. **Environment Setup**:

```bash
# Create and activate conda environment
conda create -n jupyterlab python=3.8
conda activate jupyterlab

# Install dependencies
pip install -r requirements.txt
```

2. **Run Evaluation**:

```bash
# Full evaluation
python experiments/run_full_evaluation.py

# Test mode (reduced configurations)
python experiments/run_full_evaluation.py --test
```

## Project Structure

```
dynadetectv2/
├── core/           # Core ML implementations
│   ├── models.py   # Model implementations
│   ├── dataset.py  # Dataset handling
│   └── trainer.py  # Training orchestration
├── evaluation/     # Evaluation framework
│   ├── evaluator.py # Evaluation logic
│   └── runner.py   # Experiment execution
├── config/        # Configuration management
└── attacks/       # Attack implementations

experiments/
├── run_full_evaluation.py  # Main evaluation script
├── visualization/          # Result visualization
└── results_management/     # Results handling
```

## Results and Logging

- Results are stored in CSV format in the `results/` directory
- Logs are maintained in the `logs/` directory with automatic archiving
- Each run creates timestamped files for both results and logs

## Configuration

The system uses a configuration-driven approach through `ConfigurationManager`:

- Dataset configurations (GTSRB supported)
- Model parameters
- Evaluation modes (standard, dynadetect)
- Attack parameters (label flipping types, poison rates)

## Development Status

This is an active research project. The current implementation focuses on:

- Model robustness evaluation
- Label flipping attack assessment
- Performance optimization for large-scale evaluations
- Resource utilization monitoring
