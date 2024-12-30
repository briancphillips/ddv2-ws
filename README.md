# DynaDetect v2

A high-performance machine learning framework focused on GPU-accelerated model implementations and robust security assessment, featuring advanced adversarial attack detection and mitigation strategies.

## Core Features

- **GPU-Accelerated Models**:

  - DDKNN (Dynamic Distance-based k-Nearest Neighbors)
    - Gradient-based attack support
    - Soft voting with temperature parameter
    - Distance-weighted predictions
  - GPU-optimized implementations of:
    - SVM with early stopping
    - Decision Trees with gradient support
    - Random Forest with batch processing
    - KNN with efficient distance computation

- **Advanced Training Framework**:

  - Robust feature selection and anomaly detection
  - Automatic mixed precision training
  - Sample weight computation with caching
  - GPU-accelerated Local Outlier Factor
  - Efficient batch processing

- **Performance Optimization**:
  - CUDA acceleration for traditionally CPU-bound algorithms
  - Memory-efficient batch processing
  - Feature caching system
  - Automatic mixed precision support
  - Resource monitoring and optimization

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
# Full evaluation with GPU acceleration
python experiments/run_full_evaluation.py

# Test mode (reduced configurations)
python experiments/run_full_evaluation.py --test
```

## Project Structure

```
dynadetectv2/
├── core/                     # Core ML implementations
│   ├── models.py            # GPU-accelerated model implementations
│   ├── dataset.py           # Dataset handling and feature extraction
│   └── trainer.py           # Training orchestration with GPU support
├── evaluation/              # Evaluation framework
│   ├── evaluator.py        # Performance and security evaluation
│   └── runner.py           # Experiment execution
├── config/                 # Configuration management
└── attacks/                # Attack implementations

experiments/
├── run_full_evaluation.py  # Main evaluation script
├── visualization/          # Result visualization
└── results_management/     # Results handling
```

## Results and Logging

- Comprehensive performance metrics stored in CSV format
- Automatic resource usage monitoring (CPU, GPU, memory)
- Detailed logging with timestamp-based organization
- Automatic log archiving and management

## Configuration

The system uses a configuration-driven approach:

- Model-specific parameters (temperature, batch size, etc.)
- GPU optimization settings
- Dataset configurations
- Training parameters
- Attack configurations

## Development Status

This is an active research project focusing on:

- GPU acceleration of traditional ML algorithms
- Advanced adversarial attack detection
- Performance optimization for large-scale evaluations
- Resource-efficient model implementations

## Hardware Requirements

- CUDA-capable GPU recommended
- Sufficient GPU memory for batch processing
- Fast storage for feature caching
