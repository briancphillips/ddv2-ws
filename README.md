# DynaDetect v2

A dynamic detection system for analyzing and evaluating machine learning models with a focus on robustness and security evaluation.

## Overview

DynaDetect v2 is a comprehensive framework for:

- Training and evaluating machine learning models
- Analyzing model behavior under various conditions
- Performing robustness assessments
- Visualizing and interpreting results

## Key Components

1. **Core Framework** (`dynadetectv2/core/`)

   - Model implementations (Traditional ML and Neural)
   - Dataset handling and preprocessing
   - Training orchestration

2. **Evaluation Framework** (`dynadetectv2/evaluation/`)

   - Comprehensive model evaluation
   - Metrics calculation and logging
   - Performance analysis

3. **Visualization Tools** (`experiments/visualization/`)
   - Web-based result visualization
   - Performance metrics plotting
   - Interactive analysis dashboard

## Supported Models

- DDKNN (with gradient-based capabilities)
- Logistic Regression
- Support Vector Machines
- Random Forest
- Decision Trees
- K-Nearest Neighbors

## Getting Started

1. **Installation**

```bash
pip install -r requirements.txt
```

2. **Running Evaluations**

```bash
python -m dynadetectv2.main --config path/to/config.json
```

3. **Viewing Results**

```bash
cd experiments/visualization/web
python app.py
```

## Project Structure

```
dynadetectv2/
├── core/           # Core ML components
├── evaluation/     # Evaluation framework
├── config/        # Configuration management
└── attacks/       # Attack implementations

experiments/
├── visualization/  # Result visualization
└── utils/         # Utility functions
```

## Configuration

The system uses a configuration-driven approach. Key configuration components:

- Dataset configurations
- Model parameters
- Evaluation settings
- Visualization preferences

See `dynadetectv2/config/__init__.py` for configuration options.

## Results and Logging

- Results are stored in `results/` directory
- Logs are maintained in `logs/` directory
- Evaluation outputs include metrics, timing, and resource usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

[Insert License Information]
