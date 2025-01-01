# Technical Documentation

## Core Components

### Model Implementations (`core/models.py`)

1. **DDKNN (Dynamic Distance-based k-Nearest Neighbors)**

   - PyTorch implementation with gradient computation support
   - Soft voting mechanism with temperature parameter
   - Distance-weighted predictions
   - GPU acceleration support
   - Enhanced batch processing capabilities
   - Improved memory management

2. **LogisticRegressionWrapper**

   - GPU-accelerated implementation
   - Early stopping support
   - Batch processing
   - Configurable learning rate and regularization
   - Memory-efficient training
   - Optimized CUDA operations

3. **SVMWrapper**

   - GPU-accelerated implementation
   - Hinge loss optimization
   - Batch processing support
   - Early stopping capability
   - Enhanced memory efficiency

4. **RandomForestWrapper**
   - PyTorch implementation
   - Parallel tree training
   - Early stopping support
   - Batch prediction capability
   - Improved resource utilization

### Dataset Handling (`core/dataset.py`)

1. **Feature Extraction**

   - WideResNet-based feature extractor
   - Caching mechanism for extracted features
   - Support for GTSRB dataset
   - Automatic data preprocessing
   - Enhanced validation checks
   - Improved error handling

2. **Data Poisoning**
   - Label flipping attack implementation
   - Multiple flipping strategies:
     - Random-to-random
     - Targeted flipping
     - Source-target specific
   - Configurable poison rates
   - Enhanced attack monitoring

### Evaluation Framework (`evaluation/evaluator.py`)

1. **ModelEvaluator**

   - Comprehensive metrics calculation
   - Performance monitoring
   - Resource usage tracking
   - Timeout handling
   - Results logging
   - Enhanced experiment archiving
   - Real-time progress tracking

2. **DatasetEvaluator**
   - Dataset-specific evaluation logic
   - Attack impact assessment
   - Cross-validation support
   - Batch processing
   - Improved validation checks

## Implementation Details

### Training Process

1. **Feature Extraction**

   ```python
   # Initialize feature extractor with enhanced validation
   extractor = FeatureExtractor(dataset_name, validation_mode='strict')

   # Extract and cache features with monitoring
   features = extractor(data, monitor_resources=True)
   ```

2. **Model Training**

   ```python
   # Initialize model with GPU support and enhanced memory management
   model = ModelFactory.create_model(
       classifier_name,
       memory_efficient=True,
       cuda_optimize=True
   )

   # Train with early stopping and resource monitoring
   model.fit(
       features,
       labels,
       early_stopping=True,
       validation_fraction=0.1,
       monitor_resources=True
   )
   ```

3. **Evaluation**

   ```python
   # Initialize evaluator with enhanced monitoring
   evaluator = ModelEvaluator(
       classifier_name,
       mode='standard',
       resource_monitoring=True
   )

   # Run evaluation with timeout and progress tracking
   with time_limit(timeout):
       predictions = model.predict(test_features)
       metrics = evaluator.compute_metrics(
           predictions,
           true_labels,
           track_progress=True
       )
   ```

### Attack Implementation

1. **Label Flipping**
   ```python
   def label_flipping(labels, mode, poison_rate=0.0, monitor=True):
       if mode == 'random_to_random':
           # Randomly flip labels with monitoring
           flip_indices = np.random.choice(...)
       elif mode == 'targeted':
           # Target specific classes with validation
           flip_indices = np.where(...)
       return modified_labels, monitoring_stats
   ```

### Resource Management

1. **Memory Optimization**

   - Batch processing for large datasets
   - Feature caching with size limits
   - GPU memory management
   - Automatic cleanup
   - Enhanced memory defragmentation
   - Improved resource monitoring

2. **Performance Monitoring**
   - CPU/GPU utilization tracking
   - Memory usage monitoring
   - Training/inference latency measurement
   - Resource limit enforcement
   - Real-time progress tracking
   - Enhanced experiment archiving

## Configuration System

1. **ExperimentConfig**

   - Dataset parameters
   - Model hyperparameters
   - Evaluation settings
   - Attack configurations
   - Resource monitoring options
   - Enhanced validation rules

2. **Logging System**
   - Timestamped logs
   - Automatic archiving
   - Performance metrics
   - Error tracking
   - Resource utilization logging
   - Progress monitoring

## Results Management

1. **Data Format**

   - CSV output with comprehensive metrics
   - Per-class performance measures
   - Attack impact statistics
   - Resource utilization data
   - Enhanced experiment metadata
   - Progress tracking information

2. **Visualization**
   - Performance plots
   - Attack impact visualization
   - Resource utilization graphs
   - Comparative analysis tools
   - Real-time progress monitoring
   - Enhanced data exploration
