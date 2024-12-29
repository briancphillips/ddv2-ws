# Technical Documentation

## Core Components

### Model Implementations (`core/models.py`)

1. **DDKNN (Dynamic Distance-based k-Nearest Neighbors)**

   - PyTorch implementation with gradient computation support
   - Soft voting mechanism with temperature parameter
   - Distance-weighted predictions
   - GPU acceleration support

2. **LogisticRegressionWrapper**

   - GPU-accelerated implementation
   - Early stopping support
   - Batch processing
   - Configurable learning rate and regularization
   - Memory-efficient training

3. **SVMWrapper**

   - GPU-accelerated implementation
   - Hinge loss optimization
   - Batch processing support
   - Early stopping capability

4. **RandomForestWrapper**
   - PyTorch implementation
   - Parallel tree training
   - Early stopping support
   - Batch prediction capability

### Dataset Handling (`core/dataset.py`)

1. **Feature Extraction**

   - WideResNet-based feature extractor
   - Caching mechanism for extracted features
   - Support for GTSRB dataset
   - Automatic data preprocessing

2. **Data Poisoning**
   - Label flipping attack implementation
   - Multiple flipping strategies:
     - Random-to-random
     - Targeted flipping
     - Source-target specific
   - Configurable poison rates

### Evaluation Framework (`evaluation/evaluator.py`)

1. **ModelEvaluator**

   - Comprehensive metrics calculation
   - Performance monitoring
   - Resource usage tracking
   - Timeout handling
   - Results logging

2. **DatasetEvaluator**
   - Dataset-specific evaluation logic
   - Attack impact assessment
   - Cross-validation support
   - Batch processing

## Implementation Details

### Training Process

1. **Feature Extraction**

   ```python
   # Initialize feature extractor
   extractor = FeatureExtractor(dataset_name)

   # Extract and cache features
   features = extractor(data)
   ```

2. **Model Training**

   ```python
   # Initialize model with GPU support
   model = ModelFactory.create_model(classifier_name)

   # Train with early stopping
   model.fit(features, labels,
            early_stopping=True,
            validation_fraction=0.1)
   ```

3. **Evaluation**

   ```python
   # Initialize evaluator
   evaluator = ModelEvaluator(classifier_name, mode='standard')

   # Run evaluation with timeout
   with time_limit(timeout):
       predictions = model.predict(test_features)
       metrics = evaluator.compute_metrics(predictions, true_labels)
   ```

### Attack Implementation

1. **Label Flipping**
   ```python
   def label_flipping(labels, mode, poison_rate=0.0):
       if mode == 'random_to_random':
           # Randomly flip labels
           flip_indices = np.random.choice(...)
       elif mode == 'targeted':
           # Target specific classes
           flip_indices = np.where(...)
       return modified_labels
   ```

### Resource Management

1. **Memory Optimization**

   - Batch processing for large datasets
   - Feature caching with size limits
   - GPU memory management
   - Automatic cleanup

2. **Performance Monitoring**
   - CPU/GPU utilization tracking
   - Memory usage monitoring
   - Training/inference latency measurement
   - Resource limit enforcement

## Configuration System

1. **ExperimentConfig**

   - Dataset parameters
   - Model hyperparameters
   - Evaluation settings
   - Attack configurations

2. **Logging System**
   - Timestamped logs
   - Automatic archiving
   - Performance metrics
   - Error tracking

## Results Management

1. **Data Format**

   - CSV output with comprehensive metrics
   - Per-class performance measures
   - Attack impact statistics
   - Resource utilization data

2. **Visualization**
   - Performance plots
   - Attack impact visualization
   - Resource utilization graphs
   - Comparative analysis tools
