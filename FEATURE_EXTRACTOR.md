# Feature Extraction System

## Overview

The feature extraction system in DynaDetect v2 uses a WideResNet architecture to extract meaningful features from input data, particularly optimized for the GTSRB dataset.

## Implementation Details

### Architecture (`core/dataset.py`)

1. **WideResNet Components**

   ```python
   class BasicBlock(nn.Module):
       # Basic building block with residual connections
       def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
           # Convolutional layers with batch normalization
           # Dropout for regularization
           # Residual connections

   class NetworkBlock(nn.Module):
       # Stacks multiple basic blocks
       def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
           # Creates sequence of basic blocks
           # Manages layer connections

   class WideResNet(nn.Module):
       # Main WideResNet architecture
       def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
           # Configurable depth and width
           # Adaptive to different input sizes
           # Multiple conv layers with increasing channels
   ```

2. **Feature Extractor**

   ```python
   class FeatureExtractor(nn.Module):
       def __init__(self, dataset_name: str):
           # Initializes appropriate architecture
           # Loads pretrained weights if available
           # Configures for specific dataset

       def forward(self, x):
           # Extracts features through network
           # Returns feature vectors
   ```

## Caching System

1. **Implementation**

   ```python
   class DatasetHandler:
       def __init__(self, dataset_name: str, data_dir: str = ".datasets"):
           self.cache_dir = os.path.join(data_dir, dataset_name, "cache")
           self.feature_cache = {}

       def get_cached_features(self, dataset, cache_key):
           # Check cache directory
           # Load cached features if available
           # Extract and cache if not found
   ```

2. **Cache Management**
   - Automatic cache directory creation
   - Cache invalidation on model changes
   - Memory-efficient loading
   - Disk space management

## Dataset Support

1. **GTSRB Dataset**

   ```python
   class GTSRBDataset(Dataset):
       def __init__(self, root_dir, train=True, transform=None, val_split=0.2):
           # Loads GTSRB data
           # Applies transformations
           # Handles train/val split
   ```

2. **Data Preprocessing**
   - Image resizing and normalization
   - Data augmentation (when applicable)
   - Label processing
   - Batch preparation

## Feature Extraction Process

1. **Initialization**

   ```python
   # Create feature extractor
   extractor = FeatureExtractor(dataset_name="GTSRB")

   # Load pretrained weights
   extractor.load_pretrained_wrn()
   ```

2. **Feature Extraction**

   ```python
   # Extract features with caching
   features = dataset_handler.get_cached_features(
       dataset=train_dataset,
       cache_key="train_features"
   )
   ```

3. **Memory Management**
   - Batch processing for large datasets
   - GPU memory optimization
   - Cache size limits
   - Automatic cleanup

## Performance Considerations

1. **Optimization Techniques**

   - GPU acceleration when available
   - Efficient tensor operations
   - Batch size optimization
   - Memory-mapped file handling

2. **Resource Management**
   - Memory usage monitoring
   - Cache size control
   - Disk space management
   - Cleanup procedures

## Usage Examples

1. **Basic Usage**

   ```python
   # Initialize handler
   handler = DatasetHandler(dataset_name="GTSRB")

   # Get dataset
   train_dataset = handler.get_train_data()

   # Extract features
   features = handler.get_cached_features(train_dataset, "train")
   ```

2. **Custom Configuration**

   ```python
   # Initialize with custom parameters
   extractor = FeatureExtractor(
       dataset_name="GTSRB",
       cache_dir="custom/cache/path"
   )

   # Extract with specific batch size
   features = extractor.extract_batch(
       data_batch,
       batch_size=32
   )
   ```

## Current Limitations

1. **Memory Usage**

   - High memory consumption for large datasets
   - Limited by GPU memory when available
   - Cache size grows with dataset

2. **Performance**

   - Initial extraction can be slow
   - Cache validation overhead
   - Disk I/O bottlenecks possible

3. **Dataset Support**
   - Primarily optimized for GTSRB
   - Limited support for other datasets
   - Fixed input size requirements
