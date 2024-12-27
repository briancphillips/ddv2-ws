# DynaDetect v2 TODO List

## Completed

- [x] Core framework implementation
  - [x] Dataset handling and preprocessing
  - [x] Model implementations
  - [x] Training utilities
  - [x] Attack implementations
- [x] Project restructuring
  - [x] Proper package organization
  - [x] Clean separation of concerns
  - [x] Removal of legacy code
- [x] Testing infrastructure
  - [x] Basic test suite
  - [x] Test fixtures
  - [x] CI setup
- [x] Results management
  - [x] Visualization tools
  - [x] Results collection
  - [x] Memory tracking
- [x] Dataset Improvements
  - [x] GTSRB dataset structure standardization
  - [x] Consistent directory organization
  - [x] Robust CSV handling for test set
- [x] Configuration Updates
  - [x] Updated to use full dataset sizes
  - [x] Included all label flipping attack modes
  - [x] Optimized configuration for overnight runs
- [x] Performance Optimizations
  - [x] GPU acceleration for SVM model
  - [x] GPU acceleration for Logistic Regression
  - [x] GPU acceleration for Random Forest
  - [x] GPU acceleration for KNeighbors
  - [x] Feature scaling for memory efficiency
  - [x] GPU acceleration for feature extraction
  - [x] GPU acceleration for anomaly detection
  - [x] Caching system for feature extraction
  - [x] Caching system for sample weights
  - [x] Caching system for attack generation

## In Progress

- [ ] Running full evaluation with:
  - [ ] Full dataset sizes
  - [ ] All label flipping modes (random_to_random, random_to_target, source_to_target)
  - [ ] All classifiers with GPU acceleration
  - [ ] Comprehensive metrics collection

## High Priority

- [x] GTSRB Dataset Fix
  - [x] Switched to torchvision's GTSRB dataset
  - [x] Fixed class distribution issue
  - [x] Ensured all 43 classes are available
  - [x] Fixed targets attribute handling for train/val/test splits
- [ ] SVM Optimization
  - [ ] Implement GPU-accelerated SVM (ThunderSVM/cuSVM)
  - [ ] Add model checkpointing for SVM
  - [ ] Experiment with further dimensionality reduction
  - [ ] Cache intermediate SVM training results
- [ ] Performance Optimization
  - [ ] Memory leak in feature extraction
  - [ ] Data loading pipeline efficiency
  - [ ] Training speed improvements
- [ ] Model Improvements
  - [ ] Cross-validation for hyperparameters
  - [ ] Early stopping implementation
  - [ ] Model checkpointing
- [ ] Testing
  - [ ] Edge case coverage
  - [ ] Integration tests
  - [ ] Performance benchmarks

## Medium Priority

- [ ] Feature Additions
  - [ ] Data augmentation pipeline
  - [ ] Additional attack methods
  - [ ] More visualization options
- [ ] Documentation
  - [ ] API documentation
  - [ ] Usage examples
  - [ ] Architecture overview
- [ ] Code Quality
  - [ ] Type hints completion
  - [ ] Error handling improvements
  - [ ] Logging enhancements

## Low Priority

- [ ] Infrastructure
  - [ ] Distributed training support
  - [ ] Docker containerization
  - [ ] Cloud deployment guides
- [ ] Developer Experience
  - [ ] Development environment setup
  - [ ] Contributing guidelines
  - [ ] Code style guide

## Bugs

- [ ] Memory leak in feature extraction
- [ ] Convergence warnings in LogisticRegression
- [ ] Undefined metrics in evaluation
- [ ] GPU memory management issues
