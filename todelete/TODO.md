# DynaDetect v2 TODO List

## Completed Tasks

- [x] Migrated visualization code to new `results_management/visualizer.py`
- [x] Created proper test structure with pytest fixtures
- [x] Added memory tracking utilities in `experiments/utils/memory_tracker.py`
- [x] Cleaned up old files and migrated functionality
- [x] Removed old monolithic dynadetectv2.py after code refactoring
- [x] Removed redundant experiment directories after migration to dynadetectv2 package
- [x] Updated image preprocessing for ResNet compatibility
- [x] Fixed sample size configuration
- [x] Improved results management and visualization

## In Progress

- [ ] Implement cross-validation for hyperparameter tuning
- [ ] Add data augmentation pipeline
- [ ] Optimize feature extraction process

## To Do

- [ ] Add more test cases for edge cases
- [ ] Implement early stopping for training
- [ ] Add model checkpointing
- [ ] Create comprehensive documentation
- [ ] Add performance profiling
- [ ] Implement distributed training support
- [ ] Add more visualization options
- [ ] Create benchmark suite
- [ ] Add CI/CD pipeline

## Bugs to Fix

- [ ] Fix memory leak in feature extraction
- [ ] Address convergence warnings in LogisticRegression
- [ ] Handle undefined metrics in evaluation

## Improvements

- [ ] Optimize data loading pipeline
- [ ] Reduce memory usage during training
- [ ] Improve error handling and logging
- [ ] Add progress bars for long-running operations
- [ ] Implement better exception handling
- [ ] Add type hints throughout the codebase
- [ ] Improve code documentation
