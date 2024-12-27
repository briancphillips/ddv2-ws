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

## High Priority

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
