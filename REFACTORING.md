# DynaDetect v2 Refactoring Documentation

## Architecture Overview

The project has been refactored into a clean, modular architecture with clear separation of concerns:

### Core Package (`dynadetectv2/`)

- **Core Module** (`core/`)
  - `dataset.py`: Dataset handling and preprocessing
  - `models.py`: Model implementations and factory
  - `trainer.py`: Training logic and utilities
- **Attacks** (`attacks/`): Attack implementations
- **Evaluation** (`evaluation/`): Metrics and analysis
- **Config** (`config/`): Configuration management
- **Main** (`main.py`): Entry point and orchestration

### Supporting Modules (`experiments/`)

- **Results Management** (`results_management/`)
  - Results collection and storage
  - Visualization utilities
- **Utils** (`utils/`)
  - Memory tracking
  - Performance monitoring
- **Evaluation Runner** (`run_full_evaluation.py`)

## Key Changes

1. **Package Structure**

   - Moved from monolithic design to modular package
   - Clear separation between core and supporting functionality
   - Proper Python package organization

2. **Code Organization**

   - Separated concerns into distinct modules
   - Removed redundant code
   - Improved code reusability

3. **Testing**

   - Added proper test infrastructure
   - Implemented test fixtures
   - Improved test coverage

4. **Configuration**

   - Centralized configuration management
   - Type-safe configuration objects
   - Flexible experiment configuration

5. **Results Management**
   - Structured results collection
   - Enhanced visualization capabilities
   - Memory usage tracking

## Design Decisions

1. **Core vs. Experiments**

   - Core functionality in `dynadetectv2/`
   - Supporting tools in `experiments/`
   - Clear boundary between framework and tools

2. **Configuration System**

   - Use of dataclasses for type safety
   - Hierarchical configuration structure
   - Flexible parameter management

3. **Results Management**

   - Standardized results format
   - Comprehensive metrics collection
   - Extensible visualization system

4. **Testing Strategy**
   - Unit tests for core components
   - Integration tests for workflows
   - Fixtures for common setup

## Future Considerations

1. **Performance Optimization**

   - Memory usage optimization
   - Training speed improvements
   - GPU utilization

2. **Extensibility**

   - Plugin system for new attacks
   - Custom model support
   - Dataset extensions

3. **Documentation**
   - API documentation
   - Usage examples
   - Architecture guides
