I need help rebuilding the DynaDetect visualization app in a modular way. The app needs to:

1. Display experiment results with:

   - Support for multiple datasets
   - Various plot types (Line, Bar, Box, Scatter)
   - Flexible grouping and coloring
   - Smart handling of poison rates and metrics
   - Dynamic UI controls that adapt based on context

2. Core requirements:

   - Must handle combined vs separate dataset views
   - Must properly group and aggregate data
   - Must support dynamic axis ranges
   - Must maintain proper data separation when combining datasets
   - Must have smart controls that disable invalid combinations

3. The rebuild should be organized into these modules:

   - data_processing.py: Data loading, transformation, and aggregation
   - plot_config.py: Plot configuration and generation
   - ui_controls.py: UI state management and control logic
   - stats.py: Statistics calculation and presentation
   - app.py: Main streamlit application

4. Key data structures needed:
   - Dataset configurations (sizes, poison rates)
   - Plot configurations
   - UI state management
   - Data transformation pipelines

Please help me implement this in a clean, modular way, starting with the core data structures and working outward to the UI.

Current rate_mappings for reference:
{
'GTSRB': {0: 0, 1: 392, 3: 1176, 5: 1960, 7: 2744, 10: 3920, 20: 7841},
'CIFAR100': {0: 0, 1: 500, 3: 1500, 5: 2500, 7: 3500, 10: 5000, 20: 10000},
'ImageNette': {0: 0, 1: 94, 3: 284, 5: 473, 7: 662, 10: 946, 20: 1893}
}
