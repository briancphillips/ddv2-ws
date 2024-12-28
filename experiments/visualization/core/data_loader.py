"""Data loading and processing utilities for visualization."""
import os
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime

class ResultsLoader:
    """Load and process experiment results for visualization."""
    
    def __init__(self, results_dir: str = "results"):
        """Initialize the loader with results directory path."""
        self.results_dir = results_dir
        self.archive_dir = os.path.join(results_dir, "archive")
        self._cached_data = None
        
    def load_latest_results(self, include_archived: bool = True) -> pd.DataFrame:
        """Load the most recent results file and optionally combine with archived results."""
        results_files = self._get_results_files()
        if not results_files:
            raise FileNotFoundError("No results files found")
            
        dfs = []
        for file in results_files:
            try:
                # Read the data with on_bad_lines='skip' to handle malformed rows
                df = pd.read_csv(file, on_bad_lines='skip')
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue
        
        if not dfs:
            raise FileNotFoundError("No valid results files found")
            
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Convert only numeric columns, keeping categorical columns as strings
        numeric_cols = ['iteration', 'total_images', 'num_poisoned', 'latency', 
                       'accuracy', 'precision', 'recall', 'f1']
        numeric_cols.extend([col for col in combined_df.columns if 
                           any(col.startswith(prefix) for prefix in 
                               ['precision_class_', 'recall_class_', 'f1_class_'])])
        
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                combined_df[col] = combined_df[col].fillna(0)  # Replace NaN with 0
        
        return combined_df
    
    def load_results_by_timestamp(self, timestamp: str) -> pd.DataFrame:
        """Load results file for specific timestamp."""
        filepath = os.path.join(self.results_dir, f"experiment_results_{timestamp}.csv")
        if not os.path.exists(filepath):
            filepath = os.path.join(self.archive_dir, f"experiment_results_{timestamp}.csv")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No results file found for timestamp {timestamp}")
            
        return pd.read_csv(filepath, on_bad_lines='skip')
    
    def load_all_results(self) -> pd.DataFrame:
        """Load and combine all results files."""
        if self._cached_data is not None:
            return self._cached_data
            
        self._cached_data = self.load_latest_results(include_archived=True)
        return self._cached_data
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics from results."""
        df = self.load_latest_results()
        metrics = [col for col in df.columns if any(
            col.startswith(prefix) for prefix in 
            ['accuracy', 'precision', 'recall', 'f1', 'latency']
        )]
        return sorted(metrics)
    
    def get_unique_values(self, column: str) -> List[Any]:
        """Get unique values for a given column."""
        df = self.load_latest_results()
        return sorted(df[column].unique().tolist())
    
    def _get_results_files(self) -> List[str]:
        """Get list of all results files including archived ones."""
        files = []
        
        # Check current results
        if os.path.exists(self.results_dir):
            files.extend([
                os.path.join(self.results_dir, f) 
                for f in os.listdir(self.results_dir)
                if f.endswith('.csv')
            ])
            
        # Check archived results
        if os.path.exists(self.archive_dir):
            files.extend([
                os.path.join(self.archive_dir, f)
                for f in os.listdir(self.archive_dir)
                if f.endswith('.csv')
            ])
            
        return files 