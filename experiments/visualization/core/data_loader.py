"""Data loading and processing utilities for visualization."""
import os
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime

class ResultsLoader:
    """Load and process experiment results for visualization."""
    
    def __init__(self, results_dir: str = "results"):
        """Initialize the loader with results directory path."""
        # Get absolute path to workspace root
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.results_dir = os.path.join(workspace_root, "results")
        self.archive_dir = os.path.join(self.results_dir, "archive")
        self._cached_data = None
        print(f"Results directory: {self.results_dir}")  # Debug print
        
    def load_latest_results(self, include_archived: bool = True) -> pd.DataFrame:
        """Load the most recent results file."""
        results_files = self._get_results_files()
        print(f"Found results files: {results_files}")  # Debug print
        if not results_files:
            raise FileNotFoundError(f"No results files found in {self.results_dir}")
        
        try:
            latest_file = results_files[0]
            print(f"Loading latest file: {latest_file}")  # Debug print
            # Read only the most recent file
            df = pd.read_csv(latest_file, on_bad_lines='skip')
            print(f"Loaded data shape: {df.shape}")  # Debug print
            
            # Convert only numeric columns, keeping categorical columns as strings
            numeric_cols = ['iteration', 'total_images', 'num_poisoned', 'latency', 
                           'accuracy', 'precision', 'recall', 'f1']
            numeric_cols.extend([col for col in df.columns if 
                               any(col.startswith(prefix) for prefix in 
                                   ['precision_class_', 'recall_class_', 'f1_class_'])])
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0)  # Replace NaN with 0
            
            return df
        except Exception as e:
            print(f"Error loading {results_files[0]}: {str(e)}")  # Debug print
            raise FileNotFoundError(f"Could not load the latest results file: {str(e)}")
    
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
                if f.endswith('.csv') and f != 'archive'
            ])
            
            # Sort files by timestamp in filename
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Print files found for debugging
            print(f"Files in {self.results_dir}:")
            for f in os.listdir(self.results_dir):
                print(f"  - {f}")
        else:
            print(f"Results directory not found: {self.results_dir}")
        
        # Return only the most recent file
        return files[:1] if files else [] 