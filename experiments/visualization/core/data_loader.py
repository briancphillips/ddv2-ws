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
        
    def load_latest_results(self) -> pd.DataFrame:
        """Load the most recent results file."""
        results_files = self._get_results_files()
        if not results_files:
            raise FileNotFoundError("No results files found")
            
        latest_file = max(results_files, key=os.path.getctime)
        return pd.read_csv(latest_file)
    
    def load_results_by_timestamp(self, timestamp: str) -> pd.DataFrame:
        """Load results file for specific timestamp."""
        filepath = os.path.join(self.results_dir, f"experiment_results_{timestamp}.csv")
        if not os.path.exists(filepath):
            filepath = os.path.join(self.archive_dir, f"experiment_results_{timestamp}.csv")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No results file found for timestamp {timestamp}")
            
        return pd.read_csv(filepath)
    
    def load_all_results(self) -> pd.DataFrame:
        """Load and combine all results files."""
        if self._cached_data is not None:
            return self._cached_data
            
        results_files = self._get_results_files()
        if not results_files:
            raise FileNotFoundError("No results files found")
            
        dfs = []
        for file in results_files:
            df = pd.read_csv(file)
            dfs.append(df)
            
        self._cached_data = pd.concat(dfs, ignore_index=True)
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