"""Data loading and processing utilities for visualization."""
import os
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime

class ResultsLoader:
    """Load and process experiment results for visualization."""
    
    def __init__(self, results_dir: str = "results"):
        """Initialize the loader with results directory path."""
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.results_dir = os.path.join(workspace_root, "results")
        self._cached_data = None
    
    def load_latest_results(self) -> pd.DataFrame:
        """Load the results file from the results directory."""
        try:
            # Get all CSV files in results directory
            files = [f for f in os.listdir(self.results_dir) if f.endswith('.csv')]
            if not files:
                raise FileNotFoundError(f"No results file found in {self.results_dir}")
            
            # Load the file
            filepath = os.path.join(self.results_dir, files[0])
            df = pd.read_csv(filepath)
            
            # Convert numeric columns
            numeric_cols = ['iteration', 'total_images', 'num_poisoned', 'latency', 
                          'accuracy', 'precision', 'recall', 'f1']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0)
            
            return df
            
        except Exception as e:
            print(f"Error loading results: {str(e)}")
            raise
    
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