"""
Visualization module for DynaDetect v2 results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ResultsVisualizer:
    def __init__(self, base_dir: str = 'results'):
        """
        Initialize ResultsVisualizer.
        
        Args:
            base_dir: Base directory for saving visualizations
        """
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Ensure required directories exist."""
        self.base_dir.mkdir(exist_ok=True)
        (self.base_dir / 'plots').mkdir(exist_ok=True)
        
    def plot_poisoning_impact(self, results_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot the impact of poisoning on model performance.
        
        Args:
            results_df: DataFrame containing results
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=results_df, x='poison_rate', y='test_accuracy', 
                    hue='classifier', style='modification_method')
        plt.title('Impact of Poisoning on Model Performance')
        plt.xlabel('Poison Rate')
        plt.ylabel('Test Accuracy')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_latency_comparison(self, results_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot latency comparison across different classifiers.
        
        Args:
            results_df: DataFrame containing results
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=results_df, x='classifier', y='latency')
        plt.title('Latency Comparison Across Classifiers')
        plt.xlabel('Classifier')
        plt.ylabel('Latency (seconds)')
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_performance_heatmap(self, results_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create a heatmap of model performance across different configurations.
        
        Args:
            results_df: DataFrame containing results
            save_path: Optional path to save the plot
        """
        pivot_table = pd.pivot_table(
            results_df,
            values='test_accuracy',
            index='classifier',
            columns='modification_method',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('Performance Heatmap: Classifier vs. Attack Method')
        
        if save_path:
            plt.savefig(save_path)
        plt.close() 