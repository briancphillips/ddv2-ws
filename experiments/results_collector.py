"""
Results collection and analysis for DynaDetect v2
"""
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import asdict

class ResultsCollector:
    def __init__(self, output_dir: str = 'results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results_file = self.output_dir / 'results.json'
        # Load existing results if file exists
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    self.results = json.load(f)
            except json.JSONDecodeError:
                logging.warning("Could not load existing results, starting fresh")
                self.results = []
        else:
            self.results = []
        
    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a single experiment result."""
        result['timestamp'] = time.time()
        self.results.append(result)
        self._save_results()
    
    def _save_results(self) -> None:
        """Save results to JSON file."""
        # Create a temporary file first
        temp_file = self.results_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            # Atomic replace of the old file
            temp_file.replace(self.results_file)
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            if temp_file.exists():
                temp_file.unlink()
    
    def save_metrics(self, metrics: Dict[str, Any], config: Any, 
                    phase: str, iteration: int) -> None:
        """Save metrics for a specific configuration and phase."""
        result = {
            'config': asdict(config),
            'phase': phase,
            'iteration': iteration,
            **metrics
        }
        self.add_result(result)
    
    def analyze_results(self) -> None:
        """Analyze and visualize results."""
        if not self.results:
            logging.warning("No results to analyze")
            return
        
        # Create plots directory
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Convert results to DataFrame and extract config fields
        df = pd.DataFrame(self.results)
        config_df = pd.DataFrame([r['config'] for r in self.results])
        df = pd.concat([df.drop('config', axis=1), config_df], axis=1)
        
        # Generate plots
        self._plot_poison_impact(df, plots_dir)
        self._plot_latency_comparison(df, plots_dir)
        self._plot_computational_overhead(df, plots_dir)
        
        # Generate summary statistics
        self._generate_summary_stats(df)
    
    def _plot_poison_impact(self, df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot impact of poisoning on model performance."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='poison_rate', y='accuracy', hue='modification_method')
        plt.title('Impact of Poisoning on Model Performance')
        plt.xlabel('Poison Rate')
        plt.ylabel('Accuracy')
        plt.savefig(plots_dir / 'poison_impact.png')
        plt.close()
    
    def _plot_latency_comparison(self, df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot latency comparison across different phases."""
        # Skip latency plot if metrics are not available
        if 'execution_time' not in df.columns:
            logging.warning("Execution time metrics not found, skipping latency plot")
            return
            
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='phase', y='execution_time', hue='classifier')
        plt.title('Execution Time Comparison Across Phases')
        plt.xlabel('Phase')
        plt.ylabel('Execution Time (s)')
        plt.xticks(rotation=45)
        plt.savefig(plots_dir / 'execution_time_comparison.png')
        plt.close()
    
    def _plot_computational_overhead(self, df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot computational overhead comparison."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='modification_method', y='memory_usage', hue='classifier')
        plt.title('Computational Overhead by Attack Method')
        plt.xlabel('Attack Method')
        plt.ylabel('Memory Usage (MB)')
        plt.xticks(rotation=45)
        plt.savefig(plots_dir / 'computational_overhead.png')
        plt.close()
    
    def _generate_summary_stats(self, df: pd.DataFrame) -> None:
        """Generate summary statistics."""
        summary = {
            'total_experiments': len(df),
            'avg_accuracy': df['accuracy'].mean() if 'accuracy' in df else None,
            'avg_latency': df['latency'].mean() if 'latency' in df else None,
            'avg_memory_usage': df['memory_usage'].mean() if 'memory_usage' in df else None
        }
        
        # Save summary to file
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
