"""
ResultsManager class for handling experiment results in DynaDetect v2.
This class encapsulates all result saving, processing, and export functionality.
"""

import logging
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

class ResultsManager:
    def __init__(self, base_dir: str = 'results'):
        """
        Initialize ResultsManager.
        
        Args:
            base_dir: Base directory for saving results
        """
        self.base_dir = base_dir
        self.logger = logging.getLogger(__name__)
        self._ensure_directories()
        self._current_batch = []
        
    def _ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(self.base_dir, exist_ok=True)
        
    def _generate_timestamp(self) -> str:
        """Generate timestamp for file naming."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _flatten_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten a result dictionary for CSV export with clean column names.
        
        Args:
            result: Raw result dictionary
            
        Returns:
            Flattened dictionary with clean column names
        """
        flat_result = {
            'Dataset': result['dataset'],
            'Classifier': result['classifier'],
            'Mode': result['mode'],
            'Attack Method': result['modification_method'],
            'Poison Rate': result['poison_rate'],
            'Poisoned Samples': result['num_poisoned'],
            'Total Samples': result.get('total_samples', ''),
            'Train Accuracy': result['train_metrics']['accuracy'],
            'Train Precision': result['train_metrics']['precision'],
            'Train Recall': result['train_metrics']['recall'],
            'Train F1': result['train_metrics']['f1'],
            'Test Accuracy': result['test_metrics']['accuracy'],
            'Test Precision': result['test_metrics']['precision'],
            'Test Recall': result['test_metrics']['recall'],
            'Test F1': result['test_metrics']['f1'],
            'Latency (s)': result['latency'],
            'Memory Usage (MB)': result.get('memory_usage', ''),
            'CPU Usage (%)': result.get('cpu_usage', '')
        }
        
        # Add attack-specific parameters
        if result['modification_method'] == 'label_flipping':
            flat_result['Attack Type'] = result['attack_params'].get('flip_type', '')
        elif result['modification_method'] == 'pgd':
            flat_result['Attack Strength'] = result['attack_params'].get('eps', '')
            
        # Add per-class metrics in a clean format
        for class_id, class_metrics in result['train_metrics']['per_class_metrics'].items():
            for metric_name, value in class_metrics.items():
                flat_result['Train Class {} {}'.format(class_id, metric_name.title())] = value
                
        for class_id, class_metrics in result['test_metrics']['per_class_metrics'].items():
            for metric_name, value in class_metrics.items():
                flat_result['Test Class {} {}'.format(class_id, metric_name.title())] = value
                
        return flat_result
        
    def _create_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a clean summary of a result with only essential metrics.
        
        Args:
            result: Raw result dictionary
            
        Returns:
            Summary dictionary with clean column names and essential metrics only
        """
        # Extract only the essential metrics
        train_metrics = result['train_metrics']
        test_metrics = result['test_metrics']
        
        summary = {
            'Dataset': result['dataset'],
            'Classifier': result['classifier'],
            'Mode': result['mode'],
            'Attack Method': result['modification_method'],
            'Poison Rate': result['poison_rate'],
            'Poisoned Samples': result['num_poisoned'],
            'Total Samples': result.get('total_samples', ''),
            'Train Accuracy': train_metrics['accuracy'],
            'Train F1': train_metrics['f1'],
            'Test Accuracy': test_metrics['accuracy'],
            'Test F1': test_metrics['f1'],
            'Latency (s)': result['latency']
        }
        
        # Add attack-specific parameters
        if result['modification_method'] == 'label_flipping':
            summary['Attack Type'] = result['attack_params'].get('flip_type', '')
        elif result['modification_method'] == 'pgd':
            summary['Attack Strength'] = result['attack_params'].get('eps', '')
            
        return summary
        
    def add_result(self, result: Dict[str, Any]):
        """
        Add a result to the current batch.
        
        Args:
            result: Result dictionary
        """
        self._current_batch.append(result)
        
    def save_batch(self, timestamp: Optional[str] = None):
        """
        Save the current batch of results.
        
        Args:
            timestamp: Optional timestamp to use for file naming
        """
        if not self._current_batch:
            self.logger.warning("No results to save")
            return
            
        timestamp = timestamp or self._generate_timestamp()
        
        # Save full results to JSON for reference
        json_file = os.path.join(self.base_dir, f'experiment_results_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump(self._current_batch, f, indent=4)
        self.logger.info(f"Results saved to JSON: {json_file}")
        
        # Create detailed DataFrame with flattened metrics
        detailed_results = []
        for result in self._current_batch:
            row = {
                'Dataset': result['dataset'],
                'Classifier': result['classifier'],
                'Mode': result['mode'],
                'Attack Method': result['modification_method'],
                'Poison Rate': result['poison_rate'],
                'Num Poisoned': result['num_poisoned'],
                'Train Accuracy': result['train_metrics']['accuracy'],
                'Train Precision': result['train_metrics'].get('precision_macro', result['train_metrics'].get('precision', 0)),
                'Train Recall': result['train_metrics'].get('recall_macro', result['train_metrics'].get('recall', 0)),
                'Train F1': result['train_metrics'].get('f1', 0),
                'Test Accuracy': result['test_metrics']['accuracy'],
                'Test Precision': result['test_metrics'].get('precision_macro', result['test_metrics'].get('precision', 0)),
                'Test Recall': result['test_metrics'].get('recall_macro', result['test_metrics'].get('recall', 0)),
                'Test F1': result['test_metrics'].get('f1', 0),
                'Latency (s)': result['latency']
            }
            
            # Add attack-specific parameters
            if result['modification_method'] == 'label_flipping':
                row['Attack Type'] = result['attack_params'].get('flip_type', '')
            elif result['modification_method'] == 'pgd':
                row['Attack Type'] = f"eps={result['attack_params'].get('eps', '')}"
                
            # Add per-class metrics
            for class_id, class_metrics in result['train_metrics']['per_class_metrics'].items():
                for metric_name, value in class_metrics.items():
                    if metric_name != 'support':  # Skip support counts
                        row[f'Train Class {class_id} {metric_name}'] = value
                        
            for class_id, class_metrics in result['test_metrics']['per_class_metrics'].items():
                for metric_name, value in class_metrics.items():
                    if metric_name != 'support':  # Skip support counts
                        row[f'Test Class {class_id} {metric_name}'] = value
                        
            # Add resource usage if available
            if result.get('resource_usage'):
                resource_data = result['resource_usage'][-1] if isinstance(result['resource_usage'], list) else result['resource_usage']
                row['CPU Usage (%)'] = resource_data.get('cpu_percent', 0)
                row['Memory Usage (MB)'] = resource_data.get('memory_info', {}).get('rss', 0) / (1024 * 1024)
                
            detailed_results.append(row)
            
        detailed_df = pd.DataFrame(detailed_results)
        detailed_csv = os.path.join(self.base_dir, f'experiment_results_{timestamp}.csv')
        detailed_df.to_csv(detailed_csv, index=False)
        
        # Create summary DataFrame with essential metrics only
        summary_results = []
        for result in self._current_batch:
            row = {
                'Dataset': result['dataset'],
                'Classifier': result['classifier'],
                'Mode': result['mode'],
                'Attack Method': result['modification_method'],
                'Attack Type': result['attack_params'].get('flip_type', '') if result['modification_method'] == 'label_flipping' else f"eps={result['attack_params'].get('eps', '')}",
                'Poison Rate': result['poison_rate'],
                'Num Poisoned': result['num_poisoned'],
                'Train Accuracy': result['train_metrics']['accuracy'],
                'Train F1': result['train_metrics'].get('f1', 0),
                'Test Accuracy': result['test_metrics']['accuracy'],
                'Test F1': result['test_metrics'].get('f1', 0),
                'Latency (s)': result['latency']
            }
            summary_results.append(row)
            
        summary_df = pd.DataFrame(summary_results)
        
        # Format numeric columns
        numeric_columns = [
            'Train Accuracy', 'Train F1',
            'Test Accuracy', 'Test F1',
            'Latency (s)'
        ]
        for col in numeric_columns:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].round(4)
        
        summary_csv = os.path.join(self.base_dir, f'experiment_results_{timestamp}_summary.csv')
        summary_df.to_csv(summary_csv, index=False)
        
        self.logger.info(f"Detailed results saved to CSV: {detailed_csv} (Total rows: {len(detailed_df)})")
        self.logger.info(f"Summary results saved to CSV: {summary_csv} (Total rows: {len(summary_df)})")
        
        # Clear the current batch
        self._current_batch = []
        
    def load_results(self, timestamp: str) -> List[Dict[str, Any]]:
        """
        Load results from a specific timestamp.
        
        Args:
            timestamp: Timestamp of results to load
            
        Returns:
            List of result dictionaries
        """
        json_file = os.path.join(self.base_dir, 'experiment_results_{}.json'.format(timestamp))
        if not os.path.exists(json_file):
            raise FileNotFoundError("No results found for timestamp: {}".format(timestamp))
            
        with open(json_file, 'r') as f:
            results = json.load(f)
            
        return results
``` 