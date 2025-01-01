"""Data processing utilities for advanced analysis."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class DataProcessor:
    """Process and transform experiment results for analysis."""
    
    @staticmethod
    def calculate_attack_effectiveness(
        df: pd.DataFrame,
        baseline_mode: str = 'standard'
    ) -> pd.DataFrame:
        """Calculate attack effectiveness relative to baseline."""
        # Group by relevant columns to get baseline metrics
        group_cols = ['dataset', 'classifier']
        baseline = df[df['mode'] == baseline_mode].groupby(group_cols)['accuracy'].mean()
        
        # Calculate relative impact
        df = df.copy()
        df['baseline_accuracy'] = df.apply(
            lambda x: baseline[tuple(x[col] for col in group_cols)],
            axis=1
        )
        df['relative_impact'] = (df['baseline_accuracy'] - df['accuracy']) / df['baseline_accuracy']
        
        return df
    
    @staticmethod
    def aggregate_metrics(
        df: pd.DataFrame,
        group_by: List[str],
        metrics: List[str]
    ) -> pd.DataFrame:
        """Aggregate metrics by specified grouping."""
        agg_dict = {
            metric: ['mean', 'std', 'min', 'max']
            for metric in metrics
        }
        
        return df.groupby(group_by).agg(agg_dict).round(4)
    
    @staticmethod
    def calculate_class_impact(
        df: pd.DataFrame,
        metric_prefix: str = 'f1'
    ) -> pd.DataFrame:
        """Calculate per-class impact of attacks."""
        # Get per-class metric columns
        class_cols = [col for col in df.columns if col.startswith(f'{metric_prefix}_class_')]
        
        # Calculate mean impact per class
        df_impact = df.copy()
        df_impact['mean_class_impact'] = df[class_cols].mean(axis=1)
        df_impact['std_class_impact'] = df[class_cols].std(axis=1)
        df_impact['max_class_impact'] = df[class_cols].max(axis=1)
        df_impact['min_class_impact'] = df[class_cols].min(axis=1)
        
        return df_impact
    
    @staticmethod
    def analyze_attack_patterns(
        df: pd.DataFrame,
        poison_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Analyze patterns in attack effectiveness."""
        results = {}
        
        # Analyze high-impact scenarios
        high_impact = df[df['num_poisoned'] / df['total_images'] >= poison_threshold]
        
        # Most vulnerable datasets
        dataset_impact = high_impact.groupby('dataset')['accuracy'].mean()
        results['vulnerable_datasets'] = dataset_impact.sort_values().to_dict()
        
        # Most effective attacks
        attack_impact = high_impact.groupby('modification_method')['accuracy'].mean()
        results['effective_attacks'] = attack_impact.sort_values().to_dict()
        
        # Classifier resilience
        classifier_impact = high_impact.groupby('classifier')['accuracy'].mean()
        results['classifier_resilience'] = classifier_impact.sort_values(ascending=False).to_dict()
        
        return results
    
    @staticmethod
    def prepare_comparison_data(
        df: pd.DataFrame,
        metrics: List[str],
        group_by: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """Prepare data for comparative analysis."""
        # Calculate summary statistics
        summary_stats = {}
        for metric in metrics:
            stats = df.groupby(group_by)[metric].agg(['mean', 'std', 'min', 'max'])
            summary_stats[metric] = stats.to_dict('index')
        
        # Prepare normalized comparison data
        comparison_data = df.copy()
        for metric in metrics:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            comparison_data[f'{metric}_normalized'] = (df[metric] - mean_val) / std_val
        
        return comparison_data, summary_stats 