"""Specialized plots for attack analysis."""
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

class AttackAnalysis:
    """Generate specialized plots for attack analysis."""
    
    def __init__(self):
        """Initialize with default styling."""
        self.default_layout = {
            'template': 'plotly_white',
            'font': dict(family='Arial', size=12),
            'showlegend': True
        }
        
    def attack_success_rate(
        self,
        df: pd.DataFrame,
        by_dataset: bool = False,
        by_classifier: bool = False
    ) -> go.Figure:
        """Plot attack success rate across different configurations."""
        # Calculate success rate (1 - accuracy)
        df = df.copy()
        df['success_rate'] = 1 - df['accuracy']
        
        if by_dataset and by_classifier:
            fig = px.box(
                df,
                x='dataset',
                y='success_rate',
                color='classifier',
                facet_col='modification_method',
                title='Attack Success Rate by Dataset and Classifier'
            )
        elif by_dataset:
            fig = px.box(
                df,
                x='modification_method',
                y='success_rate',
                color='dataset',
                title='Attack Success Rate by Dataset'
            )
        elif by_classifier:
            fig = px.box(
                df,
                x='modification_method',
                y='success_rate',
                color='classifier',
                title='Attack Success Rate by Classifier'
            )
        else:
            fig = px.box(
                df,
                x='modification_method',
                y='success_rate',
                title='Attack Success Rate'
            )
            
        fig.update_layout(**self.default_layout)
        return fig
        
    def attack_comparison_heatmap(
        self,
        df: pd.DataFrame,
        metric: str = 'accuracy',
        group_by: List[str] = ['modification_method', 'dataset']
    ) -> go.Figure:
        """Create a heatmap comparing attack effectiveness."""
        pivot_df = df.pivot_table(
            values=metric,
            index=group_by[0],
            columns=group_by[1],
            aggfunc='mean'
        )
        
        fig = px.imshow(
            pivot_df,
            title=f'{metric.title()} Comparison',
            color_continuous_scale='RdYlBu_r'
        )
        
        fig.update_layout(**self.default_layout)
        return fig
        
    def per_class_impact(
        self,
        df: pd.DataFrame,
        attack_method: str,
        metric_prefix: str = 'f1'
    ) -> go.Figure:
        """Analyze per-class impact of attacks."""
        # Filter for specific attack method
        df_attack = df[df['modification_method'] == attack_method]
        
        # Get per-class metrics
        class_cols = [col for col in df.columns if col.startswith(f'{metric_prefix}_class_')]
        
        # Melt the dataframe for plotting
        df_melted = pd.melt(
            df_attack,
            id_vars=['dataset', 'classifier'],
            value_vars=class_cols,
            var_name='class',
            value_name=metric_prefix
        )
        
        fig = px.box(
            df_melted,
            x='class',
            y=metric_prefix,
            color='dataset',
            facet_col='classifier',
            title=f'Per-Class {metric_prefix.upper()} Impact of {attack_method}'
        )
        
        fig.update_layout(**self.default_layout)
        return fig
        
    def attack_progression(
        self,
        df: pd.DataFrame,
        metric: str = 'accuracy',
        by_dataset: bool = False
    ) -> go.Figure:
        """Plot attack progression over poison rates."""
        if by_dataset:
            fig = px.line(
                df,
                x='num_poisoned',
                y=metric,
                color='dataset',
                line_dash='modification_method',
                title=f'{metric.title()} vs Poison Rate by Dataset'
            )
        else:
            fig = px.line(
                df,
                x='num_poisoned',
                y=metric,
                color='modification_method',
                line_dash='classifier',
                title=f'{metric.title()} vs Poison Rate'
            )
            
        fig.update_layout(
            **self.default_layout,
            xaxis_title='Number of Poisoned Samples',
            yaxis_title=metric.title()
        )
        return fig 