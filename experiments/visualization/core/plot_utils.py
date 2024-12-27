"""Common plotting utilities for visualization."""
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional
import pandas as pd

class PlotGenerator:
    """Generate common plot types with consistent styling."""
    
    def __init__(self):
        """Initialize with default styling."""
        self.default_layout = {
            'template': 'plotly_white',
            'font': dict(family='Arial', size=12),
            'showlegend': True,
            'legend': dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        }
        
    def line_plot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create a line plot."""
        fig = px.line(
            df, x=x, y=y, 
            color=color,
            title=title,
            template=self.default_layout['template']
        )
        
        fig.update_layout(
            **self.default_layout,
            **kwargs
        )
        return fig
        
    def scatter_plot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        size: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create a scatter plot."""
        fig = px.scatter(
            df, x=x, y=y,
            color=color,
            size=size,
            title=title,
            template=self.default_layout['template']
        )
        
        fig.update_layout(
            **self.default_layout,
            **kwargs
        )
        return fig
        
    def bar_plot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create a bar plot."""
        fig = px.bar(
            df, x=x, y=y,
            color=color,
            title=title,
            template=self.default_layout['template']
        )
        
        fig.update_layout(
            **self.default_layout,
            **kwargs
        )
        return fig
        
    def box_plot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create a box plot."""
        fig = px.box(
            df, x=x, y=y,
            color=color,
            title=title,
            template=self.default_layout['template']
        )
        
        fig.update_layout(
            **self.default_layout,
            **kwargs
        )
        return fig
        
    def heatmap(
        self,
        df: pd.DataFrame,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create a heatmap."""
        fig = px.imshow(
            df,
            title=title,
            template=self.default_layout['template']
        )
        
        fig.update_layout(
            **self.default_layout,
            **kwargs
        )
        return fig 