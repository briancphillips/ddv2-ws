"""Streamlit web application for interactive visualization."""
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any
import sys
import os
import gc

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_loader import ResultsLoader

# Configure Streamlit
st.set_page_config(
    page_title="DynaDetect Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=30, show_spinner=False)
def load_data():
    """Load and cache the results data."""
    try:
        loader = ResultsLoader()
        df = loader.load_latest_results()
        
        # Ensure categorical columns are strings
        categorical_cols = ['dataset', 'classifier', 'mode', 'modification_method', 'flip_type']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Apply data validation rules
        # Rule 1: If modification_method is not label_flipping, set flip_type to None
        df.loc[df['modification_method'] != 'label_flipping', 'flip_type'] = None
        
        # Remove rows where rules are violated
        df = df.dropna(subset=['modification_method'])  # Ensure core columns exist
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def validate_metric_selection(df: pd.DataFrame, metric: str, modification_method: str = None) -> bool:
    """Validate if a metric is applicable for the given modification method."""
    # Add validation rules for metrics
    if modification_method and modification_method != 'label_flipping':
        invalid_metrics = ['flip_type']  # Add other metrics that are only valid for label_flipping
        if metric in invalid_metrics:
            return False
    return True

def create_plot(df: pd.DataFrame, plot_type: str, x: str, y: str, color: str = None):
    """Create a plot based on the specified type and parameters."""
    try:
        if df.empty:
            st.warning("No data available for plotting")
            return None
            
        df = df.copy()
        
        # Handle poison rates if needed
        if x == 'num_poisoned' or y == 'num_poisoned':
            # Filter out clean baselines (num_poisoned = 0)
            df = df[df['num_poisoned'] > 0]
            
            # Define dataset-specific mappings for standard rates
            rate_mappings = {
                'GTSRB': {
                    1: int(39209 * 0.01),    # 1%
                    3: int(39209 * 0.03),    # 3%
                    5: int(39209 * 0.05),    # 5%
                    7: int(39209 * 0.07),    # 7%
                    10: int(39209 * 0.10),   # 10%
                    20: int(39209 * 0.20)    # 20%
                },
                'CIFAR100': {
                    1: int(50000 * 0.01),    # 1%
                    3: int(50000 * 0.03),    # 3%
                    5: int(50000 * 0.05),    # 5%
                    7: int(50000 * 0.07),    # 7%
                    10: int(50000 * 0.10),   # 10%
                    20: int(50000 * 0.20)    # 20%
                },
                'ImageNette': {
                    1: int(9469 * 0.01),    # 1%
                    3: int(9469 * 0.03),    # 3%
                    5: int(9469 * 0.05),    # 5%
                    7: int(9469 * 0.07),    # 7%
                    10: int(9469 * 0.10),   # 10%
                    20: int(9469 * 0.20)    # 20%
                }
            }
            
            # Create poison rate column based on dataset
            df['poison_rate'] = df.apply(
                lambda row: next(
                    (rate for rate, num in rate_mappings[row['dataset']].items() 
                     if abs(row['num_poisoned'] - num) < 10),
                    float('nan')  # Return NaN if no mapping found
                ),
                axis=1
            )
            
            # Filter to only include mapped rates for each dataset
            valid_nums = [num for mapping in rate_mappings.values() for num in mapping.values()]
            df = df[df.apply(lambda row: any(abs(row['num_poisoned'] - rate_mappings[row['dataset']][rate]) < 10 
                                           for rate in rate_mappings[row['dataset']]), axis=1)]
            
            if x == 'num_poisoned':
                grouped = df.groupby(['poison_rate', color], observed=True)[y].mean().reset_index()
                x = 'poison_rate'
            if y == 'num_poisoned':
                grouped = df.groupby(['poison_rate', color], observed=True)[x].mean().reset_index()
                y = 'poison_rate'
            df = grouped
        
        if plot_type == "Line":
            fig = px.line(df, x=x, y=y, color=color)
        elif plot_type == "Bar":
            fig = px.bar(df, x=x, y=y, color=color, barmode='group')
        elif plot_type == "Box":
            fig = px.box(df, x=x, y=y, color=color)
        else:  # Scatter
            fig = px.scatter(df, x=x, y=y, color=color)
        
        # Update axis labels and ticks
        if x == 'poison_rate':
            fig.update_layout(
                xaxis_title="Poison Rate (%)",
                xaxis=dict(
                    type='category',  # Force categorical axis
                    categoryorder='array',
                    categoryarray=[1, 3, 5, 7, 10, 20]  # Standard rates
                )
            )
        if y == 'poison_rate':
            fig.update_layout(
                yaxis_title="Poison Rate (%)",
                yaxis=dict(
                    type='category',  # Force categorical axis
                    categoryorder='array',
                    categoryarray=[1, 3, 5, 7, 10, 20]  # Standard rates
                )
            )
        
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        return fig
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None

def main():
    """Main application function."""
    try:
        st.title("DynaDetect Analysis")
        
        # Load data
        df = load_data()
        if df.empty:
            st.error("No data available. Please check the results directory.")
            return
        
        # Sidebar controls
        with st.sidebar:
            st.header("Analysis Controls")
            
            # Add refresh button at the top
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
            
            st.divider()  # Add visual separator
            
            # Dataset and Classifier Selection
            datasets = sorted(df['dataset'].unique())
            selected_datasets = st.multiselect(
                "Datasets",
                datasets,
                default=[],  # Empty default to show all datasets
                key='datasets'
            )
            
            classifiers = sorted(df['classifier'].unique())
            selected_classifiers = st.multiselect(
                "Classifiers",
                classifiers,
                default=['LogisticRegression', 'RandomForest'],
                key='classifiers'
            )
            
            # Plot Configuration
            plot_type = st.selectbox(
                "Plot Type",
                ["Line", "Bar", "Box", "Scatter"],
                index=0,  # Line plot
                key='plot_type'
            )
            
            # Get numeric columns for metrics
            numeric_cols = sorted([col for col in df.select_dtypes(include=['float64', 'int64']).columns
                                if not col.startswith('Unnamed')])
            
            # Get current modification method for metric validation
            current_mod_method = None
            if 'modification_method' in df.columns:
                mod_methods = df['modification_method'].unique()
                if len(mod_methods) == 1:
                    current_mod_method = mod_methods[0]
            
            # Metric Selection - filter based on modification method
            valid_metrics = [col for col in numeric_cols 
                           if validate_metric_selection(df, col, current_mod_method)]
            
            # Set default metric to accuracy if available
            default_metric_index = valid_metrics.index('accuracy') if 'accuracy' in valid_metrics else 0
            
            metric = st.selectbox(
                "Metric",
                valid_metrics,
                index=default_metric_index,
                key='metric'
            )
            
            # Group By Selection
            categorical_cols = df.select_dtypes(include=['object']).columns
            group_options = ["Number of Poisoned Samples"] + sorted(list(categorical_cols))
            
            group_by = st.selectbox(
                "Group By",
                group_options,
                index=0,  # Number of Poisoned Samples
                key='group_by'
            )
            
            # Convert display name back to column name
            if group_by == "Number of Poisoned Samples":
                group_by = "num_poisoned"
            
            # Color By Selection - only show categorical columns
            color_by = st.selectbox(
                "Color By",
                sorted(list(categorical_cols)),
                index=list(sorted(categorical_cols)).index('mode') if 'mode' in categorical_cols else 0,
                key='color_by'
            )
        
        # Filter data
        filtered_df = df.copy()
        
        # Apply filters only if selections are made
        if selected_datasets:
            filtered_df = filtered_df[filtered_df['dataset'].isin(selected_datasets)]
        if selected_classifiers:
            filtered_df = filtered_df[filtered_df['classifier'].isin(selected_classifiers)]
        
        if not filtered_df.empty:
            # Create plot
            fig = create_plot(
                filtered_df,
                plot_type,
                x=group_by,
                y=metric,
                color=color_by
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Summary statistics
            with st.expander("View Statistics"):
                col1, col2 = st.columns(2)
                with col1:
                    group_stats = filtered_df.groupby(group_by)[metric].agg(['mean', 'std']).round(4)
                    st.dataframe(group_stats)
                with col2:
                    color_stats = filtered_df.groupby(color_by)[metric].agg(['mean', 'std']).round(4)
                    st.dataframe(color_stats)
            
            # Raw data
            with st.expander("View Data"):
                st.dataframe(
                    filtered_df[[group_by, color_by, metric]],
                    use_container_width=True
                )
        else:
            st.info("No data available for the selected filters.")
        
        # Clean up
        gc.collect()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 