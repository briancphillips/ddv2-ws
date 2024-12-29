"""Streamlit web application for interactive visualization."""
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any
import sys
import os
import gc
import io
import plotly.io as pio

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

# Define dataset-specific mappings for standard rates
rate_mappings = {
    'GTSRB': {0: 0, 1: 197, 3: 592, 5: 987, 7: 1382, 10: 1975, 20: 3951},
    'CIFAR100': {0: 0, 1: 500, 3: 1500, 5: 2500, 7: 3500, 10: 5000, 20: 10000},
    'ImageNette': {0: 0, 1: 94, 3: 284, 5: 473, 7: 662, 10: 946, 20: 1893}
}

def calculate_poison_rate(num_poisoned: int, total_images: int, dataset: str) -> float:
    """Map the number of poisoned samples to the standard poison rate for each dataset."""
    if num_poisoned == 0:
        return 0
        
    # Get the mapping for this dataset
    mapping = rate_mappings.get(dataset, {})
    if not mapping:
        return 0
        
    # Calculate tolerance based on dataset size
    tolerance = get_tolerance(max(mapping.values()))
    
    # Find the closest standard rate within tolerance
    for rate, expected_count in mapping.items():
        if abs(num_poisoned - expected_count) <= tolerance:
            return rate
            
    # If no match found within tolerance, find the closest rate
    closest_rate = min(mapping.items(), key=lambda x: abs(x[1] - num_poisoned))[0]
    return closest_rate

def get_tolerance(dataset_size: int) -> int:
    """Calculate tolerance for poison rate mapping based on dataset size."""
    return max(5, int(dataset_size * 0.001))  # 0.1% of dataset size or minimum 5

@st.cache_data(ttl=30, show_spinner=False)
def load_data():
    """Load and cache the results data."""
    try:
        loader = ResultsLoader()
        df = loader.load_latest_results()
        
        # Ensure categorical columns are strings
        categorical_cols = ['dataset', 'classifier', 'modification_method', 'flip_type']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Handle mode column specially
        if 'mode' in df.columns:
            df['mode'] = df['mode'].map(lambda x: 'dynadetect' if 'dyna' in str(x).lower() else 'standard')
        
        # Apply data validation rules
        df.loc[df['modification_method'] != 'label_flipping', 'flip_type'] = None
        df = df.dropna(subset=['modification_method'])
        
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

def prepare_data(df: pd.DataFrame, plot_type: str, x: str, y: str, color: str = None):
    """Prepare data for plotting by handling grouping and aggregation."""
    df = df.copy()
    
    # Handle poison rates if needed
    if x == 'num_poisoned' or y == 'num_poisoned':
        # Create poison rate column using dataset-specific mappings
        df['poison_rate'] = df.apply(
            lambda row: calculate_poison_rate(row['num_poisoned'], row['total_images'], row['dataset']),
            axis=1
        )
        
        # Replace num_poisoned with poison_rate in x or y
        if x == 'num_poisoned':
            x = 'poison_rate'
        if y == 'num_poisoned':
            y = 'poison_rate'
    
    # Determine grouping based on plot type and color
    group_cols = ['poison_rate']
    if color:
        group_cols.append(color)
    if 'dataset' in df.columns:  # Always include dataset in grouping
        group_cols.append('dataset')
    
    # Group and aggregate
    df = df.groupby(group_cols)[y].mean().reset_index()
    
    # Sort to ensure proper line connection
    df = df.sort_values(['dataset', 'poison_rate'] if 'dataset' in df.columns else 'poison_rate')
    
    return df, x, y

def create_plot(df: pd.DataFrame, plot_type: str, x: str, y: str, color: str = None, color_scheme: str = "default"):
    """Create a plot based on the specified type and parameters."""
    try:
        if df.empty:
            st.warning("No data available for plotting")
            return None
        
        # Prepare data for plotting
        df, x, y = prepare_data(df, plot_type, x, y, color)
        
        # Set color scheme
        color_sequence = None
        template = "plotly"  # Default template
        
        if color_scheme != "default":
            if color_scheme == "viridis":
                color_sequence = px.colors.sequential.Viridis
            elif color_scheme == "plasma":
                color_sequence = px.colors.sequential.Plasma
            elif color_scheme == "inferno":
                color_sequence = px.colors.sequential.Inferno
            elif color_scheme == "magma":
                color_sequence = px.colors.sequential.Magma
            elif color_scheme == "colorblind":
                color_sequence = px.colors.qualitative.Plotly
            elif color_scheme == "dark":
                color_sequence = px.colors.qualitative.Dark24
                template = "plotly_dark"
        
        # Create plot based on type with template
        plot_kwargs = {
            "color_discrete_sequence": color_sequence,
            "template": template,
            "category_orders": {"mode": ["standard", "dynadetect"]}  # Fix mode order
        } if color_sequence else {"template": template, "category_orders": {"mode": ["standard", "dynadetect"]}}
        
        if plot_type == "Line":
            fig = px.line(df, x=x, y=y, color=color, line_shape='linear', **plot_kwargs)
            fig.update_traces(mode='lines+markers')
        elif plot_type == "Bar":
            fig = px.bar(df, x=x, y=y, color=color, barmode='group', **plot_kwargs)
        elif plot_type == "Box":
            fig = px.box(df, x=x, y=y, color=color, **plot_kwargs)
        else:  # Scatter
            fig = px.scatter(df, x=x, y=y, color=color, **plot_kwargs)
        
        # Update axis ranges based on data and metric type
        if y in ['accuracy', 'precision', 'recall', 'f1']:
            y_range = [0, 1]
            fig.update_layout(yaxis_range=y_range)
        
        # Update axis labels and ticks with proper handling of clean baselines
        if x == 'poison_rate' or y == 'poison_rate':
            axis_config = dict(
                type='category',
                categoryorder='array',
                categoryarray=[0, 1, 3, 5, 7, 10, 20],
                ticktext=['Clean', '1%', '3%', '5%', '7%', '10%', '20%'],
                tickvals=[0, 1, 3, 5, 7, 10, 20]
            )
            
            if x == 'poison_rate':
                fig.update_layout(
                    xaxis_title="Poison Rate",
                    xaxis=axis_config
                )
            if y == 'poison_rate':
                fig.update_layout(
                    yaxis_title="Poison Rate",
                    yaxis=axis_config
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

def get_plot_download_formats():
    """Return available plot download formats."""
    return {
        "HTML": "html",
        "PNG (High DPI)": "png",
        "SVG": "svg",
        "PDF (High DPI)": "pdf",
        "JPEG (High DPI)": "jpg"
    }

def download_plot(fig, format_type: str):
    """Generate plot in specified format for download with high DPI."""
    if format_type == "html":
        buffer = io.StringIO()
        fig.write_html(buffer)
        return buffer.getvalue()
    else:
        # Set high DPI (300) and ensure plot style is preserved
        img_bytes = pio.to_image(
            fig,
            format=format_type,
            scale=3,  # Triple the resolution (300 DPI)
            width=1920,  # Full HD width
            height=1080  # Full HD height
        )
        return img_bytes

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
                default=[],  # Empty default
                key='datasets'
            )
            
            # If no datasets selected, use all datasets
            if not selected_datasets:
                selected_datasets = datasets
            
            # Add display mode selection when multiple datasets are selected
            display_mode = "Separate Plots"
            if len(selected_datasets) > 1:
                display_mode = st.radio(
                    "Multiple Dataset Display",
                    ["Separate Plots", "Combined Plot"],
                    key='display_mode'
                )
            
            # Classifier Selection
            classifiers = sorted(df['classifier'].unique())
            selected_classifiers = st.multiselect(
                "Classifiers",
                classifiers,
                default=[],  # Empty default
                key='classifiers'
            )
            
            # Determine available plot types based on context
            all_plot_types = ["Bar", "Line", "Box", "Scatter"]
            if len(selected_datasets) > 1 and display_mode == "Combined Plot":
                # For combined plots, only allow types that work well with multiple datasets
                valid_plot_types = ["Line", "Bar"]
            else:
                valid_plot_types = all_plot_types
            
            plot_type = st.selectbox(
                "Plot Type",
                valid_plot_types,
                index=0,  # Bar is first in the list now
                key='plot_type'
            )
            
            # Add color scheme selection
            color_schemes = {
                "Default": "default",
                "Viridis": "viridis",
                "Plasma": "plasma",
                "Inferno": "inferno",
                "Magma": "magma",
                "Colorblind Friendly": "colorblind",
                "Dark": "dark"
            }
            selected_color_scheme = st.selectbox(
                "Color Scheme",
                options=list(color_schemes.keys()),
                format_func=lambda x: x,
                key='color_scheme'
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
            
            # Metric Selection - filter based on modification method and plot type
            valid_metrics = [col for col in numeric_cols 
                           if validate_metric_selection(df, col, current_mod_method)]
            
            # Remove metrics that don't make sense for the current plot type
            if plot_type == "Box":
                # Remove metrics that are always single-valued per group
                valid_metrics = [m for m in valid_metrics if df.groupby('dataset')[m].nunique().max() > 1]
            
            # Set default metric to accuracy if available
            default_metric_index = valid_metrics.index('accuracy') if 'accuracy' in valid_metrics else 0
            
            metric = st.selectbox(
                "Metric",
                valid_metrics,
                index=default_metric_index,
                key='metric'
            )
            
            # Group By Selection - adapt based on plot type and data
            categorical_cols = df.select_dtypes(include=['object']).columns
            group_options = ["Number of Poisoned Samples"]
            
            # Only add categorical columns that make sense for the current plot type
            if plot_type in ["Bar", "Box"]:
                for col in sorted(list(categorical_cols)):
                    if df[col].nunique() <= 10:  # Only include categorical columns with reasonable number of values
                        group_options.append(col)
            
            group_by = st.selectbox(
                "Group By",
                group_options,
                index=0,  # Number of Poisoned Samples
                key='group_by'
            )
            
            # Convert display name back to column name
            if group_by == "Number of Poisoned Samples":
                group_by = "num_poisoned"
            
            # Color By Selection - adapt based on context
            color_options = ['classifier', 'mode']  # Set fixed important options first
            
            # Add any other categorical columns that make sense
            for col in sorted(list(categorical_cols)):
                if col not in color_options and col != group_by and df[col].nunique() <= 10:
                    color_options.append(col)
            
            color_by = st.selectbox(
                "Color By",
                color_options,
                index=color_options.index('mode'),  # Always default to mode
                key='color_by'
            )
        
        # Filter data
        filtered_df = df.copy()
        
        # Apply dataset filter
        filtered_df = filtered_df[filtered_df['dataset'].isin(selected_datasets)]
        
        # Apply classifier filter if selections are made
        if selected_classifiers:
            filtered_df = filtered_df[filtered_df['classifier'].isin(selected_classifiers)]
        
        if not filtered_df.empty:
            if len(selected_datasets) > 1 and display_mode == "Combined Plot":
                # Create single plot with all datasets
                fig = create_plot(
                    filtered_df,
                    plot_type,
                    x=group_by,
                    y=metric,
                    color=color_by,
                    color_scheme=color_schemes[selected_color_scheme]
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    
                    # Add download controls for combined plot
                    download_formats = get_plot_download_formats()
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        download_format = st.selectbox(
                            "Download Format",
                            options=list(download_formats.keys()),
                            key='download_format_combined'
                        )
                    with col2:
                        if st.button("Download Plot", key='download_button_combined'):
                            format_type = download_formats[download_format]
                            file_extension = format_type
                            plot_data = download_plot(fig, format_type)
                            
                            if format_type == "html":
                                st.download_button(
                                    label="Click to Download",
                                    data=plot_data,
                                    file_name=f"plot_combined.{file_extension}",
                                    mime="text/html",
                                    key='download_link_combined'
                                )
                            else:
                                st.download_button(
                                    label="Click to Download",
                                    data=plot_data,
                                    file_name=f"plot_combined.{file_extension}",
                                    mime=f"image/{format_type}",
                                    key='download_link_combined'
                                )
            else:
                # Create separate plots for each dataset
                for dataset in selected_datasets:
                    dataset_df = filtered_df[filtered_df['dataset'] == dataset]
                    
                    # Create plot title
                    if len(selected_datasets) > 1:
                        st.subheader(f"Dataset: {dataset}")
                    
                    # Create plot with color scheme
                    fig = create_plot(
                        dataset_df,
                        plot_type,
                        x=group_by,
                        y=metric,
                        color=color_by,
                        color_scheme=color_schemes[selected_color_scheme]
                    )
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                        
                        # Add download controls below plot
                        download_formats = get_plot_download_formats()
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            download_format = st.selectbox(
                                "Download Format",
                                options=list(download_formats.keys()),
                                key=f'download_format_{dataset}'  # Unique key for each dataset plot
                            )
                        with col2:
                            if st.button("Download Plot", key=f'download_button_{dataset}'):  # Unique key for each dataset plot
                                format_type = download_formats[download_format]
                                file_extension = format_type
                                plot_data = download_plot(fig, format_type)
                                
                                if format_type == "html":
                                    st.download_button(
                                        label="Click to Download",
                                        data=plot_data,
                                        file_name=f"plot_{dataset}_{metric}_{color_by}.{file_extension}",
                                        mime="text/html",
                                        key=f'download_link_{dataset}'
                                    )
                                else:
                                    st.download_button(
                                        label="Click to Download",
                                        data=plot_data,
                                        file_name=f"plot_{dataset}_{metric}_{color_by}.{file_extension}",
                                        mime=f"image/{format_type}",
                                        key=f'download_link_{dataset}'
                                    )
                    
                    # Add separator between plots if not the last dataset
                    if dataset != selected_datasets[-1]:
                        st.divider()
            
            # Summary statistics
            with st.expander("View Statistics"):
                if len(selected_datasets) > 1:
                    tabs = st.tabs(["Overall"] + list(selected_datasets))
                    
                    # Overall stats in first tab
                    with tabs[0]:
                        col1, col2 = st.columns(2)
                        with col1:
                            # Group by poison_rate and dataset
                            stats_df = prepare_data(filtered_df, plot_type, group_by, metric, color_by)[0]
                            group_stats = stats_df.groupby('poison_rate')[metric].agg(['mean', 'std']).round(4)
                            st.dataframe(group_stats)
                        with col2:
                            # For color stats, use dataset directly if it's the color
                            if color_by == 'dataset':
                                color_stats = filtered_df.groupby('dataset')[metric].agg(['mean', 'std']).round(4)
                            else:
                                # First group by both color_by and dataset, then average across datasets
                                grouped = filtered_df.groupby([color_by, 'dataset'])[metric].agg(['mean', 'std'])
                                color_stats = pd.DataFrame({
                                    'mean': grouped['mean'].groupby(level=0).mean(),
                                    'std': grouped['std'].groupby(level=0).mean()
                                }).round(4)
                            st.dataframe(color_stats)
                    
                    # Per-dataset stats in subsequent tabs
                    for i, dataset in enumerate(selected_datasets, 1):
                        with tabs[i]:
                            dataset_df = filtered_df[filtered_df['dataset'] == dataset]
                            col1, col2 = st.columns(2)
                            with col1:
                                stats_df = prepare_data(dataset_df, plot_type, group_by, metric, color_by)[0]
                                group_stats = stats_df.groupby('poison_rate')[metric].agg(['mean', 'std']).round(4)
                                st.dataframe(group_stats)
                            with col2:
                                color_stats = dataset_df.groupby(color_by)[metric].agg(['mean', 'std']).round(4)
                                st.dataframe(color_stats)
                else:
                    # Single dataset stats
                    col1, col2 = st.columns(2)
                    with col1:
                        stats_df = prepare_data(filtered_df, plot_type, group_by, metric, color_by)[0]
                        group_stats = stats_df.groupby('poison_rate')[metric].agg(['mean', 'std']).round(4)
                        st.dataframe(group_stats)
                    with col2:
                        color_stats = filtered_df.groupby(color_by)[metric].agg(['mean', 'std']).round(4)
                        st.dataframe(color_stats)
            
            # Raw data
            with st.expander("View Data"):
                if len(selected_datasets) > 1:
                    # Show data grouped by dataset
                    st.dataframe(
                        filtered_df[['dataset', group_by, color_by, metric]].sort_values(['dataset', group_by]),
                        use_container_width=True
                    )
                else:
                    # Original behavior for single dataset
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