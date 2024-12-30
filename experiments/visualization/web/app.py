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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

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
    'GTSRB': {0: 0, 1: 392, 3: 1176, 5: 1960, 7: 2744, 10: 3920, 20: 7841},
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
    tolerance = total_images * 0.001  # 0.1% of total images
    
    # Find the closest standard rate within tolerance
    for rate, expected_count in mapping.items():
        if abs(num_poisoned - expected_count) <= tolerance:
            return rate
            
    # If no match found within tolerance, calculate actual rate
    return round((num_poisoned / total_images) * 100)

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

def prepare_data(df: pd.DataFrame, plot_type: str, x: str, y: str, color: str = None, secondary_group: str = None):
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
    
    # Add data quality checks
    total_combinations = len(df['poison_rate'].unique()) * len(df['mode'].unique())
    if secondary_group:
        total_combinations *= len(df[secondary_group].unique())
    actual_combinations = len(df.groupby(['poison_rate', 'mode', secondary_group if secondary_group else 'dataset']))
    
    if actual_combinations < total_combinations:
        st.warning(f"⚠️ Some data combinations are missing. This might cause unexpected patterns in the visualization.")
    
    # Check for potential outliers or anomalies
    metric_mean = df[y].mean()
    metric_std = df[y].std()
    outliers = df[abs(df[y] - metric_mean) > 3 * metric_std]
    if not outliers.empty:
        st.warning(f"⚠️ Detected potential outliers in {y}. This might cause sudden spikes or drops.")
    
    # For combined plots, we need to handle each dataset separately first
    datasets = df['dataset'].unique()
    processed_dfs = []
    
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        
        # Determine grouping based on plot type and color
        group_cols = ['poison_rate']
        if color:
            group_cols.append(color)
        if secondary_group and secondary_group in dataset_df.columns:
            group_cols.append(secondary_group)
        
        # Group and aggregate with additional statistics
        agg_dict = {
            y: ['mean', 'std', 'count']
        }
        
        df_grouped = dataset_df.groupby(group_cols).agg(agg_dict).reset_index()
        df_grouped.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df_grouped.columns]
        
        # Add confidence information
        df_grouped[f'{y}_ci'] = 1.96 * df_grouped[f'{y}_std'] / np.sqrt(df_grouped[f'{y}_count'])
        
        # Add dataset column back
        df_grouped['dataset'] = dataset
        
        processed_dfs.append(df_grouped)
    
    # Combine all processed dataframes
    df_grouped = pd.concat(processed_dfs, ignore_index=True)
    
    # Flag low sample sizes
    low_sample_mask = df_grouped[f'{y}_count'] < df_grouped[f'{y}_count'].median() / 2
    if low_sample_mask.any():
        st.warning(f"⚠️ Some points have significantly fewer samples, which might affect reliability.")
    
    # Sort to ensure proper line connection
    sort_cols = ['dataset', 'poison_rate']
    if secondary_group and secondary_group in df_grouped.columns:
        sort_cols.insert(0, secondary_group)
    df_grouped = df_grouped.sort_values(sort_cols)
    
    # Rename mean column back to original metric name for plotting
    df_grouped[y] = df_grouped[f'{y}_mean']
    
    return df_grouped, x, y

def create_plot(df: pd.DataFrame, plot_type: str, x: str, y: str, color: str = None, color_scheme: str = "default", secondary_group: str = "classifier"):
    """Create a plot based on the specified type and parameters."""
    try:
        if df.empty:
            st.warning("No data available for plotting")
            return None
        
        # Prepare data for plotting
        df, x, y = prepare_data(df, plot_type, x, y, color, secondary_group)
        
        # Define consistent color scheme for modes
        mode_colors = {'standard': 'rgb(31, 119, 180)', 'dynadetect': 'rgb(255, 127, 14)'}
        
        # Get unique poison rates and create tick values
        unique_rates = sorted(df['poison_rate'].unique())
        tick_values = unique_rates
        tick_text = [f"{rate}%" if rate > 0 else "Clean" for rate in unique_rates]
        
        # Create base plot
        if plot_type == "Line":
            fig = px.line(
                df, x=x, y=y,
                color='mode',
                line_dash='dataset',
                facet_col=secondary_group,
                facet_col_wrap=min(3, df[secondary_group].nunique()),
                height=700,
                category_orders={"mode": ["standard", "dynadetect"]},
                color_discrete_map=mode_colors
            )
            fig.update_traces(mode='lines+markers')

        elif plot_type == "Bar":
            fig = px.bar(
                df, x=x, y=y,
                color='mode',
                pattern_shape='dataset',
                barmode='group',
                facet_col=secondary_group,
                facet_col_wrap=min(3, df[secondary_group].nunique()),
                height=700,
                category_orders={"mode": ["standard", "dynadetect"]},
                color_discrete_map=mode_colors
            )

        else:
            fig = px.scatter(
                df, x=x, y=y,
                color='mode',
                symbol='dataset',
                facet_col=secondary_group,
                facet_col_wrap=min(3, df[secondary_group].nunique()),
                height=700,
                category_orders={"mode": ["standard", "dynadetect"]},
                color_discrete_map=mode_colors
            )

        # Update all subplots' x-axes
        for i in range(1, len(fig.data) + 1):
            xaxis_key = f'xaxis{i}' if i > 1 else 'xaxis'
            fig.update_layout({
                xaxis_key: {
                    'type': 'category',
                    'categoryorder': 'array',
                    'categoryarray': tick_values,
                    'ticktext': tick_text,
                    'tickvals': tick_values,
                    'title': {'text': "Poison Rate"},
                    'showticklabels': True
                }
            })
        
        # Update axis ranges based on data and metric type
        if y in ['accuracy', 'precision', 'recall', 'f1']:
            fig.update_layout(yaxis_range=[0, 1])
        
        # Ensure x-axis labels are visible on all subplots
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=50),  # Increase bottom margin
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            autosize=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None

def get_plot_download_formats():
    """Return available plot download formats with icons."""
    return {
        "png": {"icon": "📸", "mime": "image/png"},
        "svg": {"icon": "🎨", "mime": "image/svg+xml"},
        "pdf": {"icon": "📄", "mime": "application/pdf"},
        "html": {"icon": "🌐", "mime": "text/html"}
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

def calculate_improvement(df: pd.DataFrame, metric: str):
    """Calculate the relative improvement of dynadetect over standard mode."""
    df = df.copy()
    
    # Convert num_poisoned to poison_rate if needed
    if 'poison_rate' not in df.columns:
        df['poison_rate'] = df.apply(
            lambda row: calculate_poison_rate(row['num_poisoned'], row['total_images'], row['dataset']),
            axis=1
        )
    
    # First aggregate the metric values
    agg_df = df.groupby(['dataset', 'poison_rate', 'classifier', 'mode'])[metric].mean().reset_index()
    
    # Then pivot the aggregated data
    pivot_df = agg_df.pivot(
        index=['dataset', 'poison_rate', 'classifier'],
        columns='mode',
        values=metric
    ).reset_index()
    
    # Calculate absolute and relative improvement
    pivot_df['absolute_improvement'] = pivot_df['dynadetect'] - pivot_df['standard']
    # Handle division by zero in relative improvement calculation
    pivot_df['relative_improvement'] = (pivot_df['dynadetect'] - pivot_df['standard']).where(
        pivot_df['standard'] != 0,
        0
    ) / pivot_df['standard'].where(
        pivot_df['standard'] != 0,
        1
    ) * 100
    
    # Sort by poison_rate for proper plotting
    pivot_df = pivot_df.sort_values(['dataset', 'poison_rate'])
    
    return pivot_df

def create_improvement_plot(df: pd.DataFrame, metric: str, plot_type: str = "Bar"):
    """Create a plot showing the improvement of dynadetect over standard mode."""
    improvement_df = calculate_improvement(df, metric)
    
    # Get unique poison rates and create tick values
    unique_rates = sorted(improvement_df['poison_rate'].unique())
    tick_values = unique_rates
    tick_text = [f"{rate}%" if rate > 0 else "Clean" for rate in unique_rates]
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if plot_type == "Bar":
        # Add absolute improvement bars
        fig.add_trace(
            go.Bar(
                x=improvement_df['poison_rate'],
                y=improvement_df['absolute_improvement'],
                name='Absolute Improvement',
                marker_color='lightblue'
            ),
            secondary_y=False
        )
        
        # Add relative improvement line
        fig.add_trace(
            go.Scatter(
                x=improvement_df['poison_rate'],
                y=improvement_df['relative_improvement'],
                name='Relative Improvement (%)',
                marker_color='darkred',
                mode='lines+markers'
            ),
            secondary_y=True
        )
    
    # Update layout
    fig.update_layout(
        title=f"{metric.title()} Improvement: DynaDetect vs Standard",
        xaxis_title="Poison Rate",
        xaxis=dict(
            type='category',
            categoryorder='array',
            categoryarray=tick_values,
            ticktext=tick_text,
            tickvals=tick_values
        ),
        yaxis_title=f"Absolute {metric.title()} Improvement",
        yaxis2_title="Relative Improvement (%)",
        showlegend=True,
        height=700,
        margin=dict(l=20, r=20, t=40, b=20),
        autosize=True
    )
    
    return fig

def generate_insights(improvement_df: pd.DataFrame, metric: str) -> Dict[str, str]:
    """Generate insights from the improvement data."""
    insights = {}
    
    # Overall effectiveness
    avg_abs_improvement = improvement_df['absolute_improvement'].mean()
    avg_rel_improvement = improvement_df['relative_improvement'].mean()
    
    # Best and worst cases
    best_case = improvement_df.loc[improvement_df['absolute_improvement'].idxmax()]
    worst_case = improvement_df.loc[improvement_df['absolute_improvement'].idxmin()]
    
    # Poison rate analysis
    high_poison_rates = improvement_df[improvement_df['poison_rate'] >= 10]
    low_poison_rates = improvement_df[improvement_df['poison_rate'] < 10]
    high_rate_avg = high_poison_rates['absolute_improvement'].mean()
    low_rate_avg = low_poison_rates['absolute_improvement'].mean()
    
    # Generate summary
    summary = f"""
    ### Key Findings:
    - **Overall Performance**: DynaDetect {' improves ' if avg_abs_improvement > 0 else ' decreases '} {metric} by {abs(avg_abs_improvement):.3f} ({abs(avg_rel_improvement):.1f}% relative change) on average.
    
    - **Best Case**: Maximum improvement of {best_case['absolute_improvement']:.3f} ({best_case['relative_improvement']:.1f}%) at {best_case['poison_rate']}% poison rate with {best_case['classifier']}.
    
    - **Challenging Case**: Minimum improvement of {worst_case['absolute_improvement']:.3f} ({worst_case['relative_improvement']:.1f}%) at {worst_case['poison_rate']}% poison rate with {worst_case['classifier']}.
    
    - **Poison Rate Impact**: DynaDetect performs {'better' if high_rate_avg > low_rate_avg else 'worse'} at higher poison rates 
    (≥10%: {high_rate_avg:.3f} vs <10%: {low_rate_avg:.3f} average improvement).
    
    ### Areas for Improvement:
    {generate_improvement_recommendations(improvement_df, metric)}
    """
    
    return summary

def generate_improvement_recommendations(improvement_df: pd.DataFrame, metric: str) -> str:
    """Generate specific improvement recommendations based on the data."""
    recommendations = []
    
    # Check for negative improvements
    negative_cases = improvement_df[improvement_df['absolute_improvement'] < 0]
    if not negative_cases.empty:
        neg_classifiers = negative_cases['classifier'].unique()
        neg_rates = negative_cases['poison_rate'].unique()
        recommendations.append(
            f"- Focus on improving performance with {', '.join(neg_classifiers)} classifier(s) "
            f"at {', '.join(map(str, neg_rates))}% poison rates where DynaDetect underperforms standard mode."
        )
    
    # Check for high variance cases
    grouped_std = improvement_df.groupby('poison_rate')['absolute_improvement'].std()
    high_var_rates = grouped_std[grouped_std > grouped_std.mean()].index
    if len(high_var_rates) > 0:
        recommendations.append(
            f"- Work on consistency at {', '.join(map(str, high_var_rates))}% poison rates "
            f"where performance varies significantly."
        )
    
    # Check for specific classifier patterns
    classifier_avg = improvement_df.groupby('classifier')['absolute_improvement'].mean()
    worst_classifier = classifier_avg.idxmin()
    recommendations.append(
        f"- Investigate and optimize DynaDetect's interaction with {worst_classifier} "
        f"which shows the lowest average improvement."
    )
    
    if not recommendations:
        recommendations.append("- Continue monitoring performance across different scenarios to maintain effectiveness.")
    
    return "\n".join(recommendations)

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
            # Always use mode as the primary color differentiator
            color_by = 'mode'
            
            # Secondary grouping options
            secondary_group_options = ['classifier']  # Start with classifier as it's most important after mode
            
            # Add other categorical columns that make sense
            for col in sorted(list(categorical_cols)):
                if col not in ['mode', 'classifier'] and col != group_by and df[col].nunique() <= 10:
                    secondary_group_options.append(col)
            
            secondary_group = st.selectbox(
                "Secondary Grouping",
                secondary_group_options,
                index=0,  # Classifier is default
                key='secondary_group'
            )
        
        # Filter data
        filtered_df = df.copy()
        
        # Apply dataset filter
        filtered_df = filtered_df[filtered_df['dataset'].isin(selected_datasets)]
        
        # Apply classifier filter if selections are made
        if selected_classifiers:
            filtered_df = filtered_df[filtered_df['classifier'].isin(selected_classifiers)]
        
        if not filtered_df.empty:
            # Add view type selection
            view_type = st.radio(
                "View Type",
                ["Standard View", "Improvement Analysis"],
                key='view_type'
            )
            
            if view_type == "Standard View":
                if len(selected_datasets) > 1 and display_mode == "Combined Plot":
                    # Create single plot with all datasets
                    fig = create_plot(
                        filtered_df,
                        plot_type,
                        x=group_by,
                        y=metric,
                        color=color_by,
                        color_scheme=color_schemes[selected_color_scheme],
                        secondary_group=secondary_group
                    )
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                        
                        # Add single-click icon downloads
                        st.write("Download:")
                        download_formats = get_plot_download_formats()
                        
                        # Create columns with empty space on sides for centering
                        left_spacer, center_content, right_spacer = st.columns([2, 1, 2])
                        with center_content:
                            cols = st.columns(len(download_formats))
                            for col, (format_type, format_info) in zip(cols, download_formats.items()):
                                with col:
                                    plot_data = download_plot(fig, format_type)
                                    st.download_button(
                                        label=format_info["icon"],
                                        data=plot_data,
                                        file_name=f"plot_{dataset}_{metric}.{format_type}",
                                        mime=format_info["mime"],
                                        key=f'download_{dataset}_{format_type}'
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
                            color_scheme=color_schemes[selected_color_scheme],
                            secondary_group=secondary_group
                        )
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                            
                            # Add single-click icon downloads
                            st.write("Download:")
                            download_formats = get_plot_download_formats()
                            
                            # Create columns with empty space on sides for centering
                            left_spacer, center_content, right_spacer = st.columns([2, 1, 2])
                            with center_content:
                                cols = st.columns(len(download_formats))
                                for col, (format_type, format_info) in zip(cols, download_formats.items()):
                                    with col:
                                        plot_data = download_plot(fig, format_type)
                                        st.download_button(
                                            label=format_info["icon"],
                                            data=plot_data,
                                            file_name=f"plot_{dataset}_{metric}.{format_type}",
                                            mime=format_info["mime"],
                                            key=f'download_{dataset}_{format_type}'
                                        )
                        
                        # Add separator between plots if not the last dataset
                        if dataset != selected_datasets[-1]:
                            st.divider()
            else:
                # Show improvement analysis
                st.subheader("DynaDetect Improvement Analysis")
                
                # Create improvement plot for each dataset
                for dataset in selected_datasets:
                    dataset_df = filtered_df[filtered_df['dataset'] == dataset]
                    
                    if len(selected_datasets) > 1:
                        st.write(f"### Dataset: {dataset}")
                    
                    fig = create_improvement_plot(dataset_df, metric)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    
                    # Add insights
                    improvement_stats = calculate_improvement(dataset_df, metric)
                    insights = generate_insights(improvement_stats, metric)
                    st.markdown(insights)
                    
                    # Add summary statistics in expander
                    with st.expander(f"View Detailed {dataset} Statistics"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Average Absolute Improvement")
                            st.dataframe(
                                improvement_stats.groupby('poison_rate')['absolute_improvement']
                                .agg(['mean', 'std'])
                                .round(4)
                            )
                        with col2:
                            st.write("Average Relative Improvement (%)")
                            st.dataframe(
                                improvement_stats.groupby('poison_rate')['relative_improvement']
                                .agg(['mean', 'std'])
                                .round(2)
                            )
                    
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