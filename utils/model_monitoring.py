"""
Model Monitoring Utilities for CPI Analysis & Prediction Dashboard.

This module provides functions for tracking model performance over time
and visualizing performance metrics.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging

# Import visualization utilities for consistent styling
from utils.visualization import format_chart_for_dark_mode
from utils.theme import COLOR_SYSTEM, TYPOGRAPHY

# Configure logging
logger = logging.getLogger(__name__)

# Define constants
MONITORING_FILE = "model_monitoring_data.json"
DEFAULT_METRICS = ["mae", "rmse", "r2"]

class ModelMonitor:
    """Class for monitoring model performance over time."""
    
    def __init__(self, model_name: str):
        """
        Initialize the model monitor.
        
        Args:
            model_name (str): Name of the model being monitored
        """
        self.model_name = model_name
        self.monitoring_data = self._load_monitoring_data()
    
    def _load_monitoring_data(self) -> Dict[str, Any]:
        """
        Load existing monitoring data or create new structure if none exists.
        
        Returns:
            Dict[str, Any]: Dictionary containing monitoring data
        """
        try:
            if os.path.exists(MONITORING_FILE):
                with open(MONITORING_FILE, 'r') as f:
                    data = json.load(f)
                    if self.model_name not in data:
                        data[self.model_name] = {
                            "metrics": [],
                            "last_updated": None
                        }
                    return data
            else:
                # Create new monitoring data structure
                return {
                    self.model_name: {
                        "metrics": [],
                        "last_updated": None
                    }
                }
        except Exception as e:
            logger.error(f"Error loading monitoring data: {e}")
            # Return empty structure in case of error
            return {
                self.model_name: {
                    "metrics": [],
                    "last_updated": None
                }
            }
    
    def _save_monitoring_data(self) -> None:
        """Save monitoring data to file."""
        try:
            with open(MONITORING_FILE, 'w') as f:
                json.dump(self.monitoring_data, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], date: Optional[str] = None) -> None:
        """
        Log model performance metrics for a specific date.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metrics (e.g., {'mae': 2.5})
            date (Optional[str]): Date string in YYYY-MM-DD format, defaults to today
        """
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        metric_entry = {
            "date": date,
            "metrics": metrics
        }
        
        self.monitoring_data[self.model_name]["metrics"].append(metric_entry)
        self.monitoring_data[self.model_name]["last_updated"] = date
        self._save_monitoring_data()
        logger.info(f"Logged metrics for {self.model_name} on {date}")
    
    def get_metrics_history(self) -> pd.DataFrame:
        """
        Get the history of all logged metrics as a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with metrics history
        """
        metrics_data = self.monitoring_data[self.model_name]["metrics"]
        if not metrics_data:
            return pd.DataFrame()
        
        # Flatten metrics data for DataFrame
        rows = []
        for entry in metrics_data:
            date = entry["date"]
            for metric_name, value in entry["metrics"].items():
                rows.append({
                    "date": date,
                    "metric": metric_name,
                    "value": value
                })
        
        return pd.DataFrame(rows)

def create_metrics_trend_chart(metrics_df: pd.DataFrame, metric_names: Optional[List[str]] = None) -> go.Figure:
    """
    Create a line chart showing trends in model metrics over time.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame with metrics history
        metric_names (List[str]): List of metrics to include, defaults to all
        
    Returns:
        go.Figure: Plotly figure with metrics trends
    """
    if metrics_df.empty:
        # Create an empty chart with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No metrics data available yet",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color=COLOR_SYSTEM['PRIMARY']['CONTRAST'])
        )
        return format_chart_for_dark_mode(fig, "Model Metrics Trends", height=400)
    
    # Filter by selected metrics if provided
    if metric_names:
        filtered_df = metrics_df[metrics_df["metric"].isin(metric_names)]
    else:
        filtered_df = metrics_df
    
    if filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No matching metrics found",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color=COLOR_SYSTEM['PRIMARY']['CONTRAST'])
        )
        return format_chart_for_dark_mode(fig, "Model Metrics Trends", height=400)
    
    # Create figure with a separate axis for each metric
    fig = go.Figure()
    
    # Get unique metrics and assign colors
    unique_metrics = filtered_df["metric"].unique()
    colors = px.colors.qualitative.Bold[:len(unique_metrics)]
    color_map = dict(zip(unique_metrics, colors))
    
    # Add each metric as a separate line
    for metric in unique_metrics:
        metric_data = filtered_df[filtered_df["metric"] == metric]
        
        fig.add_trace(go.Scatter(
            x=metric_data["date"],
            y=metric_data["value"],
            mode="lines+markers",
            name=metric.upper(),
            line=dict(color=color_map[metric], width=2),
            marker=dict(size=8, color=color_map[metric]),
            hovertemplate=(
                "<b>%{x}</b><br>" +
                f"{metric.upper()}: %{{y:.4f}}<br>" +
                "<extra></extra>"
            )
        ))
    
    # Format chart
    fig = format_chart_for_dark_mode(fig, "Model Metrics Over Time", height=400)
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Metric Value",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def show_model_monitoring(model_name: str = "cpi_prediction") -> None:
    """
    Display model monitoring dashboard in Streamlit.
    
    Args:
        model_name (str): Name of the model to monitor
    """
    st.title("Model Performance Monitoring")
    
    # Create model monitor instance
    monitor = ModelMonitor(model_name)
    
    # Get metrics history
    metrics_df = monitor.get_metrics_history()
    
    # Add a brief explanation
    st.markdown("""
    This section tracks the CPI prediction model's performance over time to ensure predictions 
    remain accurate and consistent.
    """)
    
    # Display key monitoring stats
    col1, col2 = st.columns(2)
    
    with col1:
        num_records = len(metrics_df["date"].unique()) if not metrics_df.empty else 0
        st.metric("Performance Records", num_records)
    
    with col2:
        last_updated = monitor.monitoring_data[model_name]["last_updated"]
        if last_updated:
            st.metric("Last Updated", last_updated)
        else:
            st.metric("Last Updated", "Never")
    
    # Metrics trend chart
    st.subheader("Performance Metrics Trends")
    
    # Add metric selection
    if not metrics_df.empty:
        available_metrics = sorted(metrics_df["metric"].unique())
        selected_metrics = st.multiselect(
            "Select metrics to display",
            options=available_metrics,
            default=available_metrics[:3] if len(available_metrics) > 3 else available_metrics
        )
    else:
        selected_metrics = []
    
    metrics_chart = create_metrics_trend_chart(metrics_df, selected_metrics)
    st.plotly_chart(metrics_chart, use_container_width=True)
    
    # Add manual testing section
    with st.expander("Manual Performance Testing"):
        st.markdown("""
        Use this section to log new performance metrics after testing the model on recent data.
        This helps track model performance over time.
        """)
        
        with st.form("log_metrics_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                mae = st.number_input("Mean Absolute Error (MAE)", min_value=0.0, format="%.4f")
                rmse = st.number_input("Root Mean Squared Error (RMSE)", min_value=0.0, format="%.4f")
            
            with col2:
                r2 = st.number_input("R² Score", min_value=-1.0, max_value=1.0, format="%.4f")
                test_date = st.date_input("Test Date", value=datetime.datetime.now())
            
            submit = st.form_submit_button("Log Metrics")
            
            if submit:
                # Collect metrics
                new_metrics = {
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2
                }
                
                # Log metrics
                monitor.log_metrics(new_metrics, date=test_date.strftime("%Y-%m-%d"))
                
                st.success("✅ Metrics logged successfully.")
                
                # Refresh the page to show new data
                st.rerun()