"""
CPI Analysis & Prediction Dashboard

A Streamlit dashboard for analyzing the Cost Per Interview (CPI) between won and lost bids
and predicting optimal pricing for new bids.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Any, Optional

# Import components
from components.overview import show_overview
from components.prediction import show_prediction
from components.insights import show_insights

# Import data processing utilities
from data_processor import load_data, clean_data, engineer_features

# Import theme utilities
from utils.theme import apply_custom_theme

# Import data quality analysis utilities
from utils.data_quality import show_data_analysis

# Import model monitoring utilities
from utils.model_monitoring import show_model_monitoring

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config for the app
st.set_page_config(
    page_title="CPI Analysis & Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    """Main function to run the Streamlit app."""
    try:
        # Apply custom theming
        apply_custom_theme()
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        
        # Add app info and selection
        page = st.sidebar.radio(
            "Select Section:",
            ["Overview", "CPI Prediction", "Insights & Recommendations", "Data Analysis", "Model Monitoring"]
        )
        
        # Sidebar filters
        st.sidebar.title("Data Filters")
        
        # Toggle for extreme value filtering
        filter_extremes = st.sidebar.checkbox("Filter out extreme values", value=True)
        
        # Add sidebar info
        with st.sidebar.expander("About this Dashboard"):
            st.markdown("""
            This dashboard analyzes Cost Per Interview (CPI) data to help optimize pricing
            strategy for market research bids. Key features:
            
            - **Overview**: High-level analysis of CPI patterns
            - **CPI Prediction**: ML-based price prediction tool
            - **Insights**: Strategic recommendations
            - **Data Analysis**: Comprehensive data quality assessment and feature engineering
            - **Model Monitoring**: Track prediction model performance over time
            
            Data is filtered by default to remove extreme outliers.
            """)
        
        # Load and process data
        try:
            # Load data from Excel files
            data_dict = load_data()
            
            # Select filtered or unfiltered data based on user preference
            if filter_extremes:
                won_data = data_dict['won_filtered']
                lost_data = data_dict['lost_filtered']
                combined_data = data_dict['combined_filtered']
            else:
                won_data = data_dict['won']
                lost_data = data_dict['lost']
                combined_data = data_dict['combined']
            
            # Basic checks
            if len(won_data) == 0 or len(lost_data) == 0:
                st.warning("One or more bid categories has no data. Analysis may be limited.")
            
            # Engineer features for modeling
            combined_data_engineered = engineer_features(combined_data)
            
            # Display data source information
            st.sidebar.markdown("---")
            st.sidebar.subheader("Data Source")
            st.sidebar.markdown(
                f"**Won bids**: {len(won_data)} records<br>"
                f"**Lost bids**: {len(lost_data)} records<br>"
                f"**Total**: {len(combined_data)} records",
                unsafe_allow_html=True
            )
            
            # Add data quality information
            if filter_extremes:
                st.sidebar.markdown(
                    "**Note**: Extreme values have been filtered out to improve analysis quality.",
                    unsafe_allow_html=True
                )
        
        except Exception as e:
            logger.error(f"Error loading or processing data: {e}", exc_info=True)
            st.error(f"Error loading or processing data: {str(e)}")
            st.stop()
            
        # Display the selected page
        if page == "Overview":
            show_overview(won_data, lost_data, combined_data)
        elif page == "CPI Prediction":
            show_prediction(combined_data_engineered, won_data, lost_data)
        elif page == "Insights & Recommendations":
            show_insights(won_data, lost_data, combined_data)
        elif page == "Data Analysis":
            show_data_analysis(won_data, lost_data, combined_data)
        elif page == "Model Monitoring":
            show_model_monitoring()
    
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
