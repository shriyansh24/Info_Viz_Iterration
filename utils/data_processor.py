"""
Data processing utilities for the CPI Analysis & Prediction Dashboard.
Handles data loading, cleaning, and feature engineering with enhanced visualization and explainability.
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data() -> Optional[pd.DataFrame]:
    """
    Load CPI data from source with enhanced error handling.
    This function would typically load from a database or file.
    
    Returns:
        Optional[pd.DataFrame]: DataFrame with CPI data or None if loading fails
    """
    try:
        # For demonstration, we'll simulate loading data
        # In a real application, this would connect to a database or load from files
        
        # Check if sample data exists in this environment
        sample_data_path = os.path.join(os.path.dirname(__file__), 'data', 'cpi_data.csv')
        
        if os.path.exists(sample_data_path):
            logger.info(f"Loading data from {sample_data_path}")
            return pd.read_csv(sample_data_path)
        
        # If no sample data file exists, create a simple synthetic dataset for demonstration
        # This is just a fallback for testing the application structure
        logger.warning("No data file found. Creating minimal demo dataset.")
        
        # Create synthetic data - this is just for application structure testing
        # In a real application, you would return None or an appropriate error
        data = {
            'Type': ['Won', 'Won', 'Won', 'Lost', 'Lost', 'Won', 'Lost', 'Won', 'Lost', 'Won',
                    'Won', 'Lost', 'Won', 'Lost', 'Won', 'Lost', 'Won', 'Lost', 'Won', 'Lost'],
            'IR': [20, 45, 10, 15, 50, 30, 25, 40, 35, 5,
                  15, 55, 25, 30, 60, 5, 10, 20, 70, 40],
            'LOI': [10, 15, 20, 15, 10, 12, 18, 8, 25, 15,
                   10, 20, 15, 12, 5, 25, 20, 10, 8, 15],
            'Completes': [200, 500, 300, 250, 400, 350, 200, 600, 300, 150,
                         250, 400, 300, 350, 500, 150, 200, 250, 450, 200],
            'CPI': [15.50, 10.25, 25.75, 30.00, 18.50, 12.75, 22.00, 9.50, 28.00, 35.00,
                   18.00, 15.50, 20.00, 25.00, 8.50, 40.00, 28.00, 18.50, 6.75, 22.50]
        }
        
        return pd.DataFrame(data)
    
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        return None

def clean_data(data: pd.DataFrame, filter_extremes: bool = True) -> pd.DataFrame:
    """
    Clean and prepare the CPI data with enhanced error handling and visualization preparation.
    
    Args:
        data (pd.DataFrame): Raw data DataFrame
        filter_extremes (bool): Whether to filter out extreme values
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        # Create a copy of the data to avoid modifying the original
        cleaned_data = data.copy()
        
        # Check for required columns
        required_columns = ['Type', 'IR', 'LOI', 'Completes', 'CPI']
        for col in required_columns:
            if col not in cleaned_data.columns:
                logger.error(f"Required column '{col}' not found in data")
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Ensure consistent types
        type_mappings = {
            'Type': str,
            'IR': float,
            'LOI': float,
            'Completes': int,
            'CPI': float
        }
        
        for col, dtype in type_mappings.items():
            try:
                if col == 'Completes':
                    cleaned_data[col] = cleaned_data[col].fillna(0).astype(int)
                else:
                    cleaned_data[col] = cleaned_data[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Error converting column '{col}' to {dtype}: {e}")
                # Try to handle common issues
                if dtype == float:
                    # Try to remove non-numeric characters and convert
                    if cleaned_data[col].dtype == 'object':
                        cleaned_data[col] = pd.to_numeric(cleaned_data[col].str.replace('[^0-9.]', '', regex=True), errors='coerce')
        
        # Handle missing values
        for col in cleaned_data.columns:
            if cleaned_data[col].isnull().any():
                if col in ['IR', 'LOI', 'Completes', 'CPI']:
                    # For numeric columns, fill with median
                    median_val = cleaned_data[col].median()
                    cleaned_data[col] = cleaned_data[col].fillna(median_val)
                    logger.info(f"Filled {cleaned_data[col].isnull().sum()} missing values in '{col}' with median ({median_val})")
                else:
                    # For non-numeric columns, fill with mode
                    mode_val = cleaned_data[col].mode()[0]
                    cleaned_data[col] = cleaned_data[col].fillna(mode_val)
                    logger.info(f"Filled {cleaned_data[col].isnull().sum()} missing values in '{col}' with mode ({mode_val})")
        
        # Ensure IR is a percentage
        if cleaned_data['IR'].max() > 100:
            logger.warning("IR values greater than 100 found, scaling to percentage")
            cleaned_data['IR'] = cleaned_data['IR'] / 100
        
        # Filter out extreme values if requested
        if filter_extremes:
            # For numeric columns, filter values outside 3 standard deviations
            for col in ['IR', 'LOI', 'Completes', 'CPI']:
                mean = cleaned_data[col].mean()
                std = cleaned_data[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
                # Count extreme values
                extreme_count = ((cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)).sum()
                
                if extreme_count > 0:
                    logger.info(f"Filtering {extreme_count} extreme values in '{col}'")
                    cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)]
        
        # Create IR bins for analysis with better labels for visualization
        cleaned_data['IR_Bin'] = pd.cut(
            cleaned_data['IR'],
            bins=[0, 10, 20, 30, 40, 50, 100],
            labels=['0-10%', '11-20%', '21-30%', '31-40%', '41-50%', '51-100%']
        )
        
        # Create LOI bins for analysis with better labels for visualization
        cleaned_data['LOI_Bin'] = pd.cut(
            cleaned_data['LOI'],
            bins=[0, 10, 15, 20, 30, 60],
            labels=['0-10min', '11-15min', '16-20min', '21-30min', '31-60min']
        )
        
        # Create sample size bins for analysis with better labels for visualization
        cleaned_data['Completes_Bin'] = pd.cut(
            cleaned_data['Completes'],
            bins=[0, 100, 300, 500, 1000, float('inf')],
            labels=['1-100', '101-300', '301-500', '501-1000', '1000+']
        )
        
        # Create numeric versions of the bins for heatmap and analysis
        cleaned_data['IR_Numeric'] = pd.cut(
            cleaned_data['IR'],
            bins=[0, 10, 20, 30, 50, 100],
            labels=[5, 15, 25, 40, 75]
        ).astype(float)
        
        cleaned_data['LOI_Numeric'] = pd.cut(
            cleaned_data['LOI'],
            bins=[0, 10, 15, 20, 30, 60],
            labels=[5, 12.5, 17.5, 25, 45]
        ).astype(float)
        
        return cleaned_data
    
    except Exception as e:
        logger.error(f"Error cleaning data: {e}", exc_info=True)
        # Return the original data if cleaning fails
        return data

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering for model training with enhanced visualization features.
    
    Args:
        data (pd.DataFrame): Cleaned data DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    try:
        # Create a copy of the data to avoid modifying the original
        engineered_data = data.copy()
        
        # Create interaction features
        engineered_data['IR_LOI_Ratio'] = engineered_data['IR'] / engineered_data['LOI'].replace(0, 0.1)
        engineered_data['IR_Completes_Ratio'] = engineered_data['IR'] / engineered_data['Completes'].replace(0, 1)
        engineered_data['LOI_Completes_Ratio'] = engineered_data['LOI'] / engineered_data['Completes'].replace(0, 1)
        
        # Create polynomial features
        engineered_data['IR_Squared'] = engineered_data['IR'] ** 2
        engineered_data['LOI_Squared'] = engineered_data['LOI'] ** 2
        
        # Create log transforms for skewed variables
        engineered_data['Log_Completes'] = np.log1p(engineered_data['Completes'])
        
        # Create interaction terms
        engineered_data['IR_LOI_Product'] = engineered_data['IR'] * engineered_data['LOI']
        engineered_data['Log_IR_LOI_Product'] = np.log1p(engineered_data['IR_LOI_Product'])
        
        # Calculate derived metrics
        engineered_data['CPI_per_Minute'] = engineered_data['CPI'] / engineered_data['LOI'].replace(0, 0.1)
        
        # Create efficiency metric for visualization
        engineered_data['Efficiency'] = engineered_data['IR'] * (1/engineered_data['LOI'].replace(0, 0.1)) * np.log1p(engineered_data['Completes'])
        
        # One-hot encode categorical variables
        engineered_data = pd.get_dummies(engineered_data, columns=['Type'], prefix=['Type'])
        
        # Ensure Type_Won exists (handle case where all data is one type)
        if 'Type_Won' not in engineered_data.columns:
            engineered_data['Type_Won'] = 0
            
        # Ensure Type_Lost exists
        if 'Type_Lost' not in engineered_data.columns:
            engineered_data['Type_Lost'] = 0
        
        return engineered_data
    
    except Exception as e:
        logger.error(f"Error engineering features: {e}", exc_info=True)
        # Return the original data if feature engineering fails
        return data

def prepare_model_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for model training with enhanced preprocessing.
    
    Args:
        data (pd.DataFrame): DataFrame with engineered features
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and target variable y
    """
    try:
        # Check that required columns exist
        required_columns = ['IR', 'LOI', 'Completes', 'CPI', 'Type_Won']
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"Required column '{col}' not found in data")
                return pd.DataFrame(), pd.Series()
        
        # Create feature matrix
        features = [
            'IR', 'LOI', 'Completes',
            'IR_LOI_Ratio', 'IR_Completes_Ratio', 'LOI_Completes_Ratio',
            'IR_Squared', 'LOI_Squared', 'Log_Completes',
            'IR_LOI_Product', 'Log_IR_LOI_Product', 'Type_Won'
        ]
        
        # Filter to only include features that exist in the dataframe
        available_features = [f for f in features if f in data.columns]
        
        # If any key features are missing, return empty data
        if not all(f in available_features for f in ['IR', 'LOI', 'Completes']):
            logger.error("Key features are missing from the data")
            return pd.DataFrame(), pd.Series()
        
        X = data[available_features].copy()
        y = data['CPI'].copy()
        
        # Check for NaN or infinity values
        if X.isnull().any().any() or np.isinf(X).values.any():
            logger.warning("NaN or infinity values found in features, attempting to clean")
            # Replace NaN with median
            for col in X.columns:
                X[col] = X[col].fillna(X[col].median())
            
            # Replace infinity with large values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
        
        if y.isnull().any() or np.isinf(y).any():
            logger.warning("NaN or infinity values found in target, attempting to clean")
            # Replace NaN with median
            y = y.fillna(y.median())
            
            # Replace infinity with large values
            y = y.replace([np.inf, -np.inf], y.median())
        
        return X, y
    
    except Exception as e:
        logger.error(f"Error preparing model data: {e}", exc_info=True)
        return pd.DataFrame(), pd.Series()
