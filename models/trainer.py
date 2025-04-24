"""
ML model training and evaluation for the CPI Analysis & Prediction Dashboard.
Includes functions for building, training, and evaluating prediction models with enhanced explainability.
"""

import pandas as pd
import numpy as np
import traceback
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, RANSACRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
import logging
import os
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Define model configurations
MODEL_CONFIGS = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': [True],  # True almost always performs better for CPI
            'positive': [False, True]  # Try constraining to positive coefficients
        }
    },
    'Ridge Regression': {
        'model': Ridge(random_state=42, max_iter=2000),  # Increase max_iter for convergence
        'params': {
            'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],  # Finer alpha grid
            'fit_intercept': [True],  # True almost always performs better
            'solver': ['lsqr', 'saga']  # These solvers work best for our data
        }
    },
    'Huber Regression': {
        'model': HuberRegressor(fit_intercept=True),
        'params': {
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'epsilon': [1.1, 1.35, 1.5, 1.75, 2.0],  # Wider range to handle outliers
            'max_iter': [500, 1000, 2000]  # Increase for convergence
        }
    },
    'RANSAC Regression': {
        'model': RANSACRegressor(random_state=42),
        'params': {
            'min_samples': [0.1, 0.3, 0.5],
            'max_trials': [100, 200],
            'loss': ['absolute_loss', 'squared_loss']
        }
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),  # Use parallel processing
        'params': {
            'n_estimators': [100, 150, 200, 300],  # More trees for better performance
            'max_depth': [10, 15, 20, 25, None],   # Deeper trees for complex relationships
            'min_samples_split': [2, 5, 8],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],  # Try different feature selection methods
            'bootstrap': [True]  # Bootstrap is almost always better
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 150, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],  # Finer learning rate grid
            'max_depth': [3, 5, 7, 9],                 # Try deeper trees
            'min_samples_split': [2, 5, 8],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.7, 0.8, 0.9, 1.0],         # Try more aggressive subsampling
            'max_features': ['sqrt', 'log2', None],     # Try different feature selection
            'n_iter_no_change': [5],
            'validation_fraction': [0.1]
        }
    },
    'ElasticNet': {  # Add ElasticNet which often works well for CPI prediction
        'model': ElasticNet(random_state=42, max_iter=2000),
        'params': {
            'alpha': [0.001, 0.01, 0.1, 0.5, 1.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # Mix between L1 and L2 regularization
            'fit_intercept': [True],
            'selection': ['cyclic', 'random']
        }
    }
}

def build_models(X: pd.DataFrame, y: pd.Series, 
                do_hyperparameter_tuning: bool = False) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Build prediction models for CPI with enhanced robustness and explainability.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        do_hyperparameter_tuning (bool, optional): Whether to perform hyperparameter tuning. Defaults to False.
    
    Returns:
        Tuple[Dict[str, Any], Dict[str, Dict[str, float]], pd.DataFrame]: 
            - Dictionary of trained models
            - Dictionary of model scores
            - DataFrame with feature importance
    """
    try:
        logger.info("Starting model building process")
        
        # Validate input data
        if X.empty or len(y) == 0:
            logger.error("Empty input data provided")
            return {}, {}, pd.DataFrame(columns=['Feature', 'Importance'])
        
        # Check for NaN or infinite values
        if X.isnull().any().any() or np.isinf(X).any().any() or y.isnull().any() or np.isinf(y).any():
            logger.warning("Input data contains NaN or infinite values - attempting to clean data")
            
            # Replace NaN values with median
            for col in X.columns:
                X[col] = X[col].fillna(X[col].median())
            
            # Replace infinite values with large but finite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            # Handle target variable
            y = y.fillna(y.median())
            y = y.replace([np.inf, -np.inf], y.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        # Check for sufficient data
        if len(X_train) < 10 or len(X_test) < 5:
            logger.warning(f"Very small dataset: {len(X_train)} training samples, {len(X_test)} test samples")
            if len(X_train) < 5:
                logger.error("Insufficient data for modeling")
                return {}, {}, pd.DataFrame(columns=['Feature', 'Importance'])
        
        # Build models with or without hyperparameter tuning
        if do_hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning")
            models = build_models_with_tuning(X_train, y_train, X_test, y_test)
        else:
            logger.info("Using default model parameters")
            models = build_models_default(X_train, y_train, X_test, y_test)
        
        # Get trained models
        trained_models = models.get('trained_models', {})
        model_scores = models.get('model_scores', {})
        
        # Extract feature importance (try from multiple models if available)
        feature_importance = pd.DataFrame(columns=['Feature', 'Importance'])
        
        # Try to get feature importance from multiple model types
        importance_models = ['Random Forest', 'Gradient Boosting']
        for model_name in importance_models:
            if model_name in trained_models:
                try:
                    model = trained_models[model_name]
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        feature_importance = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False)
                        logger.info(f"Feature importance calculated from {model_name}")
                        break
                except Exception as e:
                    logger.warning(f"Could not extract feature importance from {model_name}: {e}")
        
        if feature_importance.empty:
            logger.warning("Could not calculate feature importance from any model")
            # Create empty DataFrame with proper structure
            feature_importance = pd.DataFrame(columns=['Feature', 'Importance'])
        
        return trained_models, model_scores, feature_importance
    
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in build_models: {error_details}")
        # Return empty objects
        return {}, {}, pd.DataFrame(columns=['Feature', 'Importance'])

def build_models_default(X_train: pd.DataFrame, y_train: pd.Series, 
                         X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Build models with default parameters but added robustness and explainability.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target variable
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): Test target variable
    
    Returns:
        Dict[str, Any]: Dictionary containing trained models and model scores
    """
    # Initialize dictionaries
    trained_models = {}
    model_scores = {}
    
    # Define models to train with optimized parameters for CPI prediction
    models = {
        'Ridge Regression': Ridge(
            alpha=1.0, 
            random_state=42, 
            solver='lsqr',
            fit_intercept=True,
            max_iter=2000
        ),
        'Huber Regression': HuberRegressor(
            alpha=0.01, 
            epsilon=1.35,
            max_iter=500,
            fit_intercept=True,
            tol=1e-5
        ),
        'ElasticNet': ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=42,
            max_iter=2000,
            tol=1e-4,
            selection='random'
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=150,
            max_depth=10,        # shallower trees
            min_samples_split=5,
            min_samples_leaf=5,  # larger leaf size
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            max_features='sqrt',
            random_state=42,
            n_iter_no_change=5,      # early stopping
            validation_fraction=0.1
        )
    }
    
    # Use a scaler to avoid numerical issues
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models with error handling for each model
    for name, model in models.items():
        try:
            logger.info(f"Training {name} model")
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model

            # compute CV RMSE to monitor over‑fit
            cv_mse = -cross_val_score(
                model, X_train_scaled, y_train,
                cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1
            ).mean()
            model_scores[name]['CV_RMSE'] = np.sqrt(cv_mse)

            # Make predictions
            try:
                y_pred = model.predict(X_test_scaled)
                
                # Ensure predictions are within reasonable bounds
                y_pred = np.clip(y_pred, 0, y_test.max() * 3)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_scores[name].update({
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2
                })
                
                logger.info(f"{name} model trained. R² score: {r2:.4f}")
            except Exception as eval_error:
                logger.error(f"Error evaluating {name} model: {eval_error}")
                model_scores[name].update({
                    'MSE': float('nan'),
                    'RMSE': float('nan'),
                    'MAE': float('nan'),
                    'R²': float('nan'),
                    'Error': str(eval_error)
                })
        except Exception as e:
            logger.error(f"Error training {name} model: {e}")
            # Model failed, don't add it to trained_models
            model_scores[name] = {
                'Error': str(e),
                'Status': 'Failed'
            }
    
    # If no models were successfully trained, try a very basic linear model as fallback
    if not trained_models:
        try:
            logger.warning("All standard models failed, attempting basic linear regression as fallback")
            fallback_model = LinearRegression()
            fallback_model.fit(X_train_scaled, y_train)
            trained_models['Fallback Linear'] = fallback_model
            
            # Evaluate fallback model
            y_pred = fallback_model.predict(X_test_scaled)
            y_pred = np.clip(y_pred, 0, y_test.max() * 3)
            
            mse = mean_squared_error(y_test, y_pred)
            model_scores['Fallback Linear'] = {
                'MSE': mse,
                'RMSE': np.sqrt(mse),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R²': r2_score(y_test, y_pred)
            }
        except Exception as e:
            logger.error(f"Even fallback model failed: {e}")
    
    return {
        'trained_models': trained_models,
        'model_scores': model_scores,
        'scaler': scaler  # Include the scaler for future preprocessing
    }

def build_models_with_tuning(X_train: pd.DataFrame, y_train: pd.Series, 
                           X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Build models with hyperparameter tuning using GridSearchCV with enhanced explainability.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target variable
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): Test target variable
    
    Returns:
        Dict[str, Any]: Dictionary containing tuned models and model scores
    """
    # Initialize dictionaries
    trained_models = {}
    model_scores = {}
    best_params = {}
    
    # Use a scaler to ensure numerical stability
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models with hyperparameter tuning - limit to fewer models for efficiency
    tuning_models = ['Ridge Regression', 'ElasticNet', 'Random Forest', 'Gradient Boosting']
    
    for name in tuning_models:
        if name in MODEL_CONFIGS:
            try:
                logger.info(f"Tuning {name} model")
                config = MODEL_CONFIGS[name]
                
                # Create grid search with reduced parameter space if data is small
                if len(X_train) < 50:
                    logger.warning(f"Small dataset detected, reducing parameter space for {name}")
                    
                    # Simplified params for small datasets
                    if name == 'Ridge Regression':
                        param_grid = {'alpha': [0.1, 1.0, 10.0]}
                    elif name == 'Random Forest':
                        param_grid = {
                            'n_estimators': [50],
                            'max_depth': [5, None]
                        }
                    elif name == 'Gradient Boosting':
                        param_grid = {
                            'n_estimators': [50],
                            'max_depth': [3, 5]
                        }
                    else:
                        param_grid = config['params']
                else:
                    param_grid = config['params']
                
                # Create grid search
                grid_search = GridSearchCV(
                    config['model'],
                    param_grid,
                    cv=min(5, len(X_train) // 5),  # Ensure we don't have too many folds for small datasets
                    scoring='neg_mean_squared_error',
                    n_jobs=-1 if len(X_train) > 100 else 1  # Use parallel processing for larger datasets
                )
                
                # Fit grid search
                grid_search.fit(X_train_scaled, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                best_params[name] = grid_search.best_params_
                
                # Add to trained models
                trained_models[name] = best_model
                
                # Evaluate model
                y_pred = best_model.predict(X_test_scaled)
                
                # Ensure predictions are within reasonable bounds
                y_pred = np.clip(y_pred, 0, y_test.max() * 3)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_scores[name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2,
                    'Best Params': grid_search.best_params_
                }
                
                logger.info(f"{name} model tuned. R² score: {r2:.4f}, Best params: {grid_search.best_params_}")
            
            except Exception as e:
                logger.error(f"Error tuning {name} model: {e}")
                model_scores[name] = {
                    'Error': str(e),
                    'Status': 'Failed'
                }
    
    # If no models were successfully tuned, fall back to default models
    if not trained_models:
        logger.warning("All tuning failed, falling back to default models")
        default_models = build_models_default(X_train, y_train, X_test, y_test)
        trained_models = default_models.get('trained_models', {})
        model_scores.update(default_models.get('model_scores', {}))
    
    return {
        'trained_models': trained_models,
        'model_scores': model_scores,
        'best_params': best_params,
        'scaler': scaler  # Include the scaler for future preprocessing
    }

def save_models(models: Dict[str, Any], path: str = MODEL_DIR) -> bool:
    """
    Save trained models to disk.
    
    Args:
        models (Dict[str, Any]): Dictionary of trained models
        path (str, optional): Directory to save models to. Defaults to MODEL_DIR.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save each model
        for name, model in models.items():
            model_path = os.path.join(path, f"{name.replace(' ', '_').lower()}.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Saved model {name} to {model_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        return False

def save_model_pipeline(pipeline: Dict[str, Any], path: str = MODEL_DIR) -> bool:
    """
    Save a model pipeline (models, scaler, etc.) to disk.
    
    Args:
        pipeline (Dict[str, Any]): Dictionary containing models, scaler, etc.
        path (str, optional): Directory to save pipeline to. Defaults to MODEL_DIR.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save pipeline
        pipeline_path = os.path.join(path, "model_pipeline.pkl")
        joblib.dump(pipeline, pipeline_path)
        logger.info(f"Saved model pipeline to {pipeline_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving model pipeline: {e}")
        return False

def load_models(path: str = MODEL_DIR) -> Dict[str, Any]:
    """
    Load trained models from disk.
    
    Args:
        path (str, optional): Directory to load models from. Defaults to MODEL_DIR.
    
    Returns:
        Dict[str, Any]: Dictionary of loaded models
    """
    try:
        # Check if directory exists
        if not os.path.exists(path):
            logger.warning(f"Model directory {path} does not exist")
            return {}
        
        # Check if pipeline exists
        pipeline_path = os.path.join(path, "model_pipeline.pkl")
        if os.path.exists(pipeline_path):
            logger.info(f"Loading model pipeline from {pipeline_path}")
            return joblib.load(pipeline_path)
        
        # Otherwise load individual models
        models = {}
        
        # Get all .pkl files
        model_files = [f for f in os.listdir(path) if f.endswith('.pkl')]
        
        for file in model_files:
            # Get model name
            name = file.replace('.pkl', '').replace('_', ' ').title()
            
            # Load model
            model_path = os.path.join(path, file)
            models[name] = joblib.load(model_path)
            logger.info(f"Loaded model {name} from {model_path}")
        
        return models
    
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return {}

def cross_validate_models(X: pd.DataFrame, y: pd.Series, 
                        models: Dict[str, Any] = None, 
                        cv: int = 5) -> Dict[str, Dict[str, float]]:
    """
    Perform cross-validation on multiple models for enhanced validation.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        models (Dict[str, Any], optional): Dictionary of models to cross-validate. 
            If None, default models will be used. Defaults to None.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.
    
    Returns:
        Dict[str, Dict[str, float]]: Dictionary of cross-validation scores for each model
    """
    try:
        # If no models provided, use default models
        if models is None:
            models = {
                'Ridge Regression': Ridge(alpha=1.0, random_state=42),
                'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000),
                'Random Forest': RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, random_state=42)
            }
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Initialize results dictionary
        cv_results = {}
        
        # Perform cross-validation for each model
        for name, model in models.items():
            try:
                logger.info(f"Cross-validating {name} model")
                
                # Calculate metrics
                mse_scores = -cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')
                rmse_scores = np.sqrt(mse_scores)
                r2_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
                mae_scores = -cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error')
                
                # Store results
                cv_results[name] = {
                    'MSE': {
                        'mean': mse_scores.mean(),
                        'std': mse_scores.std(),
                        'values': mse_scores.tolist()
                    },
                    'RMSE': {
                        'mean': rmse_scores.mean(),
                        'std': rmse_scores.std(),
                        'values': rmse_scores.tolist()
                    },
                    'MAE': {
                        'mean': mae_scores.mean(),
                        'std': mae_scores.std(),
                        'values': mae_scores.tolist()
                    },
                    'R²': {
                        'mean': r2_scores.mean(),
                        'std': r2_scores.std(),
                        'values': r2_scores.tolist()
                    }
                }
                
                logger.info(f"{name} cross-validation complete. Mean R²: {r2_scores.mean():.4f}")
            
            except Exception as e:
                logger.error(f"Error cross-validating {name} model: {e}")
                cv_results[name] = {
                    'Error': str(e),
                    'Status': 'Failed'
                }
        
        return cv_results
    
    except Exception as e:
        logger.error(f"Error in cross_validate_models: {e}")
        return {}

def evaluate_model_assumptions(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Evaluate assumptions of the regression model for enhanced explainability.
    
    Args:
        model (Any): Trained regression model
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
    
    Returns:
        Dict[str, Any]: Dictionary of assumption evaluation results
    """
    try:
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        
        # Calculate residuals
        residuals = y - y_pred
        
        # Check linearity - correlation between predictions and residuals
        linearity_corr = np.corrcoef(y_pred, residuals)[0, 1]
        
        # Check homoscedasticity - Breusch-Pagan test (simplified)
        # Using squared residuals correlation with predictions as proxy
        sq_residuals = residuals ** 2
        homoscedasticity_corr = np.corrcoef(y_pred, sq_residuals)[0, 1]
        
        # Check normality of residuals - Shapiro-Wilk test would be ideal
        # but it's not available in scikit-learn, so we'll use skewness and kurtosis
        skewness = float(pd.Series(residuals).skew())
        kurtosis = float(pd.Series(residuals).kurtosis())
        
        # Check for influential observations - using cook's distance approximation
        # For simplicity, we'll just count how many residuals are > 3 std devs
        std_residuals = residuals / residuals.std()
        n_influential = np.sum(np.abs(std_residuals) > 3)
        
        # Check for multicollinearity - using VIF would be ideal
        # but for simplicity, we'll use correlation matrix
        corr_matrix = X.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr = corr_matrix.max().max()
        high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j]) 
                          for i in range(len(corr_matrix.index)) 
                          for j in range(i+1, len(corr_matrix.columns))
                          if corr_matrix.iloc[i, j] > 0.8]
        
        # Compile results
        assumption_results = {
            'linearity': {
                'metric': linearity_corr,
                'interpretation': 'Good' if abs(linearity_corr) < 0.3 else 'Questionable',
                'description': 'Correlation between predictions and residuals'
            },
            'homoscedasticity': {
                'metric': homoscedasticity_corr,
                'interpretation': 'Good' if abs(homoscedasticity_corr) < 0.3 else 'Questionable',
                'description': 'Correlation between predictions and squared residuals'
            },
            'normality': {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'interpretation': 'Good' if abs(skewness) < 1 and abs(kurtosis) < 3 else 'Questionable',
                'description': 'Skewness and kurtosis of residuals'
            },
            'influential_observations': {
                'count': n_influential,
                'percentage': n_influential / len(residuals) * 100,
                'interpretation': 'Good' if n_influential / len(residuals) < 0.05 else 'Questionable',
                'description': 'Number of observations with standardized residuals > 3'
            },
            'multicollinearity': {
                'max_correlation': max_corr,
                'high_correlation_pairs': high_corr_pairs,
                'interpretation': 'Good' if max_corr < 0.8 else 'Questionable',
                'description': 'Maximum correlation between features'
            }
        }
        
        return assumption_results
    
    except Exception as e:
        logger.error(f"Error evaluating model assumptions: {e}")
        return {
            'error': str(e),
            'status': 'Failed'
        }
