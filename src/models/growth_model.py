"""
Growth forecasting model for ISM PMI
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA
from src.models.base_model import BaseForecastModel

class GrowthForecaster(BaseForecastModel):
    """Forecast economic growth using ISM PMI"""
    
    def __init__(self):
        """Initialize growth forecaster"""
        super().__init__("Growth Model (ISM PMI)")
        self.var_model = None
        self.pca_model = None
        self.factor_model = None
        self.target_name = 'ISM_PMI'
        
    def train(self, data: pd.DataFrame, target_col: str = 'ISM_PMI'):
        """
        Train VAR and factor models
        
        Args:
            data: DataFrame with growth indicators
            target_col: Target variable name
        """
        self.target_name = target_col
        
        # Select features
        feature_cols = [col for col in data.columns if col != target_col]
        self.feature_names = feature_cols
        
        # Handle missing data
        clean_data = data.dropna()
        
        if len(clean_data) < 100:
            print(f"Warning: Only {len(clean_data)} observations for training")
            
        # Train VAR model
        try:
            self._train_var(clean_data)
        except Exception as e:
            print(f"VAR training failed: {e}")
            
        # Train factor model
        try:
            self._train_factor_model(clean_data)
        except Exception as e:
            print(f"Factor model training failed: {e}")
            
        self.is_trained = True
        
    def _train_var(self, data: pd.DataFrame):
        """Train Vector Autoregression model"""
        
        # Select optimal lag order
        model = VAR(data)
        
        # Try different lag orders
        max_lags = min(12, len(data) // 10)
        
        try:
            lag_order = model.select_order(maxlags=max_lags)
            optimal_lag = lag_order.aic
            
            if optimal_lag == 0:
                optimal_lag = 2  # Default to 2 if 0 selected
                
        except:
            optimal_lag = 2  # Default fallback
            
        # Fit VAR model
        self.var_model = model.fit(optimal_lag)
        
    def _train_factor_model(self, data: pd.DataFrame):
        """Train dynamic factor model using PCA"""
        
        # Extract features only
        features = data[self.feature_names]
        
        # Standardize features
        features_std = (features - features.mean()) / features.std()
        
        # PCA for factor extraction
        self.pca_model = PCA(n_components=min(3, len(self.feature_names)))
        factors = self.pca_model.fit_transform(features_std.fillna(0))
        
        # Simple regression of target on factors
        if self.target_name in data.columns:
            from sklearn.linear_model import LinearRegression
            
            y = data[self.target_name].values
            valid_idx = ~np.isnan(y)
            
            self.factor_model = LinearRegression()
            self.factor_model.fit(factors[valid_idx], y[valid_idx])
            
    def forecast(self, data: pd.DataFrame, horizons: List[int] = [3, 6, 9]) -> Dict:
        """
        Generate growth forecasts
        
        Args:
            data: Recent data for forecasting
            horizons: Forecast horizons in months
            
        Returns:
            Dictionary with forecasts for each horizon
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
            
        forecasts = {}
        
        for horizon in horizons:
            # VAR forecast
            var_forecast = self._var_forecast(data, horizon) if self.var_model else None
            
            # Factor forecast
            factor_forecast = self._factor_forecast(data, horizon) if self.factor_model else None
            
            # Combine forecasts (60% VAR, 40% factor)
            if var_forecast is not None and factor_forecast is not None:
                combined = 0.6 * var_forecast + 0.4 * factor_forecast
            elif var_forecast is not None:
                combined = var_forecast
            elif factor_forecast is not None:
                combined = factor_forecast
            else:
                # Fallback: use last known value
                combined = data[self.target_name].iloc[-1] if self.target_name in data.columns else 50.0
                
            # Calculate confidence intervals
            lower, upper = self.calculate_confidence_intervals(combined, horizon)
            
            forecasts[f'{horizon}m'] = {
                'forecast': combined,
                'lower': lower,
                'upper': upper,
                'var_forecast': var_forecast,
                'factor_forecast': factor_forecast
            }
            
        return forecasts
        
    def _var_forecast(self, data: pd.DataFrame, horizon: int) -> Optional[float]:
        """Generate VAR forecast"""
        
        if self.var_model is None:
            return None
            
        try:
            # Prepare data for VAR
            last_values = data[self.var_model.names].iloc[-self.var_model.k_ar:].values
            
            # Generate forecast
            forecast = self.var_model.forecast(last_values, steps=horizon)
            
            # Extract target variable forecast
            target_idx = self.var_model.names.index(self.target_name)
            return forecast[-1, target_idx]
            
        except:
            return None
            
    def _factor_forecast(self, data: pd.DataFrame, horizon: int) -> Optional[float]:
        """Generate factor-based forecast"""
        
        if self.factor_model is None or self.pca_model is None:
            return None
            
        try:
            # Extract features
            features = data[self.feature_names].iloc[-1:].fillna(0)
            
            # Standardize
            features_std = (features - features.mean()) / features.std()
            
            # Transform to factors
            factors = self.pca_model.transform(features_std)
            
            # Predict
            forecast = self.factor_model.predict(factors)[0]
            
            # Add small adjustment for horizon
            forecast += 0.1 * (50 - forecast) * (horizon / 12)  # Mean reversion
            
            return forecast
            
        except:
            return None