"""
PPI forecasting model for early inflation warning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from statsmodels.tsa.ardl import ARDL
from src.models.base_model import BaseForecastModel

class PPIForecaster(BaseForecastModel):
    """Forecast PPI for early inflation signals"""
    
    def __init__(self):
        """Initialize PPI forecaster"""
        super().__init__("PPI Model (Early Warning)")
        self.ardl_model = None
        self.target_name = 'PPI_YoY'
        self.supply_weight = 0.6  # Higher weight on supply factors
        self.wage_weight = 0.4
        
    def train(self, data: pd.DataFrame, target_col: str = 'PPI_YoY'):
        """
        Train ARDL model with emphasis on supply chain factors
        
        Args:
            data: DataFrame with inflation indicators
            target_col: Target variable name
        """
        self.target_name = target_col
        
        # Key features for PPI
        supply_features = ['Baltic_Dry_Index', 'Import_Price_Index']
        wage_features = ['Employment_Cost_Index', 'NFIB_Comp_Plans']
        
        # Check available features
        available_supply = [f for f in supply_features if any(f in col for col in data.columns)]
        available_wage = [f for f in wage_features if any(f in col for col in data.columns)]
        
        self.feature_names = available_supply + available_wage
        
        # Prepare data for ARDL
        model_data = self._prepare_ardl_data(data)
        
        if len(model_data) < 50:
            print(f"Warning: Limited data for PPI model ({len(model_data)} observations)")
            
        # Train ARDL model
        try:
            self._train_ardl(model_data)
            self.is_trained = True
        except Exception as e:
            print(f"ARDL training failed: {e}")
            self._train_fallback(model_data)
            
    def _prepare_ardl_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for ARDL model"""
        
        # Select relevant columns
        cols_to_use = []
        
        # Target
        if self.target_name in data.columns:
            cols_to_use.append(self.target_name)
            
        # Features - use any column containing our feature keywords
        for feature in self.feature_names:
            matching_cols = [col for col in data.columns if feature in col]
            cols_to_use.extend(matching_cols[:2])  # Limit to 2 variations per feature
            
        # Remove duplicates and clean
        cols_to_use = list(set(cols_to_use))
        model_data = data[cols_to_use].dropna()
        
        return model_data
        
    def _train_ardl(self, data: pd.DataFrame):
        """Train ARDL model"""
        
        if self.target_name not in data.columns:
            raise ValueError(f"Target {self.target_name} not in data")
            
        # ARDL specification
        # More lags for Baltic Dry (2-3 month lead)
        # Fewer lags for wages (1-2 month lead)
        
        exog_cols = [col for col in data.columns if col != self.target_name]
        
        if len(exog_cols) == 0:
            raise ValueError("No exogenous variables available")
            
        # Create ARDL model
        self.ardl_model = ARDL(
            data[self.target_name],
            lags=4,  # PPI lags
            exog=data[exog_cols],
            order=2  # Exogenous variable lags
        )
        
        # Fit model
        self.ardl_model = self.ardl_model.fit()
        
    def _train_fallback(self, data: pd.DataFrame):
        """Simple fallback model if ARDL fails"""
        
        from sklearn.linear_model import LinearRegression
        
        if self.target_name in data.columns:
            y = data[self.target_name].values
            
            # Use simple lagged features
            X = []
            for col in data.columns:
                if col != self.target_name:
                    X.append(data[col].shift(1).fillna(0).values)
                    
            if X:
                X = np.column_stack(X)
                valid_idx = ~np.isnan(y)
                
                self.fallback_model = LinearRegression()
                self.fallback_model.fit(X[valid_idx], y[valid_idx])
                self.is_trained = True
                
    def forecast(self, data: pd.DataFrame, horizons: List[int] = [3, 6]) -> Dict:
        """
        Generate PPI forecasts with focus on turning points
        
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
            if self.ardl_model is not None:
                try:
                    # ARDL forecast
                    forecast_result = self.ardl_model.forecast(steps=horizon)
                    point_forecast = float(forecast_result.iloc[-1])
                    
                except:
                    # Fallback to simple forecast
                    point_forecast = self._simple_forecast(data, horizon)
            else:
                point_forecast = self._simple_forecast(data, horizon)
                
            # Detect potential turning points
            turning_point = self._detect_turning_point(data, point_forecast)
            
            # Calculate confidence intervals
            lower, upper = self.calculate_confidence_intervals(point_forecast, horizon)
            
            forecasts[f'{horizon}m'] = {
                'forecast': point_forecast,
                'lower': lower,
                'upper': upper,
                'turning_point_signal': turning_point
            }
            
        return forecasts
        
    def _simple_forecast(self, data: pd.DataFrame, horizon: int) -> float:
        """Simple forecast based on recent trends"""
        
        if self.target_name in data.columns:
            recent_values = data[self.target_name].iloc[-6:].values
            trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            last_value = recent_values[-1]
            
            return last_value + trend * horizon
        else:
            return 2.5  # Default PPI assumption
            
    def _detect_turning_point(self, data: pd.DataFrame, forecast: float) -> str:
        """Detect potential inflation turning points"""
        
        if self.target_name not in data.columns:
            return "uncertain"
            
        recent = data[self.target_name].iloc[-6:].mean()
        
        if forecast > recent + 0.5:
            return "acceleration"
        elif forecast < recent - 0.5:
            return "deceleration"
        else:
            return "stable"