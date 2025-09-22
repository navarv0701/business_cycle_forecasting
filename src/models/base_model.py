"""
Base model class for all forecasting models
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

class BaseForecastModel(ABC):
    """Abstract base class for forecasting models"""
    
    def __init__(self, name: str):
        """Initialize base model"""
        self.name = name
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.target_name = None
        
    @abstractmethod
    def train(self, data: pd.DataFrame, target_col: Optional[str] = None):
        """Train the model on historical data"""
        pass
        
    @abstractmethod
    def forecast(self, data: pd.DataFrame, horizons: List[int]) -> Dict:
        """Generate forecasts for specified horizons"""
        pass
        
    def backtest(self, data: pd.DataFrame, window: int = 24, step: int = 1) -> Dict:
        """
        Perform rolling window backtesting
        
        Args:
            data: Historical data for backtesting
            window: Training window size in months
            step: Step size for rolling window
            
        Returns:
            Dictionary of performance metrics
        """
        predictions = []
        actuals = []
        
        # Convert to monthly if needed
        monthly_data = data.resample('M').last()
        
        for i in range(window, len(monthly_data) - 6, step):
            # Train on window
            train_data = monthly_data.iloc[i-window:i]
            
            # Forecast 6 months ahead
            test_point = monthly_data.iloc[i:i+1]
            
            try:
                # Retrain model
                self.train(train_data)
                
                # Generate forecast
                forecast = self.forecast(test_point, horizons=[6])
                
                if '6m' in forecast and self.target_name in monthly_data.columns:
                    pred = forecast['6m']['forecast']
                    actual = monthly_data[self.target_name].iloc[i+6] if i+6 < len(monthly_data) else np.nan
                    
                    if not np.isnan(actual):
                        predictions.append(pred)
                        actuals.append(actual)
            except:
                continue
                
        if len(predictions) > 0:
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            # Directional accuracy
            if len(predictions) > 1:
                pred_direction = np.diff(predictions) > 0
                actual_direction = np.diff(actuals) > 0
                directional_accuracy = (pred_direction == actual_direction).mean()
            else:
                directional_accuracy = 0.5
                
            return {
                'rmse': rmse,
                'mae': mae,
                'directional_accuracy': directional_accuracy,
                'n_forecasts': len(predictions)
            }
        else:
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'directional_accuracy': 0.5,
                'n_forecasts': 0
            }
            
    def calculate_confidence_intervals(self, forecast: float, horizon: int, 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence intervals for forecast
        
        Simple approximation - should be overridden by specific models
        """
        # Simple approximation: wider intervals for longer horizons
        std_error = 0.5 * np.sqrt(horizon)
        z_score = 1.96 if confidence == 0.95 else 2.58
        
        lower = forecast - z_score * std_error
        upper = forecast + z_score * std_error
        
        return lower, upper