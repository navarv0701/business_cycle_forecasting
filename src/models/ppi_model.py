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
        self.fallback_model = None
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
            print("Falling back to simple linear model...")
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
        
        # Select columns that exist
        cols_to_use = [col for col in cols_to_use if col in data.columns]
        
        if not cols_to_use:
            print("Warning: No matching columns found for PPI model")
            return pd.DataFrame()
            
        model_data = data[cols_to_use].copy()
        
        # Clean the data
        # Replace infinities with NaN
        model_data = model_data.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with NaN values
        model_data = model_data.dropna()
        
        # Remove extreme outliers (values beyond 5 standard deviations)
        for col in model_data.columns:
            if col != self.target_name:  # Don't clip the target
                mean = model_data[col].mean()
                std = model_data[col].std()
                if std > 0:
                    lower_bound = mean - 5 * std
                    upper_bound = mean + 5 * std
                    model_data[col] = model_data[col].clip(lower=lower_bound, upper=upper_bound)
        
        return model_data
        
    def _train_ardl(self, data: pd.DataFrame):
        """Train ARDL model"""
        
        if self.target_name not in data.columns:
            raise ValueError(f"Target {self.target_name} not in data")
            
        # ARDL specification
        exog_cols = [col for col in data.columns if col != self.target_name]
        
        if len(exog_cols) == 0:
            raise ValueError("No exogenous variables available")
            
        # Reduce complexity if too many features
        if len(exog_cols) > 5:
            print(f"Reducing features from {len(exog_cols)} to 5 for stability")
            # Keep most correlated features
            correlations = {}
            for col in exog_cols:
                corr = data[col].corr(data[self.target_name])
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
            
            # Sort by correlation and keep top 5
            top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]
            exog_cols = [f[0] for f in top_features]
            
        # Create ARDL model with reduced complexity
        try:
            self.ardl_model = ARDL(
                data[self.target_name],
                lags=2,  # Reduced from 4
                exog=data[exog_cols],
                order=1  # Reduced from 2
            )
            
            # Fit model
            self.ardl_model = self.ardl_model.fit()
            print("   ✓ PPI ARDL model trained successfully")
            
        except Exception as e:
            print(f"   ARDL fitting failed: {e}")
            raise
        
    def _train_fallback(self, data: pd.DataFrame):
        """Simple fallback model if ARDL fails"""
        
        from sklearn.linear_model import Ridge  # More robust than LinearRegression
        
        if self.target_name not in data.columns:
            print("Warning: Target variable not found for fallback model")
            self.is_trained = False
            return
            
        y = data[self.target_name].values
        
        # Use simple lagged features
        X_list = []
        feature_names = []
        
        for col in data.columns:
            if col != self.target_name:
                # Create lagged feature
                lagged = data[col].shift(1).fillna(0).values
                
                # Check for valid values
                if not np.any(np.isnan(lagged)) and not np.any(np.isinf(lagged)):
                    X_list.append(lagged)
                    feature_names.append(col)
                    
        if not X_list:
            print("Warning: No valid features for fallback model")
            # Use a simple mean predictor
            self.fallback_model = lambda x: np.full(len(x) if hasattr(x, '__len__') else 1, 
                                                   np.nanmean(y))
            self.is_trained = True
            return
            
        X = np.column_stack(X_list)
        
        # Remove any remaining invalid values
        valid_idx = ~(np.isnan(y) | np.any(np.isnan(X), axis=1) | np.any(np.isinf(X), axis=1))
        
        if np.sum(valid_idx) < 10:
            print("Warning: Insufficient valid data for training")
            self.fallback_model = lambda x: np.full(len(x) if hasattr(x, '__len__') else 1, 
                                                   np.nanmean(y[valid_idx]) if np.any(valid_idx) else 2.5)
            self.is_trained = True
            return
            
        # Use Ridge regression for stability
        self.fallback_model = Ridge(alpha=1.0)
        self.fallback_model.fit(X[valid_idx], y[valid_idx])
        self.feature_names_fallback = feature_names
        self.is_trained = True
        print("   ✓ PPI fallback model trained")
        
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
            print("Warning: Model not trained, using default forecast")
            # Return default forecasts
            forecasts = {}
            for horizon in horizons:
                forecasts[f'{horizon}m'] = {
                    'forecast': 2.5,  # Default PPI assumption
                    'lower': 1.5,
                    'upper': 3.5,
                    'turning_point_signal': 'uncertain'
                }
            return forecasts
            
        forecasts = {}
        
        for horizon in horizons:
            if self.ardl_model is not None:
                try:
                    # ARDL forecast
                    forecast_result = self.ardl_model.forecast(steps=horizon)
                    point_forecast = float(forecast_result.iloc[-1])
                    
                except Exception as e:
                    print(f"ARDL forecast failed: {e}")
                    # Fallback to simple forecast
                    point_forecast = self._simple_forecast(data, horizon)
            else:
                point_forecast = self._simple_forecast(data, horizon)
                
            # Ensure forecast is reasonable
            point_forecast = np.clip(point_forecast, -10, 20)  # PPI typically between -10% and 20%
            
            # Detect potential turning points
            turning_point = self._detect_turning_point(data, point_forecast)
            
            # Calculate confidence intervals
            lower, upper = self.calculate_confidence_intervals(point_forecast, horizon)
            
            forecasts[f'{horizon}m'] = {
                'forecast': point_forecast,
                'lower': max(lower, -10),
                'upper': min(upper, 20),
                'turning_point_signal': turning_point
            }
            
        return forecasts
        
    def _simple_forecast(self, data: pd.DataFrame, horizon: int) -> float:
        """Simple forecast based on recent trends"""
        
        if self.target_name in data.columns:
            recent_values = data[self.target_name].dropna().iloc[-6:]
            
            if len(recent_values) > 0:
                # Check for valid values
                recent_values = recent_values.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(recent_values) > 1:
                    # Simple trend extrapolation
                    x = np.arange(len(recent_values))
                    y = recent_values.values
                    
                    # Check for valid regression
                    if not np.any(np.isnan(y)) and not np.any(np.isinf(y)):
                        try:
                            trend = np.polyfit(x, y, 1)[0]
                            last_value = y[-1]
                            forecast = last_value + trend * horizon
                            return np.clip(forecast, -10, 20)
                        except:
                            pass
                            
                # If trend fails, use mean
                if len(recent_values) > 0:
                    return float(np.mean(recent_values))
                    
        # Default PPI assumption
        return 2.5
        
    def _detect_turning_point(self, data: pd.DataFrame, forecast: float) -> str:
        """Detect potential inflation turning points"""
        
        if self.target_name not in data.columns:
            return "uncertain"
            
        recent_data = data[self.target_name].dropna().iloc[-6:]
        
        # Clean data
        recent_data = recent_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(recent_data) == 0:
            return "uncertain"
            
        recent = float(np.mean(recent_data))
        
        if forecast > recent + 0.5:
            return "acceleration"
        elif forecast < recent - 0.5:
            return "deceleration"
        else:
            return "stable"