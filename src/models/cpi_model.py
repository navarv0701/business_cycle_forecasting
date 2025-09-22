"""
CPI forecasting model for inflation confirmation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from statsmodels.tsa.ardl import ARDL
from src.models.base_model import BaseForecastModel

class CPIForecaster(BaseForecastModel):
    """Forecast CPI for inflation confirmation"""
    
    def __init__(self):
        """Initialize CPI forecaster"""
        super().__init__("CPI Model (Confirmation)")
        self.ardl_model = None
        self.target_name = 'CPI_YoY'
        self.wage_weight = 0.5  # Higher weight on wages for CPI
        self.expectation_weight = 0.3
        self.import_weight = 0.2  # Lower weight on imports
        
    def train(self, data: pd.DataFrame, target_col: str = 'CPI_YoY'):
        """
        Train ARDL model with emphasis on wage dynamics
        
        Args:
            data: DataFrame with inflation indicators
            target_col: Target variable name
        """
        self.target_name = target_col
        
        # Key features for CPI
        wage_features = ['Employment_Cost_Index', 'NFIB_Comp_Plans']
        expectation_features = ['Inflation_Expect_5Y5Y']
        import_features = ['Import_Price_Index']
        
        # Check available features
        all_features = wage_features + expectation_features + import_features
        self.feature_names = [f for f in all_features if any(f in col for col in data.columns)]
        
        # Prepare data for ARDL
        model_data = self._prepare_ardl_data(data)
        
        if len(model_data) < 50:
            print(f"Warning: Limited data for CPI model ({len(model_data)} observations)")
            
        # Train ARDL model
        try:
            self._train_ardl(model_data)
            self.is_trained = True
        except Exception as e:
            print(f"ARDL training failed: {e}")
            self._train_fallback(model_data)
            
    def _prepare_ardl_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for ARDL model"""
        
        # Select relevant columns with focus on wage dynamics
        cols_to_use = []
        
        # Target
        if self.target_name in data.columns:
            cols_to_use.append(self.target_name)
            
        # Prioritize wage-related columns
        wage_cols = [col for col in data.columns if 'Employment' in col or 'NFIB' in col or 'wage' in col]
        cols_to_use.extend(wage_cols[:3])
        
        # Add expectations
        expect_cols = [col for col in data.columns if 'Expect' in col or '5Y5Y' in col]
        cols_to_use.extend(expect_cols[:2])
        
        # Add limited import prices (less important for CPI)
        import_cols = [col for col in data.columns if 'Import' in col]
        cols_to_use.extend(import_cols[:1])
        
        # Remove duplicates and clean
        cols_to_use = list(set(cols_to_use))
        model_data = data[cols_to_use].dropna()
        
        return model_data
        
    def _train_ardl(self, data: pd.DataFrame):
        """Train ARDL model for CPI"""
        
        if self.target_name not in data.columns:
            raise ValueError(f"Target {self.target_name} not in data")
            
        exog_cols = [col for col in data.columns if col != self.target_name]
        
        if len(exog_cols) == 0:
            raise ValueError("No exogenous variables available")
            
        # ARDL with more lags for CPI (stickier)
        self.ardl_model = ARDL(
            data[self.target_name],
            lags=6,  # More CPI lags (persistence)
            exog=data[exog_cols],
            order=3  # Wage lags (3-6 month transmission)
        )
        
        # Fit model
        self.ardl_model = self.ardl_model.fit()
        
    def _train_fallback(self, data: pd.DataFrame):
        """Simple fallback model if ARDL fails"""
        
        from sklearn.linear_model import LinearRegression
        
        if self.target_name in data.columns:
            y = data[self.target_name].values
            
            # Focus on wage dynamics for fallback
            wage_cols = [col for col in data.columns if 'Employment' in col or 'NFIB' in col or 'wage' in col]
            
            if wage_cols:
                X = data[wage_cols].shift(3).fillna(method='bfill').values  # 3-month lag
                valid_idx = ~np.isnan(y)
                
                self.fallback_model = LinearRegression()
                self.fallback_model.fit(X[valid_idx], y[valid_idx])
                self.is_trained = True
                
    def forecast(self, data: pd.DataFrame, horizons: List[int] = [6, 9]) -> Dict:
        """
        Generate CPI forecasts with regime considerations
        
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
                
            # Apply regime adjustments
            point_forecast = self._apply_regime_adjustment(data, point_forecast)
            
            # Calculate confidence intervals (wider for CPI due to persistence)
            lower, upper = self.calculate_confidence_intervals(point_forecast, horizon, confidence=0.95)
            
            forecasts[f'{horizon}m'] = {
                'forecast': point_forecast,
                'lower': lower,
                'upper': upper,
                'regime': self._identify_inflation_regime(point_forecast)
            }
            
        return forecasts
        
    def _simple_forecast(self, data: pd.DataFrame, horizon: int) -> float:
        """Simple forecast based on wage trends and expectations"""
        
        if self.target_name in data.columns:
            recent_cpi = data[self.target_name].iloc[-6:].mean()
            
            # Check wage pressure
            wage_cols = [col for col in data.columns if 'Employment' in col or 'wage' in col]
            if wage_cols:
                wage_trend = data[wage_cols].iloc[-6:].mean().mean()
                wage_adjustment = (wage_trend - 3.0) * 0.3  # Wage pass-through
            else:
                wage_adjustment = 0
                
            # Check expectations
            expect_cols = [col for col in data.columns if 'Expect' in col or '5Y5Y' in col]
            if expect_cols:
                expectations = data[expect_cols].iloc[-1].mean()
                expect_adjustment = (expectations - 2.0) * 0.2  # Expectation anchoring
            else:
                expect_adjustment = 0
                
            return recent_cpi + wage_adjustment + expect_adjustment
        else:
            return 2.0  # Default CPI target
            
    def _apply_regime_adjustment(self, data: pd.DataFrame, forecast: float) -> float:
        """Adjust forecast based on inflation regime"""
        
        # CPI is stickier, apply mean reversion but slowly
        if forecast > 4.0:
            # High inflation regime - expect some policy response
            forecast *= 0.95
        elif forecast < 1.0:
            # Deflation risk - expect policy support
            forecast = max(forecast * 1.1, 1.0)
            
        return forecast
        
    def _identify_inflation_regime(self, forecast: float) -> str:
        """Identify inflation regime"""
        
        if forecast < 0:
            return "deflation"
        elif forecast < 2.0:
            return "below_target"
        elif forecast <= 3.0:
            return "on_target"
        elif forecast <= 4.0:
            return "above_target"
        else:
            return "unanchored"