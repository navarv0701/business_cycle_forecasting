"""
Model parameters and configuration for forecasting models
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class ModelConfig:
    """Configuration for all forecasting models"""
    
    # Data parameters
    train_start: str = '2010-01-01'
    train_end: str = '2019-12-31'
    test_start: str = '2020-01-01'
    
    # Forecast horizons (in months)
    forecast_horizons: List[int] = None
    
    # Model parameters
    var_max_lags: int = 4
    var_ic: str = 'aic'  # Information criterion for lag selection
    
    ardl_max_lags: int = 6
    ardl_max_order: int = 4
    
    # Feature engineering
    ma_windows: List[int] = None  # Moving average windows
    momentum_windows: List[int] = None  # Momentum calculation windows
    
    # Business cycle regimes
    regimes: Dict[str, Tuple[str, str]] = None
    
    # Backtesting
    backtest_window: int = 24  # months
    backtest_step: int = 1  # month
    
    # Confidence intervals
    confidence_level: float = 0.95
    bootstrap_iterations: int = 1000
    
    def __post_init__(self):
        """Initialize default values"""
        if self.forecast_horizons is None:
            self.forecast_horizons = [3, 6, 9]
            
        if self.ma_windows is None:
            self.ma_windows = [3, 6, 12]
            
        if self.momentum_windows is None:
            self.momentum_windows = [1, 3, 6]
            
        if self.regimes is None:
            self.regimes = {
                'goldilocks': ('growth_up', 'inflation_stable'),
                'reflation': ('growth_up', 'inflation_up'),
                'stagflation': ('growth_down', 'inflation_up'),
                'deflation': ('growth_down', 'inflation_down')
            }
    
    @property
    def growth_features(self) -> List[str]:
        """Features for growth model"""
        return [
            'yield_spread',
            'housing_permits_yoy',
            'consumer_confidence_ma3',
            'core_capital_goods_3m',
            'claims_ma4w_inverted'
        ]
    
    @property
    def ppi_features(self) -> List[str]:
        """Features for PPI model"""
        return [
            'baltic_dry_ma3',
            'import_prices_3m',
            'employment_cost_yoy',
            'nfib_comp_plans_ma3',
            'commodity_pressure'
        ]
    
    @property
    def cpi_features(self) -> List[str]:
        """Features for CPI model"""
        return [
            'employment_cost_yoy',
            'nfib_comp_plans_ma6',
            'inflation_expect_5y5y',
            'wage_pressure_composite',
            'import_prices_ma6'
        ]
    
    @property
    def regime_thresholds(self) -> Dict[str, float]:
        """Thresholds for regime identification"""
        return {
            'growth_neutral': 50.0,  # ISM PMI
            'inflation_low': 2.0,    # CPI YoY
            'inflation_high': 3.0,   # CPI YoY
            'ppi_signal': 0.5        # PPI momentum
        }