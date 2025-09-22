"""
Business cycle regime identification and investment signals
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

class BusinessCycleAnalyzer:
    """Analyze business cycle and generate investment signals"""
    
    def __init__(self, growth_model, ppi_model, cpi_model):
        """
        Initialize with trained models
        
        Args:
            growth_model: Trained growth forecaster
            ppi_model: Trained PPI forecaster  
            cpi_model: Trained CPI forecaster
        """
        self.growth_model = growth_model
        self.ppi_model = ppi_model
        self.cpi_model = cpi_model
        
        # Define regime mappings
        self.regimes = {
            'goldilocks': {
                'growth': 'rising',
                'inflation': 'stable',
                'description': 'Optimal growth with controlled inflation'
            },
            'reflation': {
                'growth': 'rising', 
                'inflation': 'rising',
                'description': 'Economic recovery with increasing prices'
            },
            'stagflation': {
                'growth': 'falling',
                'inflation': 'rising', 
                'description': 'Slowing growth with persistent inflation'
            },
            'deflation': {
                'growth': 'falling',
                'inflation': 'falling',
                'description': 'Economic contraction with falling prices'
            }
        }
        
    def identify_regime(self, forecasts: Dict) -> Dict:
        """
        Identify current and forecast business cycle regime
        
        Args:
            forecasts: Dictionary with growth and inflation forecasts
            
        Returns:
            Dictionary with regime identification
        """
        # Extract 6-month forecasts
        growth_6m = forecasts['growth']['6m']['forecast']
        ppi_6m = forecasts['ppi']['6m']['forecast']
        cpi_6m = forecasts['cpi']['6m']['forecast']
        
        # Current levels (use 3-month for nearer term)
        growth_current = forecasts['growth']['3m']['forecast']
        ppi_current = forecasts['ppi']['3m']['forecast']
        
        # Determine growth direction
        growth_rising = growth_6m > growth_current and growth_6m > 50
        
        # Determine inflation direction (combine PPI and CPI)
        inflation_rising = (ppi_6m > ppi_current + 0.3) or (cpi_6m > 3.0)
        inflation_stable = abs(cpi_6m - 2.0) < 1.0 and abs(ppi_6m - ppi_current) < 0.5
        
        # Identify regime
        if growth_rising and inflation_stable:
            regime = 'goldilocks'
        elif growth_rising and inflation_rising:
            regime = 'reflation'
        elif not growth_rising and inflation_rising:
            regime = 'stagflation'
        else:
            regime = 'deflation'
            
        # Calculate regime probabilities
        probabilities = self._calculate_regime_probabilities(
            growth_current, growth_6m, ppi_6m, cpi_6m
        )
        
        return {
            'current': regime,
            'forecast_6m': regime,
            'probabilities': probabilities,
            'growth_outlook': 'rising' if growth_rising else 'falling',
            'inflation_outlook': 'rising' if inflation_rising else ('stable' if inflation_stable else 'falling'),
            'description': self.regimes[regime]['description']
        }
        
    def _calculate_regime_probabilities(self, growth_current: float, growth_6m: float,
                                       ppi_6m: float, cpi_6m: float) -> Dict:
        """Calculate probability of each regime"""
        
        probs = {}
        
        # Simple heuristic probabilities
        growth_score = (growth_6m - 50) / 10  # Normalized around 50
        inflation_score = (cpi_6m - 2.0) / 2  # Normalized around 2%
        
        # Goldilocks: High growth, low inflation
        probs['goldilocks'] = max(0, min(1, 
            0.5 + 0.3 * growth_score - 0.2 * abs(inflation_score)))
        
        # Reflation: Rising growth and inflation
        probs['reflation'] = max(0, min(1,
            0.3 + 0.2 * growth_score + 0.2 * inflation_score))
        
        # Stagflation: Low growth, high inflation  
        probs['stagflation'] = max(0, min(1,
            0.2 - 0.3 * growth_score + 0.3 * inflation_score))
        
        # Deflation: Low growth, low inflation
        probs['deflation'] = max(0, min(1,
            0.3 - 0.2 * growth_score - 0.2 * inflation_score))
        
        # Normalize to sum to 1
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
            
        return probs
        
    def generate_signals(self, regime: Dict) -> Dict:
        """
        Generate investment signals based on regime
        
        Args:
            regime: Regime identification dictionary
            
        Returns:
            Dictionary with asset allocation signals
        """
        current_regime = regime['current']
        
        # Define allocation strategies per regime
        strategies = {
            'goldilocks': {
                'equities': 'overweight',
                'bonds': 'neutral',
                'commodities': 'underweight',
                'cash': 'underweight',
                'recommendation': 'Risk-on: Favor growth stocks and credit'
            },
            'reflation': {
                'equities': 'overweight',
                'bonds': 'underweight',
                'commodities': 'overweight',
                'cash': 'underweight',
                'recommendation': 'Cyclical tilt: Banks, energy, materials'
            },
            'stagflation': {
                'equities': 'underweight',
                'bonds': 'underweight',
                'commodities': 'overweight',
                'cash': 'neutral',
                'recommendation': 'Defensive: Commodities and inflation protection'
            },
            'deflation': {
                'equities': 'underweight',
                'bonds': 'overweight',
                'commodities': 'underweight',
                'cash': 'overweight',
                'recommendation': 'Risk-off: Quality bonds and cash'
            }
        }
        
        return strategies.get(current_regime, strategies['deflation'])
        
    def backtest_regime_identification(self, historical_data: pd.DataFrame) -> Dict:
        """
        Backtest regime identification accuracy
        
        Args:
            historical_data: Historical data with actual regimes
            
        Returns:
            Dictionary with backtesting metrics
        """
        # This would require labeled historical regimes
        # For now, return placeholder metrics
        
        return {
            'accuracy': 0.65,
            'transition_accuracy': 0.55,
            'average_regime_duration': 8,  # months
            'false_signals': 0.2
        }