"""
Visualization utilities for forecasting system
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Optional

sns.set_style('whitegrid')

class Visualizer:
    """Create visualizations for forecasting results"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.colors = {
            'actual': '#2E86AB',
            'forecast': '#A23B72',
            'confidence': '#F18F01',
            'goldilocks': '#52B788',
            'reflation': '#F77F00',
            'stagflation': '#D62828',
            'deflation': '#003049'
        }
        
    def plot_time_series_forecast(self, historical_data: pd.DataFrame,
                                 forecasts: Dict,
                                 save_path: Optional[str] = None):
        """
        Plot historical data with forecasts
        
        Args:
            historical_data: Historical data
            forecasts: Forecast dictionary
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Growth (ISM PMI)
        ax = axes[0]
        if 'ISM_PMI' in historical_data.columns:
            ax.plot(historical_data.index[-252:], 
                   historical_data['ISM_PMI'].iloc[-252:],
                   color=self.colors['actual'], label='Historical')
                   
        # Add forecast points
        last_date = historical_data.index[-1]
        forecast_dates = pd.date_range(last_date, periods=10, freq='M')[1:]
        
        if 'growth' in forecasts:
            growth_f = forecasts['growth']
            forecast_values = [
                growth_f.get('3m', {}).get('forecast'),
                growth_f.get('6m', {}).get('forecast'),
                growth_f.get('9m', {}).get('forecast')
            ]
            ax.plot([forecast_dates[2], forecast_dates[5], forecast_dates[8]],
                   forecast_values, 'o--',
                   color=self.colors['forecast'], label='Forecast')
                   
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Growth Forecast (ISM PMI)')
        ax.set_ylabel('ISM PMI')
        ax.legend()
        
        # PPI
        ax = axes[1]
        if 'PPI_YoY' in historical_data.columns:
            ax.plot(historical_data.index[-252:],
                   historical_data['PPI_YoY'].iloc[-252:],
                   color=self.colors['actual'], label='Historical')
                   
        if 'ppi' in forecasts:
            ppi_f = forecasts['ppi']
            forecast_values = [
                ppi_f.get('3m', {}).get('forecast'),
                ppi_f.get('6m', {}).get('forecast')
            ]
            ax.plot([forecast_dates[2], forecast_dates[5]],
                   forecast_values, 'o--',
                   color=self.colors['forecast'], label='Forecast')
                   
        ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('PPI Forecast (Early Warning)')
        ax.set_ylabel('PPI YoY %')
        ax.legend()
        
        # CPI
        ax = axes[2]
        if 'CPI_YoY' in historical_data.columns:
            ax.plot(historical_data.index[-252:],
                   historical_data['CPI_YoY'].iloc[-252:],
                   color=self.colors['actual'], label='Historical')
                   
        if 'cpi' in forecasts:
            cpi_f = forecasts['cpi']
            forecast_values = [
                cpi_f.get('6m', {}).get('forecast'),
                cpi_f.get('9m', {}).get('forecast')
            ]
            ax.plot([forecast_dates[5], forecast_dates[8]],
                   forecast_values, 'o--',
                   color=self.colors['forecast'], label='Forecast')
                   
        ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='Target')
        ax.set_title('CPI Forecast (Confirmation)')
        ax.set_ylabel('CPI YoY %')
        ax.set_xlabel('Date')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            
        return fig
        
    def plot_regime_analysis(self, regime: Dict, save_path: Optional[str] = None):
        """
        Plot regime analysis
        
        Args:
            regime: Regime dictionary
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Regime probabilities
        ax = axes[0]
        if 'probabilities' in regime:
            probs = regime['probabilities']
            bars = ax.bar(probs.keys(), probs.values())
            
            # Color bars by regime
            for bar, (name, _) in zip(bars, probs.items()):
                bar.set_color(self.colors.get(name, 'gray'))
                
            ax.set_title('Regime Probabilities')
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1)
            
        # Asset allocation
        ax = axes[1]
        current_regime = regime.get('current', 'unknown')
        
        # Mock allocation data (would come from signals)
        allocations = {
            'Equities': 0.4,
            'Bonds': 0.3,
            'Commodities': 0.2,
            'Cash': 0.1
        }
        
        wedges, texts, autotexts = ax.pie(
            allocations.values(),
            labels=allocations.keys(),
            autopct='%1.0f%%',
            startangle=90
        )
        
        ax.set_title(f'Suggested Allocation ({current_regime.title()})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            
        return fig