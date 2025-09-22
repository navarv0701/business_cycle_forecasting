"""
Generate sample data for development and testing
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import json

import pandas as pd
import numpy as np

class SampleDataGenerator:
    """Generate realistic sample data for model development"""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed"""
        np.random.seed(seed)
        self.start_date = pd.Timestamp('2010-01-01')
        self.end_date = pd.Timestamp.now()
        self.dates = pd.date_range(self.start_date, self.end_date, freq='D')
        
    def generate_all_data(self, output_dir: str = 'data/sample_data'):
        """Generate all sample data files"""
        
        print("Generating sample data for development...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate each series
        data = {}
        
        # Growth indicators
        data['US_10Y_Yield'] = self._generate_yield_curve(base=3.5, vol=0.5)
        data['US_2Y_Yield'] = self._generate_yield_curve(base=2.5, vol=0.3)
        data['Housing_Permits'] = self._generate_housing(base=1200, vol=100)
        data['Consumer_Confidence'] = self._generate_confidence(base=100, vol=10)
        data['Core_Capital_Goods'] = self._generate_capital_goods(base=70000, vol=2000)
        data['Core_Capital_Goods_MoM'] = self._generate_mom_series(base=0.3, vol=0.5)
        data['Unemployment_Claims_4WK'] = self._generate_claims(base=350000, vol=50000)
        
        # Inflation indicators
        data['ISM_PMI'] = self._generate_pmi(base=52, vol=3)
        data['Employment_Cost_Index'] = self._generate_eci(base=3.0, vol=0.5)
        data['Inflation_Expect_5Y5Y'] = self._generate_expectations(base=2.5, vol=0.2)
        data['Import_Price_Index'] = self._generate_import_prices(base=2.0, vol=1.5)
        data['Baltic_Dry_Index'] = self._generate_baltic_dry(base=1500, vol=500)
        data['NFIB_Comp_Plans'] = self._generate_nfib(base=20, vol=5)
        
        # Target variables
        data['PPI_YoY'] = self._generate_ppi(base=2.5, vol=1.0)
        data['CPI_YoY'] = self._generate_cpi(base=2.3, vol=0.8)
        
        # Combine all data
        df = pd.DataFrame(data, index=self.dates)
        
        # Add business cycle patterns
        df = self._add_business_cycles(df)
        
        # Save files
        master_csv = os.path.join(output_dir, 'MASTER_DATA.csv')
        master_parquet = os.path.join(output_dir, 'MASTER_DATA.parquet')
        
        df.to_csv(master_csv)
        df.to_parquet(master_parquet)
        
        # Save individual files
        for col in df.columns:
            col_df = df[[col]]
            col_df.to_csv(os.path.join(output_dir, f'{col}.csv'))
            
        # Create metadata
        metadata = {
            'extraction_date': datetime.now().isoformat(),
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'source': 'sample_generator',
            'tickers': {col: {'rows': len(df), 'name': col} for col in df.columns}
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"✓ Sample data generated: {df.shape}")
        print(f"  Files saved to: {output_dir}")
        
        return df
        
    def _generate_base_series(self, base: float, vol: float, trend: float = 0) -> pd.Series:
        """Generate a base time series with random walk"""
        
        returns = np.random.normal(trend/252, vol/np.sqrt(252), len(self.dates))
        prices = base * np.exp(np.cumsum(returns))
        
        return pd.Series(prices, index=self.dates)
        
    def _generate_yield_curve(self, base: float, vol: float) -> pd.Series:
        """Generate yield curve data"""
        
        # Mean reverting process
        series = np.zeros(len(self.dates))
        series[0] = base
        
        mean_reversion = 0.01
        
        for i in range(1, len(self.dates)):
            shock = np.random.normal(0, vol/np.sqrt(252))
            series[i] = series[i-1] + mean_reversion * (base - series[i-1]) + shock
            
        return pd.Series(series, index=self.dates)
        
    def _generate_housing(self, base: float, vol: float) -> pd.Series:
        """Generate housing permits data"""
        # Cyclical pattern with trend
        trend = np.linspace(0, 200, len(self.dates))
        cycle = 100 * np.sin(2 * np.pi * np.arange(len(self.dates)) / 1260)  # 5-year cycle
        noise = np.random.normal(0, vol, len(self.dates))
        
        series = base + trend + cycle + noise
        return pd.Series(series, index=self.dates)
        
    def _generate_confidence(self, base: float, vol: float) -> pd.Series:
        """Generate consumer confidence"""
        # More volatile, mean-reverting
        series = self._generate_yield_curve(base, vol * 2)
        # Fixed clip syntax for newer numpy
        return pd.Series(np.clip(series.values, 50, 150), index=self.dates)
        
    def _generate_capital_goods(self, base: float, vol: float) -> pd.Series:
        """Generate capital goods orders"""
        # Trending with business cycle
        trend = np.linspace(0, 5000, len(self.dates))
        cycle = 2000 * np.sin(2 * np.pi * np.arange(len(self.dates)) / 1008)  # 4-year cycle
        noise = np.random.normal(0, vol, len(self.dates))
        
        series = base + trend + cycle + noise
        return pd.Series(series, index=self.dates)
        
    def _generate_mom_series(self, base: float, vol: float) -> pd.Series:
        """Generate month-over-month series"""
        return pd.Series(np.random.normal(base, vol, len(self.dates)), index=self.dates)
        
    def _generate_claims(self, base: float, vol: float) -> pd.Series:
        """Generate unemployment claims"""
        # Counter-cyclical
        trend = -np.linspace(0, 50000, len(self.dates))
        cycle = 50000 * np.sin(2 * np.pi * np.arange(len(self.dates)) / 1260 + np.pi)  # Inverted cycle
        noise = np.random.normal(0, vol, len(self.dates))
        
        series = base + trend + cycle + noise
        # Fixed clip syntax
        series = np.maximum(series, 200000)  # Equivalent to clip(lower=200000)
        return pd.Series(series, index=self.dates)
        
    def _generate_pmi(self, base: float, vol: float) -> pd.Series:
        """Generate ISM PMI data"""
        # Business cycle indicator
        cycle = 5 * np.sin(2 * np.pi * np.arange(len(self.dates)) / 1008)  # 4-year cycle
        noise = np.random.normal(0, vol, len(self.dates))
        
        series = base + cycle + noise
        # Fixed clip syntax
        return pd.Series(np.clip(series, 40, 65), index=self.dates)
        
    def _generate_eci(self, base: float, vol: float) -> pd.Series:
        """Generate employment cost index"""
        # Slow moving with trend
        trend = np.linspace(0, 1, len(self.dates))
        noise = np.cumsum(np.random.normal(0, vol/100, len(self.dates)))
        
        series = base + trend + noise
        return pd.Series(series, index=self.dates)
        
    def _generate_expectations(self, base: float, vol: float) -> pd.Series:
        """Generate inflation expectations"""
        # Anchored around target with occasional drift
        series = self._generate_yield_curve(base, vol)
        # Fixed clip syntax
        return pd.Series(np.clip(series.values, 1.5, 3.5), index=self.dates)
        
    def _generate_import_prices(self, base: float, vol: float) -> pd.Series:
        """Generate import price index"""
        # More volatile, commodity-linked
        commodity_cycle = 2 * np.sin(2 * np.pi * np.arange(len(self.dates)) / 504)  # 2-year cycle
        noise = np.random.normal(0, vol, len(self.dates))
        
        series = base + commodity_cycle + noise
        return pd.Series(series, index=self.dates)
        
    def _generate_baltic_dry(self, base: float, vol: float) -> pd.Series:
        """Generate Baltic Dry Index"""
        # Very volatile, supply-demand driven
        series = np.zeros(len(self.dates))
        series[0] = base
        
        for i in range(1, len(self.dates)):
            shock = np.random.normal(0, vol/np.sqrt(252))
            # Random jumps
            if np.random.random() < 0.01:  # 1% chance of large move
                shock *= 5
            series[i] = series[i-1] * np.exp(shock/base)
            
        # Fixed clip syntax
        return pd.Series(np.clip(series, 500, 5000), index=self.dates)
        
    def _generate_nfib(self, base: float, vol: float) -> pd.Series:
        """Generate NFIB compensation plans"""
        # Cyclical with labor market
        cycle = 5 * np.sin(2 * np.pi * np.arange(len(self.dates)) / 756)  # 3-year cycle
        noise = np.random.normal(0, vol, len(self.dates))
        
        series = base + cycle + noise
        # Fixed clip syntax
        return pd.Series(np.clip(series, 10, 35), index=self.dates)
        
    def _generate_ppi(self, base: float, vol: float) -> pd.Series:
        """Generate PPI YoY"""
        # Leading inflation indicator
        trend = 0.5 * np.sin(2 * np.pi * np.arange(len(self.dates)) / 2016)  # 8-year cycle
        commodity = 0.5 * np.sin(2 * np.pi * np.arange(len(self.dates)) / 504)  # 2-year cycle
        noise = np.random.normal(0, vol, len(self.dates))
        
        series = base + trend + commodity + noise
        return pd.Series(series, index=self.dates)
        
    def _generate_cpi(self, base: float, vol: float) -> pd.Series:
        """Generate CPI YoY"""
        # Lagging, stickier than PPI
        trend = 0.3 * np.sin(2 * np.pi * np.arange(len(self.dates)) / 2016 - np.pi/4)  # Lag PPI
        noise = np.cumsum(np.random.normal(0, vol/100, len(self.dates)))
        
        series = base + trend + noise
        return pd.Series(series, index=self.dates)
        
    def _add_business_cycles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic business cycle correlations"""
        
        # Define cycle periods (approximate)
        recession_periods = [
            ('2010-01-01', '2010-06-30'),  # End of Great Recession
            ('2020-03-01', '2020-06-30'),  # COVID
        ]
        
        boom_periods = [
            ('2013-01-01', '2014-12-31'),
            ('2017-01-01', '2018-12-31'),
        ]
        
        # Adjust data during recessions
        for start, end in recession_periods:
            mask = (df.index >= start) & (df.index <= end)
            df.loc[mask, 'ISM_PMI'] *= 0.85
            df.loc[mask, 'Consumer_Confidence'] *= 0.8
            df.loc[mask, 'Unemployment_Claims_4WK'] *= 1.5
            df.loc[mask, 'Baltic_Dry_Index'] *= 0.7
            
        # Adjust data during booms
        for start, end in boom_periods:
            mask = (df.index >= start) & (df.index <= end)
            df.loc[mask, 'ISM_PMI'] *= 1.1
            df.loc[mask, 'Consumer_Confidence'] *= 1.15
            df.loc[mask, 'Core_Capital_Goods'] *= 1.2
            df.loc[mask, 'NFIB_Comp_Plans'] *= 1.2
            
        # Add correlations
        # Yield spread leads recessions
        df['yield_spread_calc'] = df['US_10Y_Yield'] - df['US_2Y_Yield']
        
        # When yield spread is negative, increase recession probability
        inverted_mask = df['yield_spread_calc'] < 0
        df.loc[inverted_mask, 'ISM_PMI'] *= 0.95
        
        return df