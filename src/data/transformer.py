"""
Data transformation and feature engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

class DataTransformer:
    """Transform raw data into model features"""
    
    def __init__(self):
        """Initialize transformer"""
        self.scalers = {}
        
    def transform_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformations to raw data
        
        Args:
            data: Raw data DataFrame
            
        Returns:
            Transformed DataFrame with engineered features
        """
        df = data.copy()
        
        # Handle missing data
        df = self._handle_missing_data(df)
        
        # Create basic transformations
        df = self._create_basic_features(df)
        
        # Create model-specific features
        df = self._create_growth_features(df)
        df = self._create_inflation_features(df)
        
        # Create interaction features
        df = self._create_interaction_features(df)
        
        return df
        
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data appropriately"""
        
        # Forward fill for most series (prices, indices)
        price_cols = [col for col in df.columns if 'Yield' in col or 'Index' in col or 'Price' in col]
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].ffill()
                
        # Interpolate for others
        df = df.interpolate(method='linear', limit=5)
        
        # Drop rows with too many missing values
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold)
        
        return df
        
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features"""
        
        # Yield spread
        if 'US_10Y_Yield' in df.columns and 'US_2Y_Yield' in df.columns:
            df['yield_spread'] = df['US_10Y_Yield'] - df['US_2Y_Yield']
            
        # Year-over-year changes
        for col in df.columns:
            if col not in ['yield_spread']:
                # YoY change
                df[f'{col}_yoy'] = df[col].pct_change(252) * 100
                
                # 3-month change
                df[f'{col}_3m'] = df[col].pct_change(63) * 100
                
        # Moving averages
        windows = [20, 60, 120]  # Approximately 1m, 3m, 6m
        for col in df.columns:
            if '_yoy' not in col and '_3m' not in col:
                for window in windows:
                    df[f'{col}_ma{window}'] = df[col].rolling(window=window).mean()
                    
        # Volatility
        for col in ['US_10Y_Yield', 'Baltic_Dry_Index', 'ISM_PMI']:
            if col in df.columns:
                df[f'{col}_vol'] = df[col].rolling(window=20).std()
                
        return df
        
    def _create_growth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for growth model"""
        
        # Housing momentum
        if 'Housing_Permits' in df.columns:
            df['housing_momentum'] = df['Housing_Permits'].pct_change(3) * 100
            
        # Consumer confidence trend
        if 'Consumer_Confidence' in df.columns:
            df['confidence_trend'] = (
                df['Consumer_Confidence'].rolling(3).mean() -
                df['Consumer_Confidence'].rolling(12).mean()
            )
            
        # Claims inverted (lower claims = better)
        if 'Unemployment_Claims_4WK' in df.columns:
            df['claims_inverted'] = -df['Unemployment_Claims_4WK']
            
        # Capital goods momentum
        if 'Core_Capital_Goods' in df.columns:
            df['capex_momentum'] = df['Core_Capital_Goods'].rolling(3).mean().pct_change(3) * 100
            
        return df
        
    def _create_inflation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for inflation models"""
        
        # Wage pressure composite
        wage_cols = []
        if 'Employment_Cost_Index' in df.columns:
            wage_cols.append('Employment_Cost_Index')
        if 'NFIB_Comp_Plans' in df.columns:
            wage_cols.append('NFIB_Comp_Plans')
            
        if len(wage_cols) > 0:
            df['wage_pressure'] = df[wage_cols].mean(axis=1)
            
        # Supply chain pressure
        supply_cols = []
        if 'Baltic_Dry_Index' in df.columns:
            supply_cols.append('Baltic_Dry_Index')
        if 'Import_Price_Index' in df.columns:
            supply_cols.append('Import_Price_Index')
            
        if len(supply_cols) > 0:
            # Normalize and combine
            for col in supply_cols:
                df[f'{col}_norm'] = (df[col] - df[col].mean()) / df[col].std()
            df['supply_pressure'] = df[[f'{col}_norm' for col in supply_cols]].mean(axis=1)
            
        # Inflation expectations momentum
        if 'Inflation_Expect_5Y5Y' in df.columns:
            df['expect_momentum'] = df['Inflation_Expect_5Y5Y'].diff(20)
            
        return df
        
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        
        # Yield curve * confidence
        if 'yield_spread' in df.columns and 'Consumer_Confidence' in df.columns:
            df['yield_confidence_interaction'] = (
                df['yield_spread'] * df['Consumer_Confidence'] / 100
            )
            
        # Wage * expectations
        if 'wage_pressure' in df.columns and 'Inflation_Expect_5Y5Y' in df.columns:
            df['wage_expect_interaction'] = (
                df['wage_pressure'] * df['Inflation_Expect_5Y5Y']
            )
            
        return df
        
    def get_growth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for growth model"""
        
        feature_cols = [
            'yield_spread', 'Housing_Permits_yoy', 'Consumer_Confidence_ma60',
            'confidence_trend', 'claims_inverted', 'capex_momentum',
            'Core_Capital_Goods_3m', 'yield_confidence_interaction'
        ]
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if 'ISM_PMI' in df.columns:
            return df[available_cols + ['ISM_PMI']].dropna()
        else:
            return df[available_cols].dropna()
            
    def get_ppi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for PPI model"""
        
        feature_cols = [
            'Baltic_Dry_Index_ma20', 'Import_Price_Index_3m',
            'supply_pressure', 'Employment_Cost_Index_yoy',
            'NFIB_Comp_Plans_ma60', 'Baltic_Dry_Index_vol'
        ]
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if 'PPI_YoY' in df.columns:
            return df[available_cols + ['PPI_YoY']].dropna()
        else:
            return df[available_cols].dropna()
            
    def get_cpi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for CPI model"""
        
        feature_cols = [
            'Employment_Cost_Index_yoy', 'wage_pressure',
            'NFIB_Comp_Plans_ma120', 'Inflation_Expect_5Y5Y',
            'expect_momentum', 'wage_expect_interaction',
            'Import_Price_Index_ma120'
        ]
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if 'CPI_YoY' in df.columns:
            return df[available_cols + ['CPI_YoY']].dropna()
        else:
            return df[available_cols].dropna()