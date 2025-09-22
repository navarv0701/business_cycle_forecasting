#!/usr/bin/env python3
"""
Business Cycle Forecasting System
Main execution script for growth and inflation forecasting
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import project modules
from src.config.model_params import ModelConfig
from src.data.loader import DataLoader
from src.data.transformer import DataTransformer
from src.data.sample_generator import SampleDataGenerator
from src.models.growth_model import GrowthForecaster
from src.models.ppi_model import PPIForecaster
from src.models.cpi_model import CPIForecaster
from src.models.business_cycle import BusinessCycleAnalyzer
from src.analysis.reports import ReportGenerator
from src.utils.visualization import Visualizer

warnings.filterwarnings('ignore')

class BusinessCycleForecastingSystem:
    """Main system for business cycle forecasting"""
    
    def __init__(self, mode='production', data_path=None):
        """
        Initialize the forecasting system
        
        Args:
            mode: 'bloomberg', 'development', or 'production'
            data_path: Path to data directory
        """
        self.mode = mode
        self.config = ModelConfig()
        
        # Set data path based on mode
        if data_path:
            self.data_path = Path(data_path)
        else:
            if mode == 'development':
                self.data_path = Path('data/sample_data')
            else:
                self.data_path = Path('data/bloomberg_cache')
                
        # Initialize components
        self.data_loader = None
        self.transformer = None
        self.growth_model = None
        self.ppi_model = None
        self.cpi_model = None
        self.business_cycle = None
        self.visualizer = Visualizer()
        self.report_generator = ReportGenerator()
        
        # Data containers
        self.raw_data = None
        self.transformed_data = None
        self.forecasts = {}
        
    def initialize(self):
        """Initialize all components"""
        print(f"\nInitializing Business Cycle Forecasting System")
        print(f"Mode: {self.mode}")
        print(f"Data path: {self.data_path}")
        print("-" * 60)
        
        # Check data availability
        if self.mode == 'development':
            self._ensure_sample_data()
        elif self.mode == 'production':
            self._check_bloomberg_data()
            
        # Initialize data loader
        self.data_loader = DataLoader(self.data_path)
        
        # Initialize transformer
        self.transformer = DataTransformer()
        
        # Initialize models
        self.growth_model = GrowthForecaster()
        self.ppi_model = PPIForecaster()
        self.cpi_model = CPIForecaster()
        
        print("✓ System initialized successfully")
        
    def _ensure_sample_data(self):
        """Create sample data if it doesn't exist"""
        sample_data_dir = Path('data/sample_data')
        
        if not sample_data_dir.exists() or not list(sample_data_dir.glob('*.csv')):
            print("Generating sample data for development...")
            generator = SampleDataGenerator()
            generator.generate_all_data(output_dir='data/sample_data')
            print("✓ Sample data generated")
            
    def _check_bloomberg_data(self):
        """Check if Bloomberg data exists and is recent"""
        if not self.data_path.exists():
            print(f"⚠ Warning: Bloomberg data directory not found at {self.data_path}")
            print("Please run extract_data.py on Bloomberg Terminal first")
            print("Or use --mode development to work with sample data")
            sys.exit(1)
            
        # Check metadata for freshness
        metadata_path = self.data_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            extract_date = datetime.fromisoformat(metadata['extraction_date'])
            days_old = (datetime.now() - extract_date).days
            
            if days_old > 30:
                print(f"⚠ Warning: Bloomberg data is {days_old} days old")
                print("Consider running fresh extraction for latest data")
            else:
                print(f"✓ Bloomberg data freshness: {days_old} days old")
                
    def load_data(self):
        """Load and prepare data"""
        print("\nLoading data...")
        
        # Load raw data
        self.raw_data = self.data_loader.load_all_data()
        
        if self.raw_data is None or self.raw_data.empty:
            print("✗ Failed to load data")
            return False
            
        print(f"✓ Loaded data: {self.raw_data.shape}")
        print(f"  Date range: {self.raw_data.index.min()} to {self.raw_data.index.max()}")
        
        # Transform data
        print("\nTransforming data...")
        self.transformed_data = self.transformer.transform_all(self.raw_data)
        print(f"✓ Transformed data: {self.transformed_data.shape}")
        
        # Check for required columns
        required_growth = ['US_10Y_Yield', 'US_2Y_Yield', 'Housing_Permits', 
                          'Consumer_Confidence', 'Unemployment_Claims_4WK']
        required_inflation = ['Baltic_Dry_Index', 'Import_Price_Index', 
                             'Employment_Cost_Index', 'NFIB_Comp_Plans',
                             'Inflation_Expect_5Y5Y']
        
        missing = []
        for col in required_growth + required_inflation:
            if col not in self.transformed_data.columns:
                missing.append(col)
                
        if missing:
            print(f"⚠ Warning: Missing columns: {missing}")
            
        return True
        
    def train_models(self):
        """Train all forecasting models"""
        print("\n" + "="*60)
        print("TRAINING FORECASTING MODELS")
        print("="*60)
        
        # Split data for training
        train_end = self.transformed_data.index[-252]  # Last year for testing
        train_data = self.transformed_data[:train_end]
        
        print(f"Training period: {train_data.index.min()} to {train_data.index.max()}")
        
        # Train growth model
        print("\n1. Training Growth Model (ISM PMI)...")
        growth_features = self.transformer.get_growth_features(train_data)
        self.growth_model.train(growth_features)
        print("   ✓ Growth model trained")
        
        # Train PPI model
        print("\n2. Training PPI Model (Early Warning)...")
        ppi_features = self.transformer.get_ppi_features(train_data)
        self.ppi_model.train(ppi_features)
        print("   ✓ PPI model trained")
        
        # Train CPI model
        print("\n3. Training CPI Model (Confirmation)...")
        cpi_features = self.transformer.get_cpi_features(train_data)
        self.cpi_model.train(cpi_features)
        print("   ✓ CPI model trained")
        
        # Initialize business cycle analyzer
        print("\n4. Initializing Business Cycle Analyzer...")
        self.business_cycle = BusinessCycleAnalyzer(
            self.growth_model,
            self.ppi_model,
            self.cpi_model
        )
        print("   ✓ Business cycle analyzer ready")
        
    def generate_forecasts(self):
        """Generate forecasts for all models"""
        print("\n" + "="*60)
        print("GENERATING FORECASTS")
        print("="*60)
        
        # Use latest data for forecasting
        latest_data = self.transformed_data.tail(252)  # Last year of data
        
        # Growth forecasts
        print("\n1. Growth Forecasts (ISM PMI):")
        growth_features = self.transformer.get_growth_features(latest_data)
        growth_forecast = self.growth_model.forecast(growth_features, horizons=[3, 6, 9])
        self.forecasts['growth'] = growth_forecast
        
        print(f"   3-month: {growth_forecast['3m']['forecast']:.1f}")
        print(f"   6-month: {growth_forecast['6m']['forecast']:.1f}")
        print(f"   9-month: {growth_forecast['9m']['forecast']:.1f}")
        
        # PPI forecasts
        print("\n2. Inflation Early Warning (PPI):")
        ppi_features = self.transformer.get_ppi_features(latest_data)
        ppi_forecast = self.ppi_model.forecast(ppi_features, horizons=[3, 6])
        self.forecasts['ppi'] = ppi_forecast
        
        print(f"   3-month: {ppi_forecast['3m']['forecast']:.2f}%")
        print(f"   6-month: {ppi_forecast['6m']['forecast']:.2f}%")
        
        # CPI forecasts
        print("\n3. Inflation Confirmation (CPI):")
        cpi_features = self.transformer.get_cpi_features(latest_data)
        cpi_forecast = self.cpi_model.forecast(cpi_features, horizons=[6, 9])
        self.forecasts['cpi'] = cpi_forecast
        
        print(f"   6-month: {cpi_forecast['6m']['forecast']:.2f}%")
        print(f"   9-month: {cpi_forecast['9m']['forecast']:.2f}%")
        
        # Business cycle analysis
        print("\n4. Business Cycle Regime:")
        regime = self.business_cycle.identify_regime(self.forecasts)
        self.forecasts['regime'] = regime
        
        print(f"   Current regime: {regime['current']}")
        print(f"   6-month outlook: {regime['forecast_6m']}")
        
        # Investment signals
        signals = self.business_cycle.generate_signals(regime)
        self.forecasts['signals'] = signals
        
        print("\n5. Investment Signals:")
        for asset, weight in signals.items():
            print(f"   {asset}: {weight}")
            
    def backtest(self):
        """Run backtesting analysis"""
        print("\n" + "="*60)
        print("BACKTESTING PERFORMANCE")
        print("="*60)
        
        # Prepare backtest data
        backtest_start = '2015-01-01'
        backtest_data = self.transformed_data[backtest_start:]
        
        print(f"Backtest period: {backtest_start} to {backtest_data.index.max()}")
        
        results = {}
        
        # Growth model backtest
        print("\n1. Growth Model Performance:")
        growth_features = self.transformer.get_growth_features(backtest_data)
        growth_metrics = self.growth_model.backtest(growth_features)
        results['growth'] = growth_metrics
        
        print(f"   Directional accuracy: {growth_metrics.get('directional_accuracy', 0):.1%}")
        print(f"   RMSE: {growth_metrics.get('rmse', 0):.2f}")
        
        # PPI model backtest
        print("\n2. PPI Model Performance:")
        ppi_features = self.transformer.get_ppi_features(backtest_data)
        ppi_metrics = self.ppi_model.backtest(ppi_features)
        results['ppi'] = ppi_metrics
        
        print(f"   Directional accuracy: {ppi_metrics.get('directional_accuracy', 0):.1%}")
        print(f"   Turning point accuracy: {ppi_metrics.get('turning_point_accuracy', 0):.1%}")
        
        # CPI model backtest
        print("\n3. CPI Model Performance:")
        cpi_features = self.transformer.get_cpi_features(backtest_data)
        cpi_metrics = self.cpi_model.backtest(cpi_features)
        results['cpi'] = cpi_metrics
        
        print(f"   Directional accuracy: {cpi_metrics.get('directional_accuracy', 0):.1%}")
        print(f"   RMSE: {cpi_metrics.get('rmse', 0):.2f}")
        
        # Regime identification accuracy
        print("\n4. Regime Identification:")
        regime_metrics = self.business_cycle.backtest_regime_identification(backtest_data)
        results['regime'] = regime_metrics
        
        print(f"   Regime accuracy: {regime_metrics.get('accuracy', 0):.1%}")
        print(f"   Transition detection: {regime_metrics.get('transition_accuracy', 0):.1%}")
        
        return results
        
    def generate_report(self, output_dir='outputs/reports'):
        """Generate comprehensive report"""
        print("\n" + "="*60)
        print("GENERATING REPORTS")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate forecast report
        report_path = self.report_generator.generate_forecast_report(
            self.forecasts,
            self.transformed_data,
            output_dir
        )
        
        print(f"✓ Forecast report saved: {report_path}")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # Time series plot
        ts_plot = self.visualizer.plot_time_series_forecast(
            self.transformed_data,
            self.forecasts,
            save_path=f"{output_dir}/forecast_charts.png"
        )
        
        # Regime plot
        regime_plot = self.visualizer.plot_regime_analysis(
            self.forecasts['regime'],
            save_path=f"{output_dir}/regime_analysis.png"
        )
        
        print("✓ Visualizations saved")
        
        # Save forecasts to JSON
        forecast_json = {
            'generated_at': datetime.now().isoformat(),
            'forecasts': {
                'growth': {
                    '3m': float(self.forecasts['growth']['3m']['forecast']),
                    '6m': float(self.forecasts['growth']['6m']['forecast']),
                    '9m': float(self.forecasts['growth']['9m']['forecast'])
                },
                'ppi': {
                    '3m': float(self.forecasts['ppi']['3m']['forecast']),
                    '6m': float(self.forecasts['ppi']['6m']['forecast'])
                },
                'cpi': {
                    '6m': float(self.forecasts['cpi']['6m']['forecast']),
                    '9m': float(self.forecasts['cpi']['9m']['forecast'])
                },
                'regime': self.forecasts['regime'],
                'signals': self.forecasts['signals']
            }
        }
        
        with open(f"{output_dir}/forecasts.json", 'w') as f:
            json.dump(forecast_json, f, indent=2)
            
        print("✓ Forecasts saved to JSON")
        
    def run(self):
        """Run the complete forecasting pipeline"""
        
        # Initialize system
        self.initialize()
        
        # Load data
        if not self.load_data():
            print("Failed to load data. Exiting.")
            return False
            
        # Train models
        self.train_models()
        
        # Generate forecasts
        self.generate_forecasts()
        
        # Run backtest
        backtest_results = self.backtest()
        
        # Generate reports
        self.generate_report()
        
        print("\n" + "="*60)
        print("FORECASTING COMPLETE")
        print("="*60)
        print("\nSummary:")
        print(f"• Growth outlook (6M): {self.forecasts['growth']['6m']['forecast']:.1f}")
        print(f"• Inflation outlook (6M): CPI {self.forecasts['cpi']['6m']['forecast']:.1f}%, PPI {self.forecasts['ppi']['6m']['forecast']:.1f}%")
        print(f"• Business cycle regime: {self.forecasts['regime']['current']}")
        print(f"• Primary allocation: {max(self.forecasts['signals'].items(), key=lambda x: x[1] if x[1] != 'neutral' else 0)[0]}")
        
        return True

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Business Cycle Forecasting System for Growth and Inflation'
    )
    
    parser.add_argument(
        '--mode',
        choices=['bloomberg', 'development', 'production'],
        default='production',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to data directory'
    )
    
    parser.add_argument(
        '--extract',
        action='store_true',
        help='Extract Bloomberg data (only works on Terminal)'
    )
    
    parser.add_argument(
        '--use-sample-data',
        action='store_true',
        help='Use sample data for development'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate detailed report'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/reports',
        help='Output directory for reports'
    )
    
    args = parser.parse_args()
    
    # Handle Bloomberg extraction
    if args.mode == 'bloomberg' and args.extract:
        print("Please run bloomberg_extraction/extract_data.py on Bloomberg Terminal")
        print("This script runs the forecasting models, not the data extraction")
        return
        
    # Set mode based on arguments
    if args.use_sample_data:
        args.mode = 'development'
        
    # Create and run system
    system = BusinessCycleForecastingSystem(
        mode=args.mode,
        data_path=args.data_path
    )
    
    success = system.run()
    
    if success:
        print("\n✓ All operations completed successfully")
    else:
        print("\n✗ Some operations failed")
        sys.exit(1)

if __name__ == "__main__":
    main()