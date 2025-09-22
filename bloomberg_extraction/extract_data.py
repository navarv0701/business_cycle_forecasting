#!/usr/bin/env python3
"""
Bloomberg Data Extractor for Business Cycle Forecasting
Standalone script - runs on Bloomberg Terminal
No dependencies from main project
"""

import sys
import os
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse

# Core dependencies only
import pandas as pd
import numpy as np

# Bloomberg API - will fail gracefully if not available
try:
    import blpapi
    BLOOMBERG_AVAILABLE = True
except ImportError:
    BLOOMBERG_AVAILABLE = False
    print("Warning: Bloomberg API not available. Running in test mode.")

class BloombergDataExtractor:
    """Extracts financial data from Bloomberg Terminal"""
    
    def __init__(self, output_dir: str = "bloomberg_data_export"):
        """Initialize the extractor with configuration"""
        
        # Complete ticker list with all details
        self.tickers = {
            # Growth indicators for ISM PMI model
            'USGG10YR': {
                'type': 'Index',
                'field': 'PX_LAST',
                'name': 'US_10Y_Yield',
                'description': 'US 10-Year Treasury Yield'
            },
            'USGG2YR': {
                'type': 'Index',
                'field': 'PX_LAST',
                'name': 'US_2Y_Yield',
                'description': 'US 2-Year Treasury Yield'
            },
            'NHSPSTOT': {
                'type': 'Index',
                'field': 'PX_LAST',
                'name': 'Housing_Permits',
                'description': 'US Housing Permits Total'
            },
            'CONSSENT': {
                'type': 'Index',
                'field': 'PX_LAST',
                'name': 'Consumer_Confidence',
                'description': 'Consumer Confidence Index'
            },
            'CGNOXAI': {
                'type': 'Index',
                'field': 'PX_LAST',
                'name': 'Core_Capital_Goods',
                'description': 'Core Capital Goods New Orders'
            },
            'CGNOXMOM': {
                'type': 'Index',
                'field': 'PX_LAST',
                'name': 'Core_Capital_Goods_MoM',
                'description': 'Core Capital Goods Month-over-Month'
            },
            'INJCAVG': {
                'type': 'Index',
                'field': 'PX_LAST',
                'name': 'Unemployment_Claims_4WK',
                'description': 'Initial Jobless Claims 4-Week Average'
            },
            
            # Inflation indicators
            'NAPMPMI': {
                'type': 'Index',
                'field': 'PX_LAST',
                'name': 'ISM_PMI',
                'description': 'ISM Manufacturing PMI'
            },
            'ECIWAG': {
                'type': 'Index',
                'field': 'PX_LAST',
                'name': 'Employment_Cost_Index',
                'description': 'Employment Cost Index - Wages'
            },
            'USGG5Y5Y': {
                'type': 'Index',
                'field': 'PX_LAST',
                'name': 'Inflation_Expect_5Y5Y',
                'description': '5-Year 5-Year Forward Inflation Expectation'
            },
            'USIMPY': {
                'type': 'Index',
                'field': 'PX_LAST',
                'name': 'Import_Price_Index',
                'description': 'US Import Price Index YoY'
            },
            'BDIY': {
                'type': 'Index',
                'field': 'PX_LAST',
                'name': 'Baltic_Dry_Index',
                'description': 'Baltic Dry Index'
            },
            'NFICCMPP': {
                'type': 'Index',
                'field': 'PX_LAST',
                'name': 'NFIB_Comp_Plans',
                'description': 'NFIB Small Business Compensation Plans'
            },
            
            # Target variables
            'USPPIY': {
                'type': 'Index',
                'field': 'PX_LAST',
                'name': 'PPI_YoY',
                'description': 'US Producer Price Index YoY'
            },
            'USCPIY': {
                'type': 'Index',
                'field': 'PX_LAST',
                'name': 'CPI_YoY',
                'description': 'US Consumer Price Index YoY'
            }
        }
        
        self.output_dir = output_dir
        self.session = None
        self.service = None
        
        # Default date range
        self.start_date = "20100101"
        self.end_date = datetime.today().strftime("%Y%m%d")
        
    def connect_to_bloomberg(self) -> bool:
        """Establish connection to Bloomberg Terminal"""
        
        if not BLOOMBERG_AVAILABLE:
            print("Bloomberg API not available - cannot connect")
            return False
            
        try:
            print("Connecting to Bloomberg Terminal...")
            print("  → Initializing session...")
            
            # Start a Session
            sessionOptions = blpapi.SessionOptions()
            sessionOptions.setServerHost("localhost")
            sessionOptions.setServerPort(8194)
            
            self.session = blpapi.Session(sessionOptions)
            
            # Start session
            if not self.session.start():
                print("  ✗ Failed to start Bloomberg session")
                return False
                
            print("  → Opening reference data service...")
            
            # Open service
            if not self.session.openService("//blp/refdata"):
                print("  ✗ Failed to open //blp/refdata service")
                return False
                
            self.service = self.session.getService("//blp/refdata")
            print("  ✓ Successfully connected to Bloomberg Terminal")
            return True
            
        except Exception as e:
            print(f"  ✗ Connection failed: {str(e)}")
            return False
            
    def fetch_historical_data(self, ticker: str, ticker_info: dict) -> Optional[pd.DataFrame]:
        """Fetch historical data for a single ticker"""
        
        if not self.session or not self.service:
            return None
            
        try:
            # Create request
            request = self.service.createRequest("HistoricalDataRequest")
            
            # Set security
            security = f"{ticker} {ticker_info['type']}"
            request.append("securities", security)
            
            # Set fields
            request.append("fields", ticker_info['field'])
            
            # Set date range
            request.set("startDate", self.start_date)
            request.set("endDate", self.end_date)
            
            # Set other options
            request.set("periodicitySelection", "DAILY")
            request.set("adjustmentNormal", True)
            request.set("adjustmentAbnormal", True)
            request.set("adjustmentSplit", True)
            
            # Send request
            self.session.sendRequest(request)
            
            # Process response
            data_points = []
            
            while True:
                event = self.session.nextEvent(5000)  # 5 second timeout
                
                for msg in event:
                    if msg.hasElement("securityData"):
                        securityData = msg.getElement("securityData")
                        
                        if securityData.hasElement("fieldData"):
                            fieldData = securityData.getElement("fieldData")
                            
                            for i in range(fieldData.numValues()):
                                element = fieldData.getValue(i)
                                
                                date_val = element.getElementAsString("date")
                                
                                if element.hasElement(ticker_info['field']):
                                    value = element.getElementAsFloat(ticker_info['field'])
                                    data_points.append({
                                        'date': date_val,
                                        'value': value
                                    })
                                    
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
                    
            # Convert to DataFrame
            if data_points:
                df = pd.DataFrame(data_points)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.columns = [ticker_info['name']]
                return df
            else:
                return None
                
        except Exception as e:
            print(f"    Error fetching {ticker}: {str(e)}")
            return None
            
    def extract_all_data(self) -> bool:
        """Main extraction routine"""
        
        print("\n" + "="*60)
        print("BLOOMBERG DATA EXTRACTION FOR BUSINESS CYCLE MODELS")
        print("="*60)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Connect to Bloomberg
        if not self.connect_to_bloomberg():
            print("\nFailed to connect to Bloomberg. Ensure Terminal is running.")
            return False
            
        print(f"\nExtracting data from {self.start_date} to {self.end_date}")
        print("-"*60)
        
        # Track extracted data
        all_data = {}
        failed_tickers = []
        metadata = {
            'extraction_date': datetime.now().isoformat(),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'tickers': {},
            'failed_tickers': []
        }
        
        # Extract each ticker
        for ticker, info in self.tickers.items():
            print(f"\n[{ticker}] {info['description']}")
            print(f"  → Fetching {info['type']} data...")
            
            df = self.fetch_historical_data(ticker, info)
            
            if df is not None and not df.empty:
                # Save individual file
                csv_filename = f"{info['name']}.csv"
                csv_path = os.path.join(self.output_dir, csv_filename)
                df.to_csv(csv_path)
                
                # Save as parquet for efficiency
                parquet_filename = f"{info['name']}.parquet"
                parquet_path = os.path.join(self.output_dir, parquet_filename)
                df.to_parquet(parquet_path)
                
                print(f"  ✓ Saved {len(df)} data points")
                print(f"    → CSV: {csv_filename}")
                print(f"    → Parquet: {parquet_filename}")
                
                all_data[info['name']] = df
                metadata['tickers'][ticker] = {
                    'name': info['name'],
                    'description': info['description'],
                    'rows': len(df),
                    'csv_file': csv_filename,
                    'parquet_file': parquet_filename,
                    'date_range': {
                        'start': df.index.min().strftime('%Y-%m-%d'),
                        'end': df.index.max().strftime('%Y-%m-%d')
                    }
                }
            else:
                print(f"  ✗ Failed to fetch data")
                failed_tickers.append(ticker)
                metadata['failed_tickers'].append({
                    'ticker': ticker,
                    'name': info['name'],
                    'description': info['description']
                })
                
        # Create master dataset
        if all_data:
            print("\n" + "-"*60)
            print("Creating master dataset...")
            
            # Combine all data
            master_df = pd.concat(all_data.values(), axis=1, join='outer')
            master_df = master_df.sort_index()
            
            # Save master files
            master_csv = os.path.join(self.output_dir, 'MASTER_DATA.csv')
            master_parquet = os.path.join(self.output_dir, 'MASTER_DATA.parquet')
            
            master_df.to_csv(master_csv)
            master_df.to_parquet(master_parquet)
            
            print(f"✓ Master dataset created:")
            print(f"  → Shape: {master_df.shape}")
            print(f"  → Date range: {master_df.index.min()} to {master_df.index.max()}")
            print(f"  → Files: MASTER_DATA.csv, MASTER_DATA.parquet")
            
            metadata['master_data'] = {
                'rows': len(master_df),
                'columns': len(master_df.columns),
                'date_range': {
                    'start': master_df.index.min().strftime('%Y-%m-%d'),
                    'end': master_df.index.max().strftime('%Y-%m-%d')
                }
            }
            
        # Save metadata
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Disconnect
        if self.session:
            self.session.stop()
            
        # Print summary
        print("\n" + "="*60)
        print("EXTRACTION COMPLETE")
        print("="*60)
        print(f"✓ Successfully extracted: {len(all_data)}/{len(self.tickers)} tickers")
        
        if failed_tickers:
            print(f"✗ Failed tickers: {', '.join(failed_tickers)}")
            
        print(f"\n✓ Output directory: {os.path.abspath(self.output_dir)}")
        print("\nFiles created:")
        print(f"  → {len(all_data)} individual ticker files (CSV & Parquet)")
        print(f"  → 1 master data file (CSV & Parquet)")
        print(f"  → 1 metadata file (JSON)")
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("1. Copy the entire '{}' folder to your development machine".format(self.output_dir))
        print("2. Place it in your project's data/bloomberg_cache/ directory")
        print("3. Run: python main.py --mode production")
        print("="*60)
        
        return True
        
def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Extract Bloomberg data for business cycle models')
    parser.add_argument('--start', type=str, help='Start date (YYYYMMDD)')
    parser.add_argument('--end', type=str, help='End date (YYYYMMDD)')
    parser.add_argument('--recent', action='store_true', help='Extract only last 2 years')
    parser.add_argument('--output', type=str, default='bloomberg_data_export', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = BloombergDataExtractor(output_dir=args.output)
    
    # Set date range
    if args.recent:
        extractor.start_date = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
        print("Quick mode: Extracting last 2 years only")
    elif args.start:
        extractor.start_date = args.start
    if args.end:
        extractor.end_date = args.end
        
    # Run extraction
    success = extractor.extract_all_data()
    
    if not success:
        print("\nExtraction failed. Please check Bloomberg Terminal connection.")
        sys.exit(1)
        
if __name__ == "__main__":
    main()