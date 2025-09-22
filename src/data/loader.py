"""
Data loading utilities for Bloomberg and sample data
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
import numpy as np

class DataLoader:
    """Load data from various sources"""
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize data loader
        
        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
        self.metadata = None
        
    def load_all_data(self) -> Optional[pd.DataFrame]:
        """
        Load all data from the data directory
        
        Returns:
            DataFrame with all data or None if not found
        """
        # Try to load master file first
        master_files = [
            self.data_path / 'MASTER_DATA.parquet',
            self.data_path / 'MASTER_DATA.csv',
            self.data_path / 'master_data.parquet',
            self.data_path / 'master_data.csv'
        ]
        
        for file_path in master_files:
            if file_path.exists():
                print(f"Loading data from {file_path.name}...")
                
                if file_path.suffix == '.parquet':
                    data = pd.read_parquet(file_path)
                else:
                    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                # Load metadata if available
                self._load_metadata()
                
                return data
                
        # If no master file, try to combine individual files
        print("No master file found, attempting to load individual files...")
        return self._load_individual_files()
        
    def _load_individual_files(self) -> Optional[pd.DataFrame]:
        """Load and combine individual ticker files"""
        
        data_frames = {}
        
        # Look for CSV and parquet files
        for file_path in self.data_path.glob('*.csv'):
            # Skip master files
            if 'master' in file_path.name.lower():
                continue
                
            name = file_path.stem
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            if len(df.columns) == 1:
                data_frames[name] = df
            else:
                # Multi-column file
                for col in df.columns:
                    data_frames[col] = df[[col]]
                    
        # Also check parquet files
        for file_path in self.data_path.glob('*.parquet'):
            if 'master' in file_path.name.lower():
                continue
                
            name = file_path.stem
            if name not in data_frames:  # Don't duplicate
                df = pd.read_parquet(file_path)
                
                if len(df.columns) == 1:
                    data_frames[name] = df
                else:
                    for col in df.columns:
                        data_frames[col] = df[[col]]
                        
        if data_frames:
            print(f"Found {len(data_frames)} individual data files")
            # Combine all data
            combined = pd.concat(data_frames.values(), axis=1, join='outer')
            combined = combined.sort_index()
            return combined
        else:
            print("No data files found")
            return None
            
    def _load_metadata(self):
        """Load metadata file if it exists"""
        metadata_path = self.data_path / 'metadata.json'
        
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
                
    def get_data_info(self) -> Dict:
        """Get information about loaded data"""
        
        if self.metadata:
            return self.metadata
        else:
            return {
                'source': 'Unknown',
                'path': str(self.data_path)
            }