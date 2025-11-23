"""
Data Loader Module
Handles loading and initial preprocessing of the Financial News and Stock Price Integration Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Load and preprocess the FNSPID dataset"""
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize the DataLoader
        
        Args:
            data_path: Path to the CSV file containing the dataset
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        
    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Load the dataset from CSV file
        
        Args:
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            Loaded DataFrame
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        self.df = pd.read_csv(self.data_path, **kwargs)
        
        # Validate required columns
        required_columns = ['headline', 'url', 'publisher', 'date', 'stock']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return self.df
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the loaded data
        
        Returns:
            Preprocessed DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        df = self.df.copy()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        
        # Extract date components
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.day_name()
        df['hour'] = df['date'].dt.hour
        df['date_only'] = df['date'].dt.date
        
        # Calculate headline length
        df['headline_length'] = df['headline'].astype(str).str.len()
        df['headline_word_count'] = df['headline'].astype(str).str.split().str.len()
        
        # Clean publisher field - extract domain if email-like
        df['publisher_domain'] = df['publisher'].astype(str).apply(
            lambda x: x.split('@')[1] if '@' in x else x
        )
        
        # Remove rows with missing critical data
        initial_count = len(df)
        df = df.dropna(subset=['headline', 'date', 'stock'])
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            print(f"Removed {removed_count} rows with missing critical data")
        
        self.df = df
        return df
    
    def get_data(self) -> pd.DataFrame:
        """Get the processed DataFrame"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() and preprocess_data() first.")
        return self.df



