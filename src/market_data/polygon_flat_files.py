"""
Polygon.io Flat Files Client for Historical Options Data
Downloads and processes historical options data for ML training and backtesting
"""

import os
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import gzip
import io
from loguru import logger
import requests
from urllib.parse import urljoin
import time

from src.config.polygon_config import get_polygon_api_key, get_s3_config, get_data_path


class PolygonFlatFilesClient:
    """
    Client for downloading and processing Polygon.io flat files
    
    Features:
    - Automated S3 data download
    - Historical data processing
    - Data cleaning and validation
    - Feature engineering for ML
    - Backtesting data preparation
    """
    
    def __init__(self, api_key: Optional[str] = None, data_dir: Optional[str] = None):
        """
        Initialize flat files client
        
        Args:
            api_key: Polygon.io API key (optional, uses config if not provided)
            data_dir: Local directory for storing downloaded data (optional, uses config if not provided)
        """
        self.api_key = api_key or get_polygon_api_key()
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in environment or config!")
        
        self.data_dir = Path(data_dir or get_data_path('flat_files'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # S3 configuration for Polygon flat files
        s3_config = get_s3_config()
        self.s3_client = boto3.client(
            's3',
            endpoint_url=s3_config['endpoint_url'],
            aws_access_key_id=s3_config['access_key_id'],
            aws_secret_access_key=s3_config['secret_access_key'],
            region_name=s3_config['region_name']
        )
        
        # S3 bucket name
        self.bucket_name = s3_config['bucket_name']
        
        # Data structure definitions
        self.trades_columns = [
            'timestamp', 'price', 'size', 'exchange', 'conditions', 'participant_timestamp'
        ]
        
        self.quotes_columns = [
            'timestamp', 'bid', 'ask', 'bid_size', 'ask_size', 'exchange', 'conditions'
        ]
        
        self.aggregates_columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions'
        ]
        
        logger.info("PolygonFlatFilesClient initialized")
    
    def list_available_dates(self, data_type: str = "trades") -> List[str]:
        """
        List available dates for flat files
        
        Args:
            data_type: Type of data (trades, quotes, aggregates)
            
        Returns:
            List of available dates in YYYY-MM-DD format
        """
        try:
            # Use the flatfile bucket with data type prefix
            prefix = f"options/{data_type}/"
            
            # List objects in the bucket
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                Delimiter="/"
            )
            
            dates = []
            if 'CommonPrefixes' in response:
                for prefix_obj in response['CommonPrefixes']:
                    # Extract date from path like "options/trades/2024-01-15/"
                    path = prefix_obj['Prefix']
                    date_part = path.replace(prefix, '').rstrip('/')
                    if len(date_part) == 10 and date_part.count('-') == 2:  # YYYY-MM-DD format
                        dates.append(date_part)
            
            dates.sort()
            logger.info(f"Found {len(dates)} available dates for {data_type}")
            return dates
            
        except Exception as e:
            logger.error(f"Error listing available dates: {e}")
            return []
    
    def download_data(
        self,
        data_type: str,
        date: str,
        symbols: Optional[List[str]] = None,
        force_download: bool = False
    ) -> Optional[Path]:
        """
        Download flat file data for a specific date
        
        Args:
            data_type: Type of data (trades, quotes, aggregates)
            date: Date in YYYY-MM-DD format
            symbols: List of symbols to filter (optional)
            force_download: Force re-download even if file exists
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            # Create local file path
            local_file = self.data_dir / f"{data_type}_{date}.csv.gz"
            
            # Check if file already exists
            if local_file.exists() and not force_download:
                logger.info(f"File already exists: {local_file}")
                return local_file
            
            # Download from S3 using the correct path structure
            s3_key = f"options/{data_type}/{date}/{data_type}_{date}.csv.gz"
            
            logger.info(f"Downloading {data_type} data for {date}...")
            logger.info(f"S3 Key: {s3_key}")
            
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                str(local_file)
            )
            
            logger.info(f"Downloaded {data_type} data for {date}: {local_file}")
            return local_file
            
        except Exception as e:
            logger.error(f"Error downloading {data_type} data for {date}: {e}")
            return None
    
    def load_trades_data(
        self,
        date: str,
        symbols: Optional[List[str]] = None,
        max_rows: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load trades data for a specific date
        
        Args:
            date: Date in YYYY-MM-DD format
            symbols: List of symbols to filter (optional)
            max_rows: Maximum number of rows to load (optional)
            
        Returns:
            DataFrame with trades data or None if failed
        """
        try:
            # Download data if not exists
            file_path = self.download_data("trades", date)
            if not file_path or not file_path.exists():
                return None
            
            logger.info(f"Loading trades data for {date}...")
            
            # Read compressed CSV
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f, nrows=max_rows)
            
            # Rename columns to standard format
            if 't' in df.columns:
                df = df.rename(columns={'t': 'timestamp'})
            if 'p' in df.columns:
                df = df.rename(columns={'p': 'price'})
            if 's' in df.columns:
                df = df.rename(columns={'s': 'size'})
            if 'x' in df.columns:
                df = df.rename(columns={'x': 'exchange'})
            if 'c' in df.columns:
                df = df.rename(columns={'c': 'conditions'})
            if 'z' in df.columns:
                df = df.rename(columns={'z': 'participant_timestamp'})
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            
            # Filter by symbols if provided
            if symbols and 'sym' in df.columns:
                df = df[df['sym'].isin(symbols)]
            
            # Add derived columns
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            
            logger.info(f"Loaded {len(df)} trades records for {date}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading trades data for {date}: {e}")
            return None
    
    def load_quotes_data(
        self,
        date: str,
        symbols: Optional[List[str]] = None,
        max_rows: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load quotes data for a specific date
        
        Args:
            date: Date in YYYY-MM-DD format
            symbols: List of symbols to filter (optional)
            max_rows: Maximum number of rows to load (optional)
            
        Returns:
            DataFrame with quotes data or None if failed
        """
        try:
            # Download data if not exists
            file_path = self.download_data("quotes", date)
            if not file_path or not file_path.exists():
                return None
            
            logger.info(f"Loading quotes data for {date}...")
            
            # Read compressed CSV
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f, nrows=max_rows)
            
            # Rename columns to standard format
            if 't' in df.columns:
                df = df.rename(columns={'t': 'timestamp'})
            if 'bp' in df.columns:
                df = df.rename(columns={'bp': 'bid'})
            if 'ap' in df.columns:
                df = df.rename(columns={'ap': 'ask'})
            if 'bs' in df.columns:
                df = df.rename(columns={'bs': 'bid_size'})
            if 'as' in df.columns:
                df = df.rename(columns={'as': 'ask_size'})
            if 'x' in df.columns:
                df = df.rename(columns={'x': 'exchange'})
            if 'c' in df.columns:
                df = df.rename(columns={'c': 'conditions'})
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            
            # Filter by symbols if provided
            if symbols and 'sym' in df.columns:
                df = df[df['sym'].isin(symbols)]
            
            # Add derived columns
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['spread'] = df['ask'] - df['bid']
            df['mid_price'] = (df['bid'] + df['ask']) / 2
            df['spread_pct'] = (df['spread'] / df['mid_price']) * 100
            
            logger.info(f"Loaded {len(df)} quotes records for {date}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading quotes data for {date}: {e}")
            return None
    
    def load_aggregates_data(
        self,
        date: str,
        symbols: Optional[List[str]] = None,
        max_rows: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load aggregates data for a specific date
        
        Args:
            date: Date in YYYY-MM-DD format
            symbols: List of symbols to filter (optional)
            max_rows: Maximum number of rows to load (optional)
            
        Returns:
            DataFrame with aggregates data or None if failed
        """
        try:
            # Download data if not exists
            file_path = self.download_data("aggregates", date)
            if not file_path or not file_path.exists():
                return None
            
            logger.info(f"Loading aggregates data for {date}...")
            
            # Read compressed CSV
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f, nrows=max_rows)
            
            # Rename columns to standard format
            if 't' in df.columns:
                df = df.rename(columns={'t': 'timestamp'})
            if 'o' in df.columns:
                df = df.rename(columns={'o': 'open'})
            if 'h' in df.columns:
                df = df.rename(columns={'h': 'high'})
            if 'l' in df.columns:
                df = df.rename(columns={'l': 'low'})
            if 'c' in df.columns:
                df = df.rename(columns={'c': 'close'})
            if 'v' in df.columns:
                df = df.rename(columns={'v': 'volume'})
            if 'vw' in df.columns:
                df = df.rename(columns={'vw': 'vwap'})
            if 'n' in df.columns:
                df = df.rename(columns={'n': 'transactions'})
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            
            # Filter by symbols if provided
            if symbols and 'sym' in df.columns:
                df = df[df['sym'].isin(symbols)]
            
            # Add derived columns
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['range'] = df['high'] - df['low']
            df['range_pct'] = (df['range'] / df['close']) * 100
            
            logger.info(f"Loaded {len(df)} aggregates records for {date}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading aggregates data for {date}: {e}")
            return None
    
    def get_historical_data_range(
        self,
        data_type: str,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        max_rows_per_day: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical data for a date range
        
        Args:
            data_type: Type of data (trades, quotes, aggregates)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            symbols: List of symbols to filter (optional)
            max_rows_per_day: Maximum rows per day (optional)
            
        Returns:
            Combined DataFrame with historical data
        """
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            all_data = []
            current_date = start_dt
            
            while current_date <= end_dt:
                date_str = current_date.strftime('%Y-%m-%d')
                
                logger.info(f"Loading {data_type} data for {date_str}...")
                
                if data_type == "trades":
                    data = self.load_trades_data(date_str, symbols, max_rows_per_day)
                elif data_type == "quotes":
                    data = self.load_quotes_data(date_str, symbols, max_rows_per_day)
                elif data_type == "aggregates":
                    data = self.load_aggregates_data(date_str, symbols, max_rows_per_day)
                else:
                    logger.error(f"Unknown data type: {data_type}")
                    current_date += timedelta(days=1)
                    continue
                
                if data is not None and len(data) > 0:
                    all_data.append(data)
                
                current_date += timedelta(days=1)
                
                # Rate limiting
                time.sleep(0.1)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                logger.info(f"Combined {len(all_data)} days of {data_type} data: {len(combined_df)} total records")
                return combined_df
            else:
                logger.warning(f"No data found for {data_type} between {start_date} and {end_date}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting historical data range: {e}")
            return pd.DataFrame()
    
    def create_ml_features(
        self,
        trades_df: pd.DataFrame,
        quotes_df: pd.DataFrame,
        aggregates_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create ML features from historical data
        
        Args:
            trades_df: Trades DataFrame
            quotes_df: Quotes DataFrame
            aggregates_df: Aggregates DataFrame
            
        Returns:
            DataFrame with ML features
        """
        try:
            logger.info("Creating ML features from historical data...")
            
            features = []
            
            # Process each symbol
            symbols = set()
            if not trades_df.empty and 'sym' in trades_df.columns:
                symbols.update(trades_df['sym'].unique())
            if not quotes_df.empty and 'sym' in quotes_df.columns:
                symbols.update(quotes_df['sym'].unique())
            if not aggregates_df.empty and 'sym' in aggregates_df.columns:
                symbols.update(aggregates_df['sym'].unique())
            
            for symbol in symbols:
                logger.info(f"Processing features for {symbol}...")
                
                # Get data for this symbol
                symbol_trades = trades_df[trades_df['sym'] == symbol] if not trades_df.empty else pd.DataFrame()
                symbol_quotes = quotes_df[quotes_df['sym'] == symbol] if not quotes_df.empty else pd.DataFrame()
                symbol_aggregates = aggregates_df[aggregates_df['sym'] == symbol] if not aggregates_df.empty else pd.DataFrame()
                
                # Create daily features
                daily_features = self._create_daily_features(
                    symbol, symbol_trades, symbol_quotes, symbol_aggregates
                )
                
                if daily_features is not None:
                    features.append(daily_features)
            
            if features:
                features_df = pd.concat(features, ignore_index=True)
                logger.info(f"Created ML features for {len(features_df)} symbol-days")
                return features_df
            else:
                logger.warning("No features created")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error creating ML features: {e}")
            return pd.DataFrame()
    
    def _create_daily_features(
        self,
        symbol: str,
        trades_df: pd.DataFrame,
        quotes_df: pd.DataFrame,
        aggregates_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Create daily features for a symbol"""
        try:
            if trades_df.empty and quotes_df.empty and aggregates_df.empty:
                return None
            
            # Get unique dates
            dates = set()
            if not trades_df.empty:
                dates.update(trades_df['date'].unique())
            if not quotes_df.empty:
                dates.update(quotes_df['date'].unique())
            if not aggregates_df.empty:
                dates.update(aggregates_df['date'].unique())
            
            daily_features = []
            
            for date in dates:
                # Filter data for this date
                day_trades = trades_df[trades_df['date'] == date] if not trades_df.empty else pd.DataFrame()
                day_quotes = quotes_df[quotes_df['date'] == date] if not quotes_df.empty else pd.DataFrame()
                day_aggregates = aggregates_df[aggregates_df['date'] == date] if not aggregates_df.empty else pd.DataFrame()
                
                # Create features for this day
                features = {
                    'symbol': symbol,
                    'date': date,
                    'timestamp': datetime.combine(date, datetime.min.time())
                }
                
                # Trade features
                if not day_trades.empty:
                    features.update(self._extract_trade_features(day_trades))
                
                # Quote features
                if not day_quotes.empty:
                    features.update(self._extract_quote_features(day_quotes))
                
                # Aggregate features
                if not day_aggregates.empty:
                    features.update(self._extract_aggregate_features(day_aggregates))
                
                daily_features.append(features)
            
            if daily_features:
                return pd.DataFrame(daily_features)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error creating daily features for {symbol}: {e}")
            return None
    
    def _extract_trade_features(self, trades_df: pd.DataFrame) -> Dict:
        """Extract features from trades data"""
        features = {}
        
        try:
            # Basic trade statistics
            features['trade_count'] = len(trades_df)
            features['total_volume'] = trades_df['size'].sum()
            features['avg_trade_size'] = trades_df['size'].mean()
            features['max_trade_size'] = trades_df['size'].max()
            features['min_trade_size'] = trades_df['size'].min()
            
            # Price statistics
            features['avg_trade_price'] = trades_df['price'].mean()
            features['max_trade_price'] = trades_df['price'].max()
            features['min_trade_price'] = trades_df['price'].min()
            features['price_std'] = trades_df['price'].std()
            
            # Time-based features
            features['first_trade_time'] = trades_df['timestamp'].min()
            features['last_trade_time'] = trades_df['timestamp'].max()
            features['trading_duration_hours'] = (
                trades_df['timestamp'].max() - trades_df['timestamp'].min()
            ).total_seconds() / 3600
            
            # Volume-weighted average price
            if 'size' in trades_df.columns and 'price' in trades_df.columns:
                vwap = (trades_df['price'] * trades_df['size']).sum() / trades_df['size'].sum()
                features['vwap'] = vwap
            
            # Price volatility (intraday)
            if len(trades_df) > 1:
                price_returns = trades_df['price'].pct_change().dropna()
                features['price_volatility'] = price_returns.std()
                features['price_skewness'] = price_returns.skew()
                features['price_kurtosis'] = price_returns.kurtosis()
            
            # Exchange distribution
            if 'exchange' in trades_df.columns:
                exchange_counts = trades_df['exchange'].value_counts()
                features['top_exchange'] = exchange_counts.index[0] if len(exchange_counts) > 0 else None
                features['top_exchange_pct'] = exchange_counts.iloc[0] / len(trades_df) if len(exchange_counts) > 0 else 0
            
        except Exception as e:
            logger.error(f"Error extracting trade features: {e}")
        
        return features
    
    def _extract_quote_features(self, quotes_df: pd.DataFrame) -> Dict:
        """Extract features from quotes data"""
        features = {}
        
        try:
            # Basic quote statistics
            features['quote_count'] = len(quotes_df)
            
            # Spread features
            if 'spread' in quotes_df.columns:
                features['avg_spread'] = quotes_df['spread'].mean()
                features['min_spread'] = quotes_df['spread'].min()
                features['max_spread'] = quotes_df['spread'].max()
                features['spread_std'] = quotes_df['spread'].std()
            
            if 'spread_pct' in quotes_df.columns:
                features['avg_spread_pct'] = quotes_df['spread_pct'].mean()
                features['min_spread_pct'] = quotes_df['spread_pct'].min()
                features['max_spread_pct'] = quotes_df['spread_pct'].max()
                features['spread_pct_std'] = quotes_df['spread_pct'].std()
            
            # Bid/Ask features
            if 'bid' in quotes_df.columns and 'ask' in quotes_df.columns:
                features['avg_bid'] = quotes_df['bid'].mean()
                features['avg_ask'] = quotes_df['ask'].mean()
                features['avg_mid_price'] = quotes_df['mid_price'].mean() if 'mid_price' in quotes_df.columns else (quotes_df['bid'] + quotes_df['ask']).mean() / 2
            
            # Size features
            if 'bid_size' in quotes_df.columns and 'ask_size' in quotes_df.columns:
                features['avg_bid_size'] = quotes_df['bid_size'].mean()
                features['avg_ask_size'] = quotes_df['ask_size'].mean()
                features['total_bid_size'] = quotes_df['bid_size'].sum()
                features['total_ask_size'] = quotes_df['ask_size'].sum()
            
            # Time-based features
            features['first_quote_time'] = quotes_df['timestamp'].min()
            features['last_quote_time'] = quotes_df['timestamp'].max()
            features['quote_duration_hours'] = (
                quotes_df['timestamp'].max() - quotes_df['timestamp'].min()
            ).total_seconds() / 3600
            
            # Quote frequency
            if len(quotes_df) > 1:
                time_diff = quotes_df['timestamp'].diff().dropna()
                features['avg_quote_interval_seconds'] = time_diff.dt.total_seconds().mean()
                features['quote_frequency_per_minute'] = 60 / features['avg_quote_interval_seconds'] if features['avg_quote_interval_seconds'] > 0 else 0
            
        except Exception as e:
            logger.error(f"Error extracting quote features: {e}")
        
        return features
    
    def _extract_aggregate_features(self, aggregates_df: pd.DataFrame) -> Dict:
        """Extract features from aggregates data"""
        features = {}
        
        try:
            # Basic aggregate statistics
            features['aggregate_count'] = len(aggregates_df)
            
            # OHLC features
            if all(col in aggregates_df.columns for col in ['open', 'high', 'low', 'close']):
                features['open_price'] = aggregates_df['open'].iloc[0] if len(aggregates_df) > 0 else None
                features['close_price'] = aggregates_df['close'].iloc[-1] if len(aggregates_df) > 0 else None
                features['high_price'] = aggregates_df['high'].max()
                features['low_price'] = aggregates_df['low'].min()
                features['price_range'] = features['high_price'] - features['low_price']
                features['price_range_pct'] = (features['price_range'] / features['close_price']) * 100 if features['close_price'] and features['close_price'] > 0 else 0
                
                # Daily return
                if features['open_price'] and features['close_price']:
                    features['daily_return'] = (features['close_price'] - features['open_price']) / features['open_price']
            
            # Volume features
            if 'volume' in aggregates_df.columns:
                features['total_volume'] = aggregates_df['volume'].sum()
                features['avg_volume'] = aggregates_df['volume'].mean()
                features['max_volume'] = aggregates_df['volume'].max()
                features['min_volume'] = aggregates_df['volume'].min()
                features['volume_std'] = aggregates_df['volume'].std()
            
            # VWAP features
            if 'vwap' in aggregates_df.columns:
                features['avg_vwap'] = aggregates_df['vwap'].mean()
                features['final_vwap'] = aggregates_df['vwap'].iloc[-1] if len(aggregates_df) > 0 else None
            
            # Transaction features
            if 'transactions' in aggregates_df.columns:
                features['total_transactions'] = aggregates_df['transactions'].sum()
                features['avg_transactions'] = aggregates_df['transactions'].mean()
            
            # Time-based features
            features['first_aggregate_time'] = aggregates_df['timestamp'].min()
            features['last_aggregate_time'] = aggregates_df['timestamp'].max()
            features['aggregate_duration_hours'] = (
                aggregates_df['timestamp'].max() - aggregates_df['timestamp'].min()
            ).total_seconds() / 3600
            
        except Exception as e:
            logger.error(f"Error extracting aggregate features: {e}")
        
        return features
    
    def save_ml_features(self, features_df: pd.DataFrame, filename: str) -> bool:
        """
        Save ML features to file
        
        Args:
            features_df: DataFrame with ML features
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = self.data_dir / filename
            features_df.to_csv(output_path, index=False)
            logger.info(f"Saved ML features to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving ML features: {e}")
            return False
    
    def load_ml_features(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Load ML features from file
        
        Args:
            filename: Input filename
            
        Returns:
            DataFrame with ML features or None if failed
        """
        try:
            input_path = self.data_dir / filename
            if not input_path.exists():
                logger.warning(f"Features file not found: {input_path}")
                return None
            
            features_df = pd.read_csv(input_path)
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
            features_df['date'] = pd.to_datetime(features_df['date']).dt.date
            
            logger.info(f"Loaded ML features from {input_path}: {len(features_df)} records")
            return features_df
        except Exception as e:
            logger.error(f"Error loading ML features: {e}")
            return None
