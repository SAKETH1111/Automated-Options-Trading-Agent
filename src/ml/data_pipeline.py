"""
Advanced ML Data Pipeline for Options Trading
Uses Polygon.io flat files data for comprehensive ML model training
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from src.market_data.polygon_flat_files import PolygonFlatFilesClient


class OptionsMLDataPipeline:
    """
    Advanced ML data pipeline for options trading
    
    Features:
    - Historical data processing from flat files
    - Advanced feature engineering
    - Multiple ML model training
    - Backtesting framework
    - Model evaluation and selection
    """
    
    def __init__(self, api_key: Optional[str] = None, data_dir: str = "data/ml"):
        """
        Initialize ML data pipeline
        
        Args:
            api_key: Polygon.io API key
            data_dir: Directory for ML data storage
        """
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize flat files client
        self.flat_files_client = PolygonFlatFilesClient(api_key, str(self.data_dir / "flat_files"))
        
        # ML models
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Feature engineering
        self.feature_columns = []
        self.target_columns = []
        
        logger.info("OptionsMLDataPipeline initialized")
    
    def create_comprehensive_dataset(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        save_features: bool = True
    ) -> pd.DataFrame:
        """
        Create comprehensive ML dataset from historical data
        
        Args:
            symbols: List of option symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            save_features: Whether to save features to file
            
        Returns:
            DataFrame with comprehensive ML features
        """
        try:
            logger.info(f"Creating comprehensive dataset for {len(symbols)} symbols from {start_date} to {end_date}")
            
            # Load historical data
            logger.info("Loading historical trades data...")
            trades_df = self.flat_files_client.get_historical_data_range(
                "trades", start_date, end_date, symbols
            )
            
            logger.info("Loading historical quotes data...")
            quotes_df = self.flat_files_client.get_historical_data_range(
                "quotes", start_date, end_date, symbols
            )
            
            logger.info("Loading historical aggregates data...")
            aggregates_df = self.flat_files_client.get_historical_data_range(
                "aggregates", start_date, end_date, symbols
            )
            
            # Create ML features
            logger.info("Creating ML features...")
            features_df = self.flat_files_client.create_ml_features(
                trades_df, quotes_df, aggregates_df
            )
            
            if features_df.empty:
                logger.error("No features created from historical data")
                return pd.DataFrame()
            
            # Add advanced features
            logger.info("Adding advanced features...")
            enhanced_features = self._add_advanced_features(features_df)
            
            # Add technical indicators
            logger.info("Adding technical indicators...")
            technical_features = self._add_technical_indicators(enhanced_features)
            
            # Add market microstructure features
            logger.info("Adding market microstructure features...")
            microstructure_features = self._add_microstructure_features(technical_features)
            
            # Add volatility features
            logger.info("Adding volatility features...")
            volatility_features = self._add_volatility_features(microstructure_features)
            
            # Add time-based features
            logger.info("Adding time-based features...")
            time_features = self._add_time_features(volatility_features)
            
            # Add target variables
            logger.info("Adding target variables...")
            final_dataset = self._add_target_variables(time_features)
            
            # Save features if requested
            if save_features:
                filename = f"ml_features_{start_date}_{end_date}.csv"
                self.flat_files_client.save_ml_features(final_dataset, filename)
            
            logger.info(f"Created comprehensive dataset with {len(final_dataset)} records and {len(final_dataset.columns)} features")
            return final_dataset
            
        except Exception as e:
            logger.error(f"Error creating comprehensive dataset: {e}")
            return pd.DataFrame()
    
    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced features to the dataset"""
        try:
            df = df.copy()
            
            # Sort by symbol and date
            df = df.sort_values(['symbol', 'date'])
            
            # Add lagged features
            for col in ['avg_trade_price', 'total_volume', 'avg_spread_pct']:
                if col in df.columns:
                    df[f'{col}_lag_1'] = df.groupby('symbol')[col].shift(1)
                    df[f'{col}_lag_2'] = df.groupby('symbol')[col].shift(2)
                    df[f'{col}_lag_3'] = df.groupby('symbol')[col].shift(3)
            
            # Add rolling statistics
            for col in ['avg_trade_price', 'total_volume', 'avg_spread_pct']:
                if col in df.columns:
                    df[f'{col}_rolling_mean_3'] = df.groupby('symbol')[col].rolling(3).mean().values
                    df[f'{col}_rolling_std_3'] = df.groupby('symbol')[col].rolling(3).std().values
                    df[f'{col}_rolling_mean_7'] = df.groupby('symbol')[col].rolling(7).mean().values
                    df[f'{col}_rolling_std_7'] = df.groupby('symbol')[col].rolling(7).std().values
            
            # Add price momentum
            if 'avg_trade_price' in df.columns:
                df['price_momentum_1'] = df.groupby('symbol')['avg_trade_price'].pct_change(1)
                df['price_momentum_3'] = df.groupby('symbol')['avg_trade_price'].pct_change(3)
                df['price_momentum_7'] = df.groupby('symbol')['avg_trade_price'].pct_change(7)
            
            # Add volume momentum
            if 'total_volume' in df.columns:
                df['volume_momentum_1'] = df.groupby('symbol')['total_volume'].pct_change(1)
                df['volume_momentum_3'] = df.groupby('symbol')['total_volume'].pct_change(3)
                df['volume_momentum_7'] = df.groupby('symbol')['total_volume'].pct_change(7)
            
            # Add spread momentum
            if 'avg_spread_pct' in df.columns:
                df['spread_momentum_1'] = df.groupby('symbol')['avg_spread_pct'].pct_change(1)
                df['spread_momentum_3'] = df.groupby('symbol')['avg_spread_pct'].pct_change(3)
                df['spread_momentum_7'] = df.groupby('symbol')['avg_spread_pct'].pct_change(7)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding advanced features: {e}")
            return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset"""
        try:
            df = df.copy()
            
            # Sort by symbol and date
            df = df.sort_values(['symbol', 'date'])
            
            # Add moving averages
            for col in ['avg_trade_price', 'close_price']:
                if col in df.columns:
                    df[f'{col}_sma_5'] = df.groupby('symbol')[col].rolling(5).mean().values
                    df[f'{col}_sma_10'] = df.groupby('symbol')[col].rolling(10).mean().values
                    df[f'{col}_sma_20'] = df.groupby('symbol')[col].rolling(20).mean().values
                    df[f'{col}_ema_5'] = df.groupby('symbol')[col].ewm(span=5).mean().values
                    df[f'{col}_ema_10'] = df.groupby('symbol')[col].ewm(span=10).mean().values
                    df[f'{col}_ema_20'] = df.groupby('symbol')[col].ewm(span=20).mean().values
            
            # Add RSI
            for col in ['avg_trade_price', 'close_price']:
                if col in df.columns:
                    df[f'{col}_rsi_14'] = self._calculate_rsi(df.groupby('symbol')[col], 14)
            
            # Add MACD
            for col in ['avg_trade_price', 'close_price']:
                if col in df.columns:
                    macd_data = self._calculate_macd(df.groupby('symbol')[col])
                    df[f'{col}_macd'] = macd_data['macd']
                    df[f'{col}_macd_signal'] = macd_data['signal']
                    df[f'{col}_macd_histogram'] = macd_data['histogram']
            
            # Add Bollinger Bands
            for col in ['avg_trade_price', 'close_price']:
                if col in df.columns:
                    bb_data = self._calculate_bollinger_bands(df.groupby('symbol')[col], 20, 2)
                    df[f'{col}_bb_upper'] = bb_data['upper']
                    df[f'{col}_bb_middle'] = bb_data['middle']
                    df[f'{col}_bb_lower'] = bb_data['lower']
                    df[f'{col}_bb_width'] = bb_data['width']
                    df[f'{col}_bb_position'] = bb_data['position']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        try:
            df = df.copy()
            
            # Add bid-ask spread features
            if 'avg_spread' in df.columns and 'avg_mid_price' in df.columns:
                df['spread_to_mid_ratio'] = df['avg_spread'] / df['avg_mid_price']
                df['spread_impact'] = df['avg_spread'] / df['avg_trade_price'] if 'avg_trade_price' in df.columns else 0
            
            # Add volume features
            if 'total_volume' in df.columns and 'trade_count' in df.columns:
                df['avg_trade_size'] = df['total_volume'] / df['trade_count']
                df['volume_per_trade'] = df['total_volume'] / df['trade_count']
            
            # Add quote frequency features
            if 'quote_count' in df.columns and 'quote_duration_hours' in df.columns:
                df['quotes_per_hour'] = df['quote_count'] / df['quote_duration_hours']
                df['quote_intensity'] = df['quote_count'] / df['total_volume'] if 'total_volume' in df.columns else 0
            
            # Add trade frequency features
            if 'trade_count' in df.columns and 'trading_duration_hours' in df.columns:
                df['trades_per_hour'] = df['trade_count'] / df['trading_duration_hours']
                df['trade_intensity'] = df['trade_count'] / df['total_volume'] if 'total_volume' in df.columns else 0
            
            # Add price impact features
            if 'price_volatility' in df.columns and 'total_volume' in df.columns:
                df['volume_price_impact'] = df['price_volatility'] / df['total_volume']
                df['price_impact_per_trade'] = df['price_volatility'] / df['trade_count'] if 'trade_count' in df.columns else 0
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding microstructure features: {e}")
            return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        try:
            df = df.copy()
            
            # Sort by symbol and date
            df = df.sort_values(['symbol', 'date'])
            
            # Add realized volatility
            for col in ['avg_trade_price', 'close_price']:
                if col in df.columns:
                    returns = df.groupby('symbol')[col].pct_change()
                    df[f'{col}_realized_vol_5'] = returns.rolling(5).std() * np.sqrt(252)
                    df[f'{col}_realized_vol_10'] = returns.rolling(10).std() * np.sqrt(252)
                    df[f'{col}_realized_vol_20'] = returns.rolling(20).std() * np.sqrt(252)
            
            # Add volatility of volatility
            for col in ['avg_trade_price', 'close_price']:
                if col in df.columns:
                    returns = df.groupby('symbol')[col].pct_change()
                    vol = returns.rolling(10).std()
                    df[f'{col}_vol_of_vol_10'] = vol.rolling(5).std()
                    df[f'{col}_vol_of_vol_20'] = vol.rolling(10).std()
            
            # Add GARCH-like features
            for col in ['avg_trade_price', 'close_price']:
                if col in df.columns:
                    returns = df.groupby('symbol')[col].pct_change()
                    df[f'{col}_garch_vol_5'] = returns.rolling(5).apply(lambda x: np.sqrt(np.mean(x**2)))
                    df[f'{col}_garch_vol_10'] = returns.rolling(10).apply(lambda x: np.sqrt(np.mean(x**2)))
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding volatility features: {e}")
            return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            df = df.copy()
            
            # Convert timestamp to datetime if it's not already
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Add time-based features
                df['year'] = df['timestamp'].dt.year
                df['month'] = df['timestamp'].dt.month
                df['day'] = df['timestamp'].dt.day
                df['dayofweek'] = df['timestamp'].dt.dayofweek
                df['dayofyear'] = df['timestamp'].dt.dayofyear
                df['week'] = df['timestamp'].dt.isocalendar().week
                df['quarter'] = df['timestamp'].dt.quarter
                
                # Add cyclical features
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
                df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
                df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
                df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
                df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding time features: {e}")
            return df
    
    def _add_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variables for ML models"""
        try:
            df = df.copy()
            
            # Sort by symbol and date
            df = df.sort_values(['symbol', 'date'])
            
            # Add future returns (target variables)
            for col in ['avg_trade_price', 'close_price']:
                if col in df.columns:
                    # 1-day ahead return
                    df[f'{col}_return_1d'] = df.groupby('symbol')[col].pct_change(1).shift(-1)
                    # 3-day ahead return
                    df[f'{col}_return_3d'] = df.groupby('symbol')[col].pct_change(3).shift(-3)
                    # 7-day ahead return
                    df[f'{col}_return_7d'] = df.groupby('symbol')[col].pct_change(7).shift(-7)
            
            # Add future volatility (target variable)
            if 'avg_trade_price' in df.columns:
                returns = df.groupby('symbol')['avg_trade_price'].pct_change()
                df['future_volatility_5d'] = returns.rolling(5).std().shift(-5)
                df['future_volatility_10d'] = returns.rolling(10).std().shift(-10)
            
            # Add future volume (target variable)
            if 'total_volume' in df.columns:
                df['future_volume_1d'] = df.groupby('symbol')['total_volume'].shift(-1)
                df['future_volume_3d'] = df.groupby('symbol')['total_volume'].shift(-3)
                df['future_volume_7d'] = df.groupby('symbol')['total_volume'].shift(-7)
            
            # Add future spread (target variable)
            if 'avg_spread_pct' in df.columns:
                df['future_spread_1d'] = df.groupby('symbol')['avg_spread_pct'].shift(-1)
                df['future_spread_3d'] = df.groupby('symbol')['avg_spread_pct'].shift(-3)
                df['future_spread_7d'] = df.groupby('symbol')['avg_spread_pct'].shift(-7)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding target variables: {e}")
            return df
    
    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(index=series.index, dtype=float)
    
    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator"""
        try:
            ema_fast = series.ewm(span=fast).mean()
            ema_slow = series.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            
            return {
                'macd': macd,
                'signal': signal_line,
                'histogram': histogram
            }
        except:
            return {
                'macd': pd.Series(index=series.index, dtype=float),
                'signal': pd.Series(index=series.index, dtype=float),
                'histogram': pd.Series(index=series.index, dtype=float)
            }
    
    def _calculate_bollinger_bands(self, series: pd.Series, window: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            middle = series.rolling(window=window).mean()
            std = series.rolling(window=window).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            width = upper - lower
            position = (series - lower) / (upper - lower)
            
            return {
                'upper': upper,
                'middle': middle,
                'lower': lower,
                'width': width,
                'position': position
            }
        except:
            return {
                'upper': pd.Series(index=series.index, dtype=float),
                'middle': pd.Series(index=series.index, dtype=float),
                'lower': pd.Series(index=series.index, dtype=float),
                'width': pd.Series(index=series.index, dtype=float),
                'position': pd.Series(index=series.index, dtype=float)
            }
    
    def prepare_ml_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare data for ML training
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            test_size: Test set size
            random_state: Random state for reproducibility
            
        Returns:
            X_train, y_train, X_test, y_test
        """
        try:
            # Remove rows with missing target values
            df_clean = df.dropna(subset=[target_column])
            
            # Select feature columns (exclude target and metadata columns)
            exclude_columns = [
                'symbol', 'date', 'timestamp', target_column,
                'year', 'month', 'day', 'dayofweek', 'dayofyear', 'week', 'quarter'
            ]
            
            feature_columns = [col for col in df_clean.columns if col not in exclude_columns]
            
            # Remove columns with too many missing values
            missing_threshold = 0.5
            feature_columns = [col for col in feature_columns if df_clean[col].isnull().sum() / len(df_clean) < missing_threshold]
            
            # Prepare features and target
            X = df_clean[feature_columns].fillna(0)  # Fill remaining NaN with 0
            y = df_clean[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert back to DataFrame
            X_train_df = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
            X_test_df = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)
            
            self.feature_columns = feature_columns
            self.scalers[target_column] = scaler
            
            logger.info(f"Prepared ML data: {len(X_train_df)} train samples, {len(X_test_df)} test samples, {len(feature_columns)} features")
            
            return X_train_df, y_train, X_test_df, y_test
            
        except Exception as e:
            logger.error(f"Error preparing ML data: {e}")
            return pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()
    
    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_column: str
    ) -> Dict[str, any]:
        """
        Train multiple ML models
        
        Args:
            X_train: Training features
            y_train: Training target
            target_column: Target column name
            
        Returns:
            Dictionary of trained models
        """
        try:
            models = {
                'linear_regression': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=0.1),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            trained_models = {}
            
            for name, model in models.items():
                logger.info(f"Training {name} for {target_column}...")
                
                try:
                    model.fit(X_train, y_train)
                    trained_models[name] = model
                    logger.info(f"✅ {name} trained successfully")
                except Exception as e:
                    logger.error(f"❌ Error training {name}: {e}")
            
            self.models[target_column] = trained_models
            return trained_models
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}
    
    def evaluate_models(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_column: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate trained models
        
        Args:
            X_test: Test features
            y_test: Test target
            target_column: Target column name
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if target_column not in self.models:
                logger.error(f"No models found for {target_column}")
                return {}
            
            results = {}
            
            for name, model in self.models[target_column].items():
                try:
                    y_pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results[name] = {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'rmse': np.sqrt(mse)
                    }
                    
                    logger.info(f"{name} - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {name}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
            return {}
    
    def save_models(self, target_column: str, filename_prefix: str = "ml_model"):
        """Save trained models"""
        try:
            if target_column not in self.models:
                logger.error(f"No models found for {target_column}")
                return False
            
            for name, model in self.models[target_column].items():
                filename = f"{filename_prefix}_{target_column}_{name}.joblib"
                filepath = self.data_dir / filename
                joblib.dump(model, filepath)
                logger.info(f"Saved {name} model to {filepath}")
            
            # Save scaler
            if target_column in self.scalers:
                scaler_filename = f"{filename_prefix}_{target_column}_scaler.joblib"
                scaler_filepath = self.data_dir / scaler_filename
                joblib.dump(self.scalers[target_column], scaler_filepath)
                logger.info(f"Saved scaler to {scaler_filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self, target_column: str, filename_prefix: str = "ml_model"):
        """Load trained models"""
        try:
            models = {}
            
            for name in ['linear_regression', 'ridge', 'lasso', 'random_forest', 'gradient_boosting']:
                filename = f"{filename_prefix}_{target_column}_{name}.joblib"
                filepath = self.data_dir / filename
                
                if filepath.exists():
                    model = joblib.load(filepath)
                    models[name] = model
                    logger.info(f"Loaded {name} model from {filepath}")
            
            if models:
                self.models[target_column] = models
                
                # Load scaler
                scaler_filename = f"{filename_prefix}_{target_column}_scaler.joblib"
                scaler_filepath = self.data_dir / scaler_filename
                
                if scaler_filepath.exists():
                    scaler = joblib.load(scaler_filepath)
                    self.scalers[target_column] = scaler
                    logger.info(f"Loaded scaler from {scaler_filepath}")
                
                return True
            else:
                logger.warning(f"No models found for {target_column}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
