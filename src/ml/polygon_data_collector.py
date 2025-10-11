"""
Polygon Data Collector for ML Training
Uses Polygon.io API for reliable historical data
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from loguru import logger

from polygon import RESTClient

try:
    from src.ml.options_feature_engineer import OptionsFeatureEngineer
    OPTIONS_FEATURES_AVAILABLE = True
except ImportError:
    OPTIONS_FEATURES_AVAILABLE = False
    logger.warning("Options feature engineer not available")


class PolygonDataCollector:
    """
    Collect historical data using Polygon.io
    Much more reliable than yfinance or Alpaca for historical data
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize Polygon data collector
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        
        # Get API key from environment
        api_key = os.getenv('POLYGON_API_KEY')
        
        if not api_key:
            raise ValueError("POLYGON_API_KEY not found in environment variables!")
        
        self.client = RESTClient(api_key)
        
        # Initialize options feature engineer if available
        if OPTIONS_FEATURES_AVAILABLE:
            try:
                self.options_engineer = OptionsFeatureEngineer(db_session)
                logger.info("Polygon Data Collector initialized with options features")
            except Exception as e:
                logger.warning(f"Options features not available: {e}")
                self.options_engineer = None
        else:
            self.options_engineer = None
            logger.info("Polygon Data Collector initialized")
    
    def collect_training_data(
        self,
        symbols: List[str],
        lookback_days: int = 365,
        timeframe: str = "1Day"
    ) -> pd.DataFrame:
        """
        Collect historical data for training
        
        Args:
            symbols: List of symbols to collect
            lookback_days: Days of historical data
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day, 1Week, 1Month, 3Month, 6Month, 1Year)
            
        Returns:
            DataFrame with historical data
        """
        try:
            logger.info(f"Collecting {lookback_days} days of data for {symbols} from Polygon")
            
            all_data = []
            
            for i, symbol in enumerate(symbols):
                logger.info(f"Fetching data for {symbol}...")
                
                # Add delay between requests to avoid rate limiting
                if i > 0:
                    import time
                    time.sleep(2)  # 2 second delay between symbols
                    logger.debug(f"Rate limit delay (2s)...")
                
                df = self._fetch_bars(symbol, lookback_days, timeframe)
                
                if df.empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Calculate technical indicators
                df = self._calculate_indicators(df)
                
                # Add options features (if available)
                if self.options_engineer:
                    try:
                        df = self.options_engineer.add_options_features(df, symbol)
                        logger.info(f"Added options features for {symbol}")
                    except Exception as e:
                        logger.warning(f"Could not add options features for {symbol}: {e}")
                
                all_data.append(df)
                
                logger.info(f"✅ Collected {len(df)} bars for {symbol}")
            
            if not all_data:
                logger.error("No data collected")
                return pd.DataFrame()
            
            # Combine all symbols
            combined = pd.concat(all_data, ignore_index=True)
            
            logger.info(f"✅ Collected {len(combined):,} total samples across {len(symbols)} symbols")
            
            return combined
            
        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _fetch_bars(
        self,
        symbol: str,
        lookback_days: int,
        timeframe: str
    ) -> pd.DataFrame:
        """Fetch historical bars from Polygon"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Map timeframe to Polygon format
            timespan_map = {
                "1Min": ("minute", 1),
                "5Min": ("minute", 5),
                "15Min": ("minute", 15),
                "1Hour": ("hour", 1),
                "1Day": ("day", 1),
                "1Week": ("week", 1),
                "1Month": ("month", 1),
                "3Month": ("month", 3),
                "6Month": ("month", 6),
                "1Year": ("year", 1),
            }
            
            timespan, multiplier = timespan_map.get(timeframe, ("day", 1))
            
            logger.info(f"Fetching {symbol} from {start_date.date()} to {end_date.date()} ({multiplier} {timespan} bars)")
            
            # Fetch aggregates from Polygon
            aggs = []
            for agg in self.client.list_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                limit=50000
            ):
                aggs.append(agg)
            
            if not aggs:
                logger.warning(f"No data returned from Polygon for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for agg in aggs:
                data.append({
                    'timestamp': datetime.fromtimestamp(agg.timestamp / 1000),  # Polygon uses milliseconds
                    'open': float(agg.open),
                    'high': float(agg.high),
                    'low': float(agg.low),
                    'close': float(agg.close),
                    'volume': float(agg.volume) if agg.volume else 0
                })
            
            df = pd.DataFrame(data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} bars from Polygon for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching bars from Polygon for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Moving averages (skip 100 and 200 day for small datasets)
            windows = [5, 10, 20, 50]
            if len(df) > 200:
                windows.extend([100, 200])
            
            for window in windows:
                df[f'sma_{window}'] = df['close'].rolling(window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            bb_sma = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = bb_sma + (bb_std * 2)
            df['bb_lower'] = bb_sma - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_sma
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
            df['atr_pct'] = df['atr'] / df['close'] * 100
            
            # ADX (Average Directional Index)
            df['adx'] = self._calculate_adx(df)
            
            # Volume indicators
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # Price momentum
            for period in [1, 3, 5, 10, 20]:
                df[f'momentum_{period}'] = df['close'].pct_change(period)
            
            # Volatility
            for window in [5, 10, 20]:
                df[f'volatility_{window}'] = df['returns'].rolling(window).std() * np.sqrt(252)
            
            # Support/Resistance levels (use available data length)
            lookback_window = min(len(df) - 1, 252)  # 52 weeks or available data
            if lookback_window > 20:
                df['high_52w'] = df['high'].rolling(lookback_window).max()
                df['low_52w'] = df['low'].rolling(lookback_window).min()
                df['distance_from_high'] = (df['close'] - df['high_52w']) / df['high_52w']
                df['distance_from_low'] = (df['close'] - df['low_52w']) / df['low_52w']
            else:
                # Not enough data for 52w lookback
                df['high_52w'] = df['high']
                df['low_52w'] = df['low']
                df['distance_from_high'] = 0
                df['distance_from_low'] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX indicator"""
        try:
            # Calculate +DM and -DM
            high_diff = df['high'].diff()
            low_diff = -df['low'].diff()
            
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            # Calculate TR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Smooth using Wilder's method
            atr = tr.rolling(period).mean()
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
            
            # Calculate DX and ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            return adx
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return pd.Series([50] * len(df))
    
    def create_labels(
        self,
        df: pd.DataFrame,
        strategy: str = "bull_put_spread",
        forward_periods: int = 20,
        profit_threshold: float = 0.02,
        loss_threshold: float = -0.05
    ) -> pd.DataFrame:
        """Create labels for supervised learning"""
        try:
            logger.info(f"Creating labels for {strategy} strategy")
            
            # Calculate future returns
            df['future_return'] = df['close'].pct_change(forward_periods).shift(-forward_periods)
            
            # Create binary label
            if strategy in ['bull_put_spread', 'cash_secured_put']:
                df['label'] = (df['future_return'] >= profit_threshold).astype(int)
            elif strategy == 'bear_call_spread':
                df['label'] = (df['future_return'] <= -profit_threshold).astype(int)
            elif strategy == 'iron_condor':
                df['label'] = (
                    (df['future_return'] > loss_threshold) &
                    (df['future_return'] < -loss_threshold)
                ).astype(int)
            else:
                df['label'] = (df['future_return'] > 0).astype(int)
            
            # Create win probability target
            df['win_probability'] = df['future_return'].apply(
                lambda x: 1 / (1 + np.exp(-10 * x))
            )
            
            logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            return df
    
    def split_train_test(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets"""
        try:
            n = len(df)
            train_end = int(n * (1 - test_size - validation_size))
            val_end = int(n * (1 - test_size))
            
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[train_end:val_end].copy()
            test_df = df.iloc[val_end:].copy()
            
            logger.info(f"Split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return df, pd.DataFrame(), pd.DataFrame()



