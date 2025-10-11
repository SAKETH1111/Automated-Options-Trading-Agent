"""
Historical Data Collector for ML Training (with yfinance backup)
Collects and prepares historical market data for model training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from loguru import logger

from src.brokers.alpaca_client import AlpacaClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


class HistoricalDataCollector:
    """
    Collect historical data for ML training
    Fetches price data, calculates indicators, and prepares features
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize data collector
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        self.alpaca = AlpacaClient()
        logger.info("Historical Data Collector initialized")
    
    def collect_training_data(
        self,
        symbols: List[str],
        lookback_days: int = 365,
        timeframe: str = "5Min"
    ) -> pd.DataFrame:
        """
        Collect historical data for training
        
        Args:
            symbols: List of symbols to collect
            lookback_days: Days of historical data
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            
        Returns:
            DataFrame with historical data
        """
        try:
            logger.info(f"Collecting {lookback_days} days of data for {symbols}")
            
            all_data = []
            
            for symbol in symbols:
                logger.info(f"Fetching data for {symbol}...")
                
                # Try Alpaca first, fallback to yfinance
                df = self._fetch_historical_bars_with_fallback(symbol, lookback_days, timeframe)
                
                if df.empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Calculate technical indicators
                df = self._calculate_indicators(df)
                
                all_data.append(df)
            
            if not all_data:
                logger.error("No data collected")
                return pd.DataFrame()
            
            # Combine all symbols
            combined = pd.concat(all_data, ignore_index=True)
            
            logger.info(f"Collected {len(combined)} total samples across {len(symbols)} symbols")
            
            return combined
            
        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            return pd.DataFrame()
    
    def _fetch_historical_bars_with_fallback(
        self,
        symbol: str,
        lookback_days: int,
        timeframe: str
    ) -> pd.DataFrame:
        """Fetch historical bars from Alpaca, fallback to yfinance"""
        # Try Alpaca first
        df = self._fetch_from_alpaca(symbol, lookback_days, timeframe)
        
        if not df.empty:
            return df
        
        # Fallback to yfinance
        logger.info(f"Alpaca failed, trying yfinance for {symbol}...")
        return self._fetch_from_yfinance(symbol, lookback_days, timeframe)
    
    def _fetch_from_alpaca(
        self,
        symbol: str,
        lookback_days: int,
        timeframe: str
    ) -> pd.DataFrame:
        """Fetch historical bars from Alpaca"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Map timeframe string to Alpaca TimeFrame
            timeframe_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day,
            }
            
            tf = timeframe_map.get(timeframe, TimeFrame(5, TimeFrameUnit.Minute))
            
            # Fetch bars
            bars = self.alpaca.get_historical_bars(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=tf
            )
            
            if not bars:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(bars)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.warning(f"Alpaca fetch failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_from_yfinance(
        self,
        symbol: str,
        lookback_days: int,
        timeframe: str
    ) -> pd.DataFrame:
        """Fetch historical data from yfinance as backup"""
        try:
            import yfinance as yf
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Map timeframe to yfinance interval
            interval_map = {
                "1Min": "1m",
                "5Min": "5m",
                "15Min": "15m",
                "1Hour": "1h",
                "1Day": "1d",
            }
            
            interval = interval_map.get(timeframe, "1d")
            
            # yfinance has limits on intraday data (7 days for 1m/5m)
            # For daily data, no limits
            if interval in ["1m", "5m", "15m"] and lookback_days > 7:
                logger.info(f"Limiting lookback to 7 days for {interval} interval")
                start_date = end_date - timedelta(days=7)
            elif interval == "1h" and lookback_days > 730:
                start_date = end_date - timedelta(days=730)
            
            logger.info(f"Downloading {symbol} from {start_date.date()} to {end_date.date()} ({interval} interval)")
            
            # Try using download() method instead of Ticker().history()
            # This is more reliable
            import time
            time.sleep(1)  # Rate limit protection
            
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                show_errors=False
            )
            
            if df is None or df.empty:
                logger.warning(f"yfinance download returned empty for {symbol}")
                return pd.DataFrame()
            
            # Rename columns to match Alpaca format
            df = df.reset_index()
            df.columns = [col.lower() for col in df.columns]
            
            # Rename 'date' or 'datetime' to 'timestamp'
            if 'date' in df.columns:
                df.rename(columns={'date': 'timestamp'}, inplace=True)
            elif 'datetime' in df.columns:
                df.rename(columns={'datetime': 'timestamp'}, inplace=True)
            
            # Keep only needed columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing columns for {symbol}: {missing_cols}")
                logger.info(f"Available columns: {list(df.columns)}")
                return pd.DataFrame()
            
            df = df[required_cols]
            
            logger.info(f"✅ Fetched {len(df)} bars from yfinance for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"yfinance fetch failed for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Moving averages
            for window in [5, 10, 20, 50, 100, 200]:
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
            
            # Support/Resistance levels
            df['high_52w'] = df['high'].rolling(252).max()
            df['low_52w'] = df['low'].rolling(252).min()
            df['distance_from_high'] = (df['close'] - df['high_52w']) / df['high_52w']
            df['distance_from_low'] = (df['close'] - df['low_52w']) / df['low_52w']
            
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
        """
        Create labels for supervised learning
        
        Args:
            df: DataFrame with price data
            strategy: Strategy type to optimize for
            forward_periods: Periods to look forward
            profit_threshold: Profit threshold for 'good' trades
            loss_threshold: Loss threshold for 'bad' trades
            
        Returns:
            DataFrame with labels
        """
        try:
            logger.info(f"Creating labels for {strategy} strategy")
            
            # Calculate future returns
            df['future_return'] = df['close'].pct_change(forward_periods).shift(-forward_periods)
            
            # Create binary label (1 = good trade, 0 = bad trade)
            if strategy in ['bull_put_spread', 'cash_secured_put']:
                # For bullish strategies, profit if price stays above or goes up
                df['label'] = (df['future_return'] >= profit_threshold).astype(int)
                
            elif strategy == 'bear_call_spread':
                # For bearish strategies, profit if price stays below or goes down
                df['label'] = (df['future_return'] <= -profit_threshold).astype(int)
                
            elif strategy == 'iron_condor':
                # For neutral strategies, profit if price stays in range
                df['label'] = (
                    (df['future_return'] > loss_threshold) &
                    (df['future_return'] < -loss_threshold)
                ).astype(int)
            
            else:
                # Default: profit if price goes up
                df['label'] = (df['future_return'] > 0).astype(int)
            
            # Create multi-class labels (strong signal, weak signal, no signal)
            conditions = [
                df['future_return'] >= profit_threshold * 2,  # Strong bullish
                df['future_return'] >= profit_threshold,      # Weak bullish
                df['future_return'] <= loss_threshold,        # Bearish
            ]
            choices = [2, 1, 0]
            df['label_multiclass'] = np.select(conditions, choices, default=1)
            
            # Create win probability target (continuous)
            df['win_probability'] = df['future_return'].apply(
                lambda x: 1 / (1 + np.exp(-10 * x))  # Sigmoid transform
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
        """
        Split data into train/validation/test sets
        Uses time-based split (no shuffling)
        
        Args:
            df: DataFrame to split
            test_size: Fraction for test set
            validation_size: Fraction for validation set
            
        Returns:
            train_df, val_df, test_df
        """
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

