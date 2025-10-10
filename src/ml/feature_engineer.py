"""
Feature Engineering Module
Create ML features from market data and technical indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from loguru import logger

from src.database.models import IndexTickData, TechnicalIndicators, MarketRegime, ImpliedVolatility


class FeatureEngineer:
    """
    Engineer features for machine learning models
    Combines price data, technical indicators, and market regime
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize feature engineer
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        logger.info("Feature Engineer initialized")
    
    def create_features(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set
        
        Args:
            symbol: Symbol to create features for
            lookback_hours: Hours of data to use
            
        Returns:
            DataFrame with features
        """
        try:
            logger.info(f"Creating features for {symbol}")
            
            # Get base price data
            df = self._get_price_data(symbol, lookback_hours)
            
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return pd.DataFrame()
            
            # Add price-based features
            df = self._add_price_features(df)
            
            # Add technical indicator features
            df = self._add_technical_features(df, symbol)
            
            # Add regime features
            df = self._add_regime_features(df, symbol)
            
            # Add IV features
            df = self._add_iv_features(df, symbol)
            
            # Add time-based features
            df = self._add_time_features(df)
            
            # Add lag features
            df = self._add_lag_features(df)
            
            # Drop NaN values
            df = df.dropna()
            
            logger.info(f"Created {len(df.columns)} features for {len(df)} samples")
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return pd.DataFrame()
    
    def _get_price_data(
        self,
        symbol: str,
        lookback_hours: int
    ) -> pd.DataFrame:
        """Get price data from database"""
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        query = self.db.query(IndexTickData).filter(
            IndexTickData.symbol == symbol,
            IndexTickData.timestamp >= cutoff
        ).order_by(IndexTickData.timestamp.asc())
        
        data = query.all()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame([{
            'timestamp': d.timestamp,
            'price': d.price,
            'volume': d.volume or 0,
            'bid': d.bid,
            'ask': d.ask,
            'spread': d.spread
        } for d in data])
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Returns
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        
        # Price changes
        df['price_change'] = df['price'].diff()
        df['price_change_pct'] = df['price'].pct_change() * 100
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            df[f'price_mean_{window}'] = df['price'].rolling(window).mean()
            df[f'price_std_{window}'] = df['price'].rolling(window).std()
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
        
        # Price momentum
        df['momentum_5'] = df['price'] / df['price'].shift(5) - 1
        df['momentum_10'] = df['price'] / df['price'].shift(10) - 1
        df['momentum_20'] = df['price'] / df['price'].shift(20) - 1
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_10']
        
        # Spread features
        df['spread_pct'] = (df['ask'] - df['bid']) / df['price'] * 100
        df['spread_ma_10'] = df['spread'].rolling(10).mean()
        
        return df
    
    def _add_technical_features(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """Add technical indicator features"""
        # Calculate basic indicators
        prices = df['price']
        
        # Moving averages
        df['sma_10'] = prices.rolling(10).mean()
        df['sma_20'] = prices.rolling(20).mean()
        df['sma_50'] = prices.rolling(50).mean()
        df['ema_12'] = prices.ewm(span=12).mean()
        df['ema_26'] = prices.ewm(span=26).mean()
        
        # Price vs MA
        df['price_vs_sma10'] = (df['price'] - df['sma_10']) / df['sma_10']
        df['price_vs_sma20'] = (df['price'] - df['sma_20']) / df['sma_20']
        
        # MA crossovers
        df['sma_10_20_cross'] = (df['sma_10'] > df['sma_20']).astype(int)
        
        # RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = prices.rolling(20).mean()
        bb_std = prices.rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _add_regime_features(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """Add market regime features"""
        # Get latest regime
        latest_regime = self.db.query(MarketRegime).filter(
            MarketRegime.symbol == symbol
        ).order_by(MarketRegime.timestamp.desc()).first()
        
        if latest_regime:
            # Encode regime as numeric
            regime_map = {
                'STRONG_UPTREND': 2,
                'UPTREND': 1,
                'RANGING': 0,
                'DOWNTREND': -1,
                'STRONG_DOWNTREND': -2
            }
            
            df['trend_regime_encoded'] = regime_map.get(latest_regime.trend_regime, 0)
            df['volatility_percentile'] = latest_regime.volatility_percentile or 0.5
            df['rsi_value'] = latest_regime.rsi_value or 50
        else:
            df['trend_regime_encoded'] = 0
            df['volatility_percentile'] = 0.5
            df['rsi_value'] = 50
        
        return df
    
    def _add_iv_features(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """Add IV features"""
        # Get latest IV metrics
        latest_iv = self.db.query(ImpliedVolatility).filter(
            ImpliedVolatility.symbol == symbol
        ).order_by(ImpliedVolatility.timestamp.desc()).first()
        
        if latest_iv:
            df['iv_rank'] = latest_iv.iv_rank or 50
            df['iv_percentile'] = latest_iv.iv_percentile or 50
            df['iv_hv_ratio'] = latest_iv.iv_hv_ratio or 1.0
        else:
            df['iv_rank'] = 50
            df['iv_percentile'] = 50
            df['iv_hv_ratio'] = 1.0
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Market session
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        df['is_morning'] = ((df['hour'] >= 9) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 16)).astype(int)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        # Lag price and returns
        for lag in [1, 2, 3, 5, 10]:
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        return df
    
    def create_target_variable(
        self,
        df: pd.DataFrame,
        target_type: str = 'direction',
        forward_periods: int = 10
    ) -> pd.DataFrame:
        """
        Create target variable for supervised learning
        
        Args:
            df: DataFrame with features
            target_type: 'direction', 'returns', or 'volatility'
            forward_periods: Periods to look forward
            
        Returns:
            DataFrame with target variable
        """
        if target_type == 'direction':
            # Predict if price will go up or down
            df['future_price'] = df['price'].shift(-forward_periods)
            df['target'] = (df['future_price'] > df['price']).astype(int)
        
        elif target_type == 'returns':
            # Predict future returns
            df['target'] = df['price'].pct_change(forward_periods).shift(-forward_periods)
        
        elif target_type == 'volatility':
            # Predict future volatility
            df['target'] = df['returns'].rolling(forward_periods).std().shift(-forward_periods)
        
        return df

