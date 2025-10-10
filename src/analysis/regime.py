"""
Market Regime Detection Module
Identifies market conditions: trending, ranging, volatile, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger


class MarketRegimeDetector:
    """
    Detect and classify market regimes
    Supports: Trending, Ranging, Volatile, Calm, Bullish, Bearish
    """
    
    def __init__(self):
        """Initialize market regime detector"""
        self.regime_history = []
        logger.info("Market Regime Detector initialized")
    
    def detect_volatility_regime(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        window: int = 20
    ) -> Dict[str, any]:
        """
        Detect volatility regime
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            window: Lookback window
            
        Returns:
            Dictionary with volatility regime info
        """
        prices = df[price_col].tail(window * 2)
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Calculate volatility (standard deviation of returns)
        current_vol = returns.tail(window).std()
        historical_vol = returns.std()
        
        # Calculate percentile
        rolling_vol = returns.rolling(window=window).std()
        vol_percentile = (rolling_vol < current_vol).sum() / len(rolling_vol)
        
        # Classify regime
        if vol_percentile > 0.8:
            regime = 'HIGH_VOLATILITY'
            description = 'High volatility - expect large price swings'
        elif vol_percentile > 0.6:
            regime = 'ELEVATED_VOLATILITY'
            description = 'Elevated volatility - increased price movement'
        elif vol_percentile > 0.4:
            regime = 'NORMAL_VOLATILITY'
            description = 'Normal volatility - typical price movement'
        elif vol_percentile > 0.2:
            regime = 'LOW_VOLATILITY'
            description = 'Low volatility - reduced price movement'
        else:
            regime = 'VERY_LOW_VOLATILITY'
            description = 'Very low volatility - calm market'
        
        return {
            'regime': regime,
            'description': description,
            'current_volatility': current_vol,
            'historical_volatility': historical_vol,
            'percentile': vol_percentile,
            'vol_ratio': current_vol / historical_vol if historical_vol != 0 else 1.0
        }
    
    def detect_trend_regime(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        window: int = 50
    ) -> Dict[str, any]:
        """
        Detect trend regime
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            window: Lookback window
            
        Returns:
            Dictionary with trend regime info
        """
        prices = df[price_col].tail(window)
        
        # Calculate moving averages
        sma_short = prices.rolling(window=10).mean()
        sma_medium = prices.rolling(window=20).mean()
        sma_long = prices.rolling(window=50).mean()
        
        current_price = prices.iloc[-1]
        current_sma_short = sma_short.iloc[-1]
        current_sma_medium = sma_medium.iloc[-1]
        current_sma_long = sma_long.iloc[-1]
        
        # Calculate trend strength using ADX-like logic
        price_changes = prices.diff().abs()
        trend_strength = price_changes.mean() / prices.std() if prices.std() != 0 else 0
        
        # Classify regime
        if current_price > current_sma_short > current_sma_medium > current_sma_long:
            regime = 'STRONG_UPTREND'
            description = 'Strong uptrend - all MAs aligned bullish'
        elif current_price > current_sma_short > current_sma_medium:
            regime = 'UPTREND'
            description = 'Uptrend - price above short/medium MAs'
        elif current_price < current_sma_short < current_sma_medium < current_sma_long:
            regime = 'STRONG_DOWNTREND'
            description = 'Strong downtrend - all MAs aligned bearish'
        elif current_price < current_sma_short < current_sma_medium:
            regime = 'DOWNTREND'
            description = 'Downtrend - price below short/medium MAs'
        else:
            regime = 'RANGING'
            description = 'Ranging market - no clear trend'
        
        return {
            'regime': regime,
            'description': description,
            'trend_strength': trend_strength,
            'price_vs_sma_short': (current_price - current_sma_short) / current_sma_short,
            'price_vs_sma_long': (current_price - current_sma_long) / current_sma_long
        }
    
    def detect_momentum_regime(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        window: int = 14
    ) -> Dict[str, any]:
        """
        Detect momentum regime using RSI-like logic
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            window: Lookback window
            
        Returns:
            Dictionary with momentum regime info
        """
        prices = df[price_col].tail(window * 2)
        
        # Calculate price changes
        changes = prices.diff()
        
        # Calculate gains and losses
        gains = changes.where(changes > 0, 0)
        losses = -changes.where(changes < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=window).mean().iloc[-1]
        avg_loss = losses.rolling(window=window).mean().iloc[-1]
        
        # Calculate RS and RSI
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100
        
        # Classify regime
        if rsi > 70:
            regime = 'OVERBOUGHT'
            description = 'Overbought - potential reversal down'
        elif rsi > 60:
            regime = 'BULLISH_MOMENTUM'
            description = 'Bullish momentum - upward pressure'
        elif rsi > 40:
            regime = 'NEUTRAL_MOMENTUM'
            description = 'Neutral momentum - balanced'
        elif rsi > 30:
            regime = 'BEARISH_MOMENTUM'
            description = 'Bearish momentum - downward pressure'
        else:
            regime = 'OVERSOLD'
            description = 'Oversold - potential reversal up'
        
        return {
            'regime': regime,
            'description': description,
            'rsi': rsi,
            'avg_gain': avg_gain,
            'avg_loss': avg_loss
        }
    
    def detect_volume_regime(
        self,
        df: pd.DataFrame,
        volume_col: str = 'volume',
        window: int = 20
    ) -> Dict[str, any]:
        """
        Detect volume regime
        
        Args:
            df: DataFrame with volume data
            volume_col: Name of volume column
            window: Lookback window
            
        Returns:
            Dictionary with volume regime info
        """
        if volume_col not in df.columns:
            return {
                'regime': 'UNKNOWN',
                'description': 'Volume data not available'
            }
        
        volumes = df[volume_col].tail(window * 2)
        
        # Calculate average volumes
        current_volume = volumes.iloc[-1]
        avg_volume = volumes.tail(window).mean()
        historical_avg = volumes.mean()
        
        # Calculate volume percentile
        vol_percentile = (volumes < current_volume).sum() / len(volumes)
        
        # Classify regime
        if current_volume > avg_volume * 2:
            regime = 'EXTREME_VOLUME'
            description = 'Extreme volume - major market event'
        elif current_volume > avg_volume * 1.5:
            regime = 'HIGH_VOLUME'
            description = 'High volume - increased activity'
        elif current_volume > avg_volume * 0.8:
            regime = 'NORMAL_VOLUME'
            description = 'Normal volume - typical activity'
        else:
            regime = 'LOW_VOLUME'
            description = 'Low volume - reduced activity'
        
        return {
            'regime': regime,
            'description': description,
            'current_volume': current_volume,
            'average_volume': avg_volume,
            'volume_ratio': current_volume / avg_volume if avg_volume != 0 else 1.0,
            'percentile': vol_percentile
        }
    
    def detect_market_hours_regime(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> Dict[str, any]:
        """
        Detect market hours regime (open, mid-day, close)
        
        Args:
            df: DataFrame with timestamp data
            timestamp_col: Name of timestamp column
            
        Returns:
            Dictionary with market hours regime info
        """
        if timestamp_col not in df.columns:
            current_time = datetime.now()
        else:
            current_time = df[timestamp_col].iloc[-1]
            if isinstance(current_time, str):
                current_time = pd.to_datetime(current_time)
        
        # Get hour in Eastern Time (market time)
        hour = current_time.hour
        minute = current_time.minute
        
        # Classify regime
        if hour < 9 or (hour == 9 and minute < 30):
            regime = 'PRE_MARKET'
            description = 'Pre-market - limited liquidity'
        elif hour == 9 and minute < 45:
            regime = 'MARKET_OPEN'
            description = 'Market open - high volatility period'
        elif hour < 12:
            regime = 'MORNING_SESSION'
            description = 'Morning session - active trading'
        elif hour < 14:
            regime = 'MIDDAY'
            description = 'Midday - typically slower'
        elif hour < 15:
            regime = 'AFTERNOON_SESSION'
            description = 'Afternoon session - renewed activity'
        elif hour == 15:
            regime = 'POWER_HOUR'
            description = 'Power hour - increased volatility'
        elif hour < 16:
            regime = 'MARKET_CLOSE'
            description = 'Market close - high volume'
        else:
            regime = 'AFTER_HOURS'
            description = 'After hours - limited liquidity'
        
        return {
            'regime': regime,
            'description': description,
            'hour': hour,
            'minute': minute
        }
    
    def detect_correlation_regime(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        price_col: str = 'price',
        window: int = 50
    ) -> Dict[str, any]:
        """
        Detect correlation regime between two symbols (e.g., SPY and QQQ)
        
        Args:
            df1: DataFrame for first symbol
            df2: DataFrame for second symbol
            price_col: Name of price column
            window: Lookback window
            
        Returns:
            Dictionary with correlation regime info
        """
        # Align dataframes by timestamp
        df1_prices = df1[price_col].tail(window)
        df2_prices = df2[price_col].tail(window)
        
        # Calculate returns
        returns1 = df1_prices.pct_change().dropna()
        returns2 = df2_prices.pct_change().dropna()
        
        # Calculate correlation
        if len(returns1) > 0 and len(returns2) > 0:
            min_len = min(len(returns1), len(returns2))
            correlation = returns1.tail(min_len).corr(returns2.tail(min_len))
        else:
            correlation = 0
        
        # Classify regime
        if correlation > 0.8:
            regime = 'HIGH_CORRELATION'
            description = 'High correlation - moving together'
        elif correlation > 0.5:
            regime = 'MODERATE_CORRELATION'
            description = 'Moderate correlation - some relationship'
        elif correlation > -0.5:
            regime = 'LOW_CORRELATION'
            description = 'Low correlation - independent movement'
        else:
            regime = 'NEGATIVE_CORRELATION'
            description = 'Negative correlation - inverse movement'
        
        return {
            'regime': regime,
            'description': description,
            'correlation': correlation
        }
    
    def detect_overall_regime(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        volume_col: str = 'volume'
    ) -> Dict[str, any]:
        """
        Detect overall market regime combining all factors
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            volume_col: Name of volume column
            
        Returns:
            Dictionary with overall regime classification
        """
        try:
            # Detect all regimes
            volatility = self.detect_volatility_regime(df, price_col)
            trend = self.detect_trend_regime(df, price_col)
            momentum = self.detect_momentum_regime(df, price_col)
            volume = self.detect_volume_regime(df, volume_col)
            market_hours = self.detect_market_hours_regime(df)
            
            # Combine into overall assessment
            result = {
                'volatility': volatility,
                'trend': trend,
                'momentum': momentum,
                'volume': volume,
                'market_hours': market_hours,
                'timestamp': datetime.now()
            }
            
            # Generate trading recommendation
            recommendation = self._generate_recommendation(result)
            result['recommendation'] = recommendation
            
            # Store in history
            self.regime_history.append(result)
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            logger.info(f"Market regime: {trend['regime']}, {volatility['regime']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {}
    
    def _generate_recommendation(self, regime_data: Dict) -> Dict[str, str]:
        """Generate trading recommendation based on regime"""
        trend_regime = regime_data['trend']['regime']
        vol_regime = regime_data['volatility']['regime']
        momentum_regime = regime_data['momentum']['regime']
        
        # Simple recommendation logic
        if 'UPTREND' in trend_regime and momentum_regime in ['BULLISH_MOMENTUM', 'NEUTRAL_MOMENTUM']:
            action = 'BULLISH'
            strategy = 'Consider bull put spreads or cash-secured puts'
        elif 'DOWNTREND' in trend_regime and momentum_regime in ['BEARISH_MOMENTUM', 'NEUTRAL_MOMENTUM']:
            action = 'BEARISH'
            strategy = 'Consider bear call spreads or protective strategies'
        elif trend_regime == 'RANGING' and vol_regime in ['LOW_VOLATILITY', 'NORMAL_VOLATILITY']:
            action = 'NEUTRAL'
            strategy = 'Consider iron condors or range-bound strategies'
        elif vol_regime in ['HIGH_VOLATILITY', 'ELEVATED_VOLATILITY']:
            action = 'CAUTIOUS'
            strategy = 'Reduce position sizes, wait for stability'
        else:
            action = 'WAIT'
            strategy = 'No clear setup, wait for better conditions'
        
        return {
            'action': action,
            'strategy': strategy
        }

