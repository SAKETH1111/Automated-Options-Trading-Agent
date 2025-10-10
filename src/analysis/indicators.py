"""
Technical Indicators Module
Calculates real-time technical indicators for trading analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger


class TechnicalIndicators:
    """
    Calculate technical indicators for market analysis
    Supports: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, and more
    """
    
    def __init__(self):
        """Initialize technical indicators calculator"""
        self.indicators_cache = {}
        logger.info("Technical Indicators module initialized")
    
    # ==================== Moving Averages ====================
    
    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            prices: Price series
            period: Number of periods
            
        Returns:
            SMA series
        """
        return prices.rolling(window=period, min_periods=period).mean()
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            prices: Price series
            period: Number of periods
            
        Returns:
            EMA series
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_wma(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Weighted Moving Average
        
        Args:
            prices: Price series
            period: Number of periods
            
        Returns:
            WMA series
        """
        weights = np.arange(1, period + 1)
        return prices.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    # ==================== Momentum Indicators ====================
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            prices: Price series
            period: RSI period (default: 14)
            
        Returns:
            RSI series (0-100)
        """
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(
        self, 
        prices: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Price series
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_stochastic(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period
            smooth_k: K smoothing period
            smooth_d: D smoothing period
            
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = k.rolling(window=smooth_k).mean()
        d = k.rolling(window=smooth_d).mean()
        
        return k, d
    
    # ==================== Volatility Indicators ====================
    
    def calculate_bollinger_bands(
        self, 
        prices: pd.Series, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Price series
            period: SMA period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle_band = self.calculate_sma(prices, period)
        std = prices.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    def calculate_atr(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period (default: 14)
            
        Returns:
            ATR series
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_keltner_channels(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        atr_multiplier: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Keltner Channels
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: EMA period
            atr_multiplier: ATR multiplier
            
        Returns:
            Tuple of (Upper channel, Middle channel, Lower channel)
        """
        middle = self.calculate_ema(close, period)
        atr = self.calculate_atr(high, low, close, period)
        
        upper = middle + (atr * atr_multiplier)
        lower = middle - (atr * atr_multiplier)
        
        return upper, middle, lower
    
    # ==================== Volume Indicators ====================
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume
        
        Args:
            close: Close prices
            volume: Volume
            
        Returns:
            OBV series
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    def calculate_vwap(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate Volume Weighted Average Price
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            
        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    # ==================== Trend Indicators ====================
    
    def calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Average Directional Index (ADX)
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period
            
        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        # Calculate True Range
        tr = self.calculate_atr(high, low, close, 1)
        
        # Calculate Directional Movement
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        # Smooth the values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    def calculate_supertrend(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
        multiplier: float = 3.0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Supertrend Indicator
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            multiplier: ATR multiplier
            
        Returns:
            Tuple of (Supertrend, Direction)
        """
        atr = self.calculate_atr(high, low, close, period)
        hl_avg = (high + low) / 2
        
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(close)):
            if close.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif close.iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
        
        return supertrend, direction
    
    # ==================== Composite Analysis ====================
    
    def calculate_all_indicators(
        self,
        df: pd.DataFrame,
        price_col: str = 'price'
    ) -> pd.DataFrame:
        """
        Calculate all technical indicators for a dataframe
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Name of price column
            
        Returns:
            DataFrame with all indicators added
        """
        result = df.copy()
        prices = result[price_col]
        
        try:
            # Moving Averages
            result['sma_10'] = self.calculate_sma(prices, 10)
            result['sma_20'] = self.calculate_sma(prices, 20)
            result['sma_50'] = self.calculate_sma(prices, 50)
            result['sma_200'] = self.calculate_sma(prices, 200)
            
            result['ema_12'] = self.calculate_ema(prices, 12)
            result['ema_26'] = self.calculate_ema(prices, 26)
            result['ema_50'] = self.calculate_ema(prices, 50)
            
            # Momentum Indicators
            result['rsi'] = self.calculate_rsi(prices, 14)
            
            macd, signal, hist = self.calculate_macd(prices)
            result['macd'] = macd
            result['macd_signal'] = signal
            result['macd_histogram'] = hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(prices, 20, 2.0)
            result['bb_upper'] = bb_upper
            result['bb_middle'] = bb_middle
            result['bb_lower'] = bb_lower
            result['bb_width'] = (bb_upper - bb_lower) / bb_middle * 100
            
            # If we have OHLC data, calculate more indicators
            if all(col in result.columns for col in ['high', 'low', 'close']):
                high = result['high']
                low = result['low']
                close = result['close']
                
                # ATR
                result['atr'] = self.calculate_atr(high, low, close, 14)
                
                # ADX
                adx, plus_di, minus_di = self.calculate_adx(high, low, close, 14)
                result['adx'] = adx
                result['plus_di'] = plus_di
                result['minus_di'] = minus_di
                
                # Stochastic
                k, d = self.calculate_stochastic(high, low, close)
                result['stoch_k'] = k
                result['stoch_d'] = d
            
            # If we have volume data
            if 'volume' in result.columns:
                result['obv'] = self.calculate_obv(prices, result['volume'])
            
            logger.info(f"Calculated {len([c for c in result.columns if c not in df.columns])} indicators")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return result
    
    def get_indicator_signals(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Get trading signals from indicators
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Dictionary of signals
        """
        signals = {}
        
        try:
            latest = df.iloc[-1]
            
            # RSI Signal
            if 'rsi' in df.columns and pd.notna(latest['rsi']):
                if latest['rsi'] < 30:
                    signals['rsi'] = 'OVERSOLD (Bullish)'
                elif latest['rsi'] > 70:
                    signals['rsi'] = 'OVERBOUGHT (Bearish)'
                else:
                    signals['rsi'] = 'NEUTRAL'
            
            # MACD Signal
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                if pd.notna(latest['macd']) and pd.notna(latest['macd_signal']):
                    if latest['macd'] > latest['macd_signal']:
                        signals['macd'] = 'BULLISH'
                    else:
                        signals['macd'] = 'BEARISH'
            
            # Moving Average Signal
            if all(col in df.columns for col in ['sma_10', 'sma_30', 'price']):
                price = latest['price']
                sma_10 = latest['sma_10']
                sma_30 = latest['sma_30']
                
                if pd.notna(sma_10) and pd.notna(sma_30):
                    if price > sma_10 > sma_30:
                        signals['trend'] = 'STRONG BULLISH'
                    elif price > sma_10:
                        signals['trend'] = 'BULLISH'
                    elif price < sma_10 < sma_30:
                        signals['trend'] = 'STRONG BEARISH'
                    else:
                        signals['trend'] = 'BEARISH'
            
            # Bollinger Bands Signal
            if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'price']):
                price = latest['price']
                bb_upper = latest['bb_upper']
                bb_lower = latest['bb_lower']
                
                if pd.notna(bb_upper) and pd.notna(bb_lower):
                    if price >= bb_upper:
                        signals['bollinger'] = 'OVERBOUGHT'
                    elif price <= bb_lower:
                        signals['bollinger'] = 'OVERSOLD'
                    else:
                        signals['bollinger'] = 'NEUTRAL'
            
            # ADX Trend Strength
            if 'adx' in df.columns and pd.notna(latest['adx']):
                adx = latest['adx']
                if adx > 25:
                    signals['trend_strength'] = 'STRONG TREND'
                elif adx > 20:
                    signals['trend_strength'] = 'TRENDING'
                else:
                    signals['trend_strength'] = 'WEAK/RANGING'
            
        except Exception as e:
            logger.error(f"Error getting indicator signals: {e}")
        
        return signals

