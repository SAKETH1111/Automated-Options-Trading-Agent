"""
Pattern Recognition Module
Identifies chart patterns, support/resistance, and price action signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger


class PatternRecognition:
    """
    Identify chart patterns and price action signals
    Supports: Support/Resistance, Trends, Breakouts, Reversals
    """
    
    def __init__(self):
        """Initialize pattern recognition"""
        self.support_resistance_cache = {}
        logger.info("Pattern Recognition module initialized")
    
    # ==================== Support & Resistance ====================
    
    def find_support_resistance(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        window: int = 20,
        num_levels: int = 5
    ) -> Dict[str, List[float]]:
        """
        Find support and resistance levels
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            window: Window for finding local extrema
            num_levels: Number of levels to return
            
        Returns:
            Dictionary with 'support' and 'resistance' levels
        """
        prices = df[price_col].values
        
        # Find local minima (support) and maxima (resistance)
        support_levels = []
        resistance_levels = []
        
        for i in range(window, len(prices) - window):
            # Check if local minimum
            if prices[i] == min(prices[i-window:i+window+1]):
                support_levels.append(prices[i])
            
            # Check if local maximum
            if prices[i] == max(prices[i-window:i+window+1]):
                resistance_levels.append(prices[i])
        
        # Cluster nearby levels
        support_levels = self._cluster_levels(support_levels, num_levels)
        resistance_levels = self._cluster_levels(resistance_levels, num_levels)
        
        return {
            'support': sorted(support_levels),
            'resistance': sorted(resistance_levels, reverse=True)
        }
    
    def _cluster_levels(self, levels: List[float], num_clusters: int) -> List[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        
        if len(levels) <= num_clusters:
            return levels
        
        # Simple clustering by grouping nearby values
        threshold = np.std(levels) * 0.5
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if level - current_cluster[-1] <= threshold:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clustered.append(np.mean(current_cluster))
        
        # Return top N by frequency
        return sorted(clustered)[:num_clusters]
    
    def check_near_support_resistance(
        self,
        current_price: float,
        levels: Dict[str, List[float]],
        tolerance: float = 0.01
    ) -> Dict[str, any]:
        """
        Check if price is near support or resistance
        
        Args:
            current_price: Current price
            levels: Support/resistance levels
            tolerance: Distance tolerance (as percentage)
            
        Returns:
            Dictionary with nearest levels and distances
        """
        result = {
            'near_support': False,
            'near_resistance': False,
            'nearest_support': None,
            'nearest_resistance': None,
            'support_distance': None,
            'resistance_distance': None
        }
        
        # Check support levels
        if levels['support']:
            nearest_support = min(levels['support'], key=lambda x: abs(x - current_price))
            distance = abs(current_price - nearest_support) / current_price
            
            result['nearest_support'] = nearest_support
            result['support_distance'] = distance
            result['near_support'] = distance <= tolerance
        
        # Check resistance levels
        if levels['resistance']:
            nearest_resistance = min(levels['resistance'], key=lambda x: abs(x - current_price))
            distance = abs(current_price - nearest_resistance) / current_price
            
            result['nearest_resistance'] = nearest_resistance
            result['resistance_distance'] = distance
            result['near_resistance'] = distance <= tolerance
        
        return result
    
    # ==================== Trend Detection ====================
    
    def detect_trend(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        window: int = 20
    ) -> Dict[str, any]:
        """
        Detect current trend direction and strength
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            window: Lookback window
            
        Returns:
            Dictionary with trend information
        """
        prices = df[price_col].tail(window)
        
        # Linear regression for trend
        x = np.arange(len(prices))
        y = prices.values
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Calculate R-squared for trend strength
        y_pred = np.poly1d(np.polyfit(x, y, 1))(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Determine trend direction
        if slope > 0:
            direction = 'UPTREND'
        elif slope < 0:
            direction = 'DOWNTREND'
        else:
            direction = 'SIDEWAYS'
        
        # Determine strength
        if r_squared > 0.7:
            strength = 'STRONG'
        elif r_squared > 0.4:
            strength = 'MODERATE'
        else:
            strength = 'WEAK'
        
        # Calculate trend angle
        angle = np.degrees(np.arctan(slope))
        
        return {
            'direction': direction,
            'strength': strength,
            'slope': slope,
            'r_squared': r_squared,
            'angle': angle,
            'description': f"{strength} {direction}"
        }
    
    def detect_higher_highs_lows(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        window: int = 5
    ) -> Dict[str, bool]:
        """
        Detect higher highs/lows or lower highs/lows
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            window: Window for finding peaks/troughs
            
        Returns:
            Dictionary indicating pattern presence
        """
        prices = df[price_col].values
        
        # Find peaks and troughs
        peaks = []
        troughs = []
        
        for i in range(window, len(prices) - window):
            if prices[i] == max(prices[i-window:i+window+1]):
                peaks.append((i, prices[i]))
            if prices[i] == min(prices[i-window:i+window+1]):
                troughs.append((i, prices[i]))
        
        result = {
            'higher_highs': False,
            'higher_lows': False,
            'lower_highs': False,
            'lower_lows': False,
            'pattern': 'UNKNOWN'
        }
        
        # Check for patterns
        if len(peaks) >= 2:
            result['higher_highs'] = peaks[-1][1] > peaks[-2][1]
            result['lower_highs'] = peaks[-1][1] < peaks[-2][1]
        
        if len(troughs) >= 2:
            result['higher_lows'] = troughs[-1][1] > troughs[-2][1]
            result['lower_lows'] = troughs[-1][1] < troughs[-2][1]
        
        # Determine overall pattern
        if result['higher_highs'] and result['higher_lows']:
            result['pattern'] = 'UPTREND'
        elif result['lower_highs'] and result['lower_lows']:
            result['pattern'] = 'DOWNTREND'
        elif result['higher_highs'] and result['lower_lows']:
            result['pattern'] = 'EXPANDING'
        elif result['lower_highs'] and result['higher_lows']:
            result['pattern'] = 'CONTRACTING'
        
        return result
    
    # ==================== Breakout Detection ====================
    
    def detect_breakout(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        volume_col: str = 'volume',
        window: int = 20,
        threshold: float = 0.02
    ) -> Dict[str, any]:
        """
        Detect price breakouts
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            volume_col: Name of volume column
            window: Lookback window
            threshold: Breakout threshold (as percentage)
            
        Returns:
            Dictionary with breakout information
        """
        recent_data = df.tail(window)
        current_price = df[price_col].iloc[-1]
        
        # Calculate range
        high = recent_data[price_col].max()
        low = recent_data[price_col].min()
        range_size = high - low
        
        # Check for breakout
        breakout_up = current_price > high * (1 + threshold)
        breakout_down = current_price < low * (1 - threshold)
        
        # Check volume confirmation
        volume_confirmed = False
        if volume_col in df.columns:
            avg_volume = recent_data[volume_col].mean()
            current_volume = df[volume_col].iloc[-1]
            volume_confirmed = current_volume > avg_volume * 1.5
        
        result = {
            'breakout': breakout_up or breakout_down,
            'direction': 'UP' if breakout_up else ('DOWN' if breakout_down else 'NONE'),
            'volume_confirmed': volume_confirmed,
            'range_high': high,
            'range_low': low,
            'range_size': range_size,
            'current_price': current_price
        }
        
        return result
    
    def detect_consolidation(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        window: int = 20,
        threshold: float = 0.03
    ) -> Dict[str, any]:
        """
        Detect price consolidation (tight range)
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            window: Lookback window
            threshold: Consolidation threshold
            
        Returns:
            Dictionary with consolidation information
        """
        recent_data = df.tail(window)
        prices = recent_data[price_col]
        
        # Calculate volatility
        volatility = prices.std() / prices.mean()
        
        # Check if consolidating
        is_consolidating = volatility < threshold
        
        # Calculate range
        high = prices.max()
        low = prices.min()
        range_pct = (high - low) / low
        
        return {
            'consolidating': is_consolidating,
            'volatility': volatility,
            'range_pct': range_pct,
            'high': high,
            'low': low,
            'duration': len(recent_data)
        }
    
    # ==================== Reversal Patterns ====================
    
    def detect_reversal_signals(
        self,
        df: pd.DataFrame,
        price_col: str = 'price'
    ) -> Dict[str, any]:
        """
        Detect potential reversal signals
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            
        Returns:
            Dictionary with reversal signals
        """
        if len(df) < 10:
            return {'reversal_detected': False}
        
        prices = df[price_col].tail(10).values
        
        # Check for V-shaped reversal (bottom)
        v_bottom = (
            prices[-5] < prices[-6] and
            prices[-4] < prices[-5] and
            prices[-3] > prices[-4] and
            prices[-2] > prices[-3] and
            prices[-1] > prices[-2]
        )
        
        # Check for inverted V (top)
        v_top = (
            prices[-5] > prices[-6] and
            prices[-4] > prices[-5] and
            prices[-3] < prices[-4] and
            prices[-2] < prices[-3] and
            prices[-1] < prices[-2]
        )
        
        # Check for double bottom
        double_bottom = False
        if len(prices) >= 10:
            lows = [i for i in range(1, len(prices)-1) 
                   if prices[i] < prices[i-1] and prices[i] < prices[i+1]]
            if len(lows) >= 2:
                last_two_lows = [prices[i] for i in lows[-2:]]
                double_bottom = abs(last_two_lows[0] - last_two_lows[1]) / last_two_lows[0] < 0.02
        
        # Check for double top
        double_top = False
        if len(prices) >= 10:
            highs = [i for i in range(1, len(prices)-1)
                    if prices[i] > prices[i-1] and prices[i] > prices[i+1]]
            if len(highs) >= 2:
                last_two_highs = [prices[i] for i in highs[-2:]]
                double_top = abs(last_two_highs[0] - last_two_highs[1]) / last_two_highs[0] < 0.02
        
        result = {
            'reversal_detected': v_bottom or v_top or double_bottom or double_top,
            'v_bottom': v_bottom,
            'v_top': v_top,
            'double_bottom': double_bottom,
            'double_top': double_top
        }
        
        if v_bottom or double_bottom:
            result['type'] = 'BULLISH_REVERSAL'
        elif v_top or double_top:
            result['type'] = 'BEARISH_REVERSAL'
        else:
            result['type'] = 'NONE'
        
        return result
    
    # ==================== Composite Analysis ====================
    
    def analyze_all_patterns(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        volume_col: str = 'volume'
    ) -> Dict[str, any]:
        """
        Perform comprehensive pattern analysis
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            volume_col: Name of volume column
            
        Returns:
            Dictionary with all pattern analysis
        """
        try:
            current_price = df[price_col].iloc[-1]
            
            # Find support/resistance
            sr_levels = self.find_support_resistance(df, price_col)
            sr_check = self.check_near_support_resistance(current_price, sr_levels)
            
            # Detect trend
            trend = self.detect_trend(df, price_col)
            hh_hl = self.detect_higher_highs_lows(df, price_col)
            
            # Detect breakout
            breakout = self.detect_breakout(df, price_col, volume_col)
            consolidation = self.detect_consolidation(df, price_col)
            
            # Detect reversals
            reversals = self.detect_reversal_signals(df, price_col)
            
            result = {
                'current_price': current_price,
                'support_resistance': sr_levels,
                'near_levels': sr_check,
                'trend': trend,
                'higher_highs_lows': hh_hl,
                'breakout': breakout,
                'consolidation': consolidation,
                'reversals': reversals,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Pattern analysis complete: {trend['description']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return {}

