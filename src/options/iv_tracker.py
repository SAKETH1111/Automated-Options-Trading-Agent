"""
Implied Volatility Tracker Module
Calculate IV Rank, IV Percentile, and track volatility metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from loguru import logger

from src.database.models import IndexTickData, OptionsChain, ImpliedVolatility


class IVTracker:
    """
    Track and analyze implied volatility metrics
    Calculate IV Rank, IV Percentile, and historical volatility
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize IV tracker
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        logger.info("IV Tracker initialized")
    
    def calculate_historical_volatility(
        self,
        symbol: str,
        days: int = 30
    ) -> Optional[float]:
        """
        Calculate historical volatility from price data
        
        Args:
            symbol: Symbol to analyze
            days: Number of days for calculation
            
        Returns:
            Annualized historical volatility
        """
        try:
            # Get price data
            cutoff_date = datetime.utcnow() - timedelta(days=days+5)
            
            query = self.db.query(IndexTickData).filter(
                IndexTickData.symbol == symbol,
                IndexTickData.timestamp >= cutoff_date
            ).order_by(IndexTickData.timestamp.asc())
            
            data = query.all()
            
            if len(data) < days:
                logger.warning(f"Insufficient data for HV calculation: {len(data)} days")
                return None
            
            # Calculate daily returns
            prices = [d.price for d in data]
            returns = np.diff(np.log(prices))
            
            # Calculate standard deviation
            std_dev = np.std(returns)
            
            # Annualize (assuming 252 trading days)
            hv = std_dev * np.sqrt(252)
            
            return float(hv)
            
        except Exception as e:
            logger.error(f"Error calculating HV: {e}")
            return None
    
    def get_iv_from_options(
        self,
        symbol: str,
        dte_target: int = 30,
        delta_range: tuple = (0.3, 0.7)
    ) -> Optional[float]:
        """
        Get average IV from options chain
        
        Args:
            symbol: Symbol to analyze
            dte_target: Target days to expiration
            delta_range: Delta range to filter (min, max)
            
        Returns:
            Average implied volatility
        """
        try:
            # Get recent options data
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            
            query = self.db.query(OptionsChain).filter(
                OptionsChain.symbol == symbol,
                OptionsChain.timestamp >= cutoff_time,
                OptionsChain.dte >= dte_target - 7,
                OptionsChain.dte <= dte_target + 7,
                OptionsChain.delta >= delta_range[0],
                OptionsChain.delta <= delta_range[1],
                OptionsChain.implied_volatility.isnot(None)
            )
            
            options = query.all()
            
            if not options:
                logger.warning(f"No options data found for {symbol}")
                return None
            
            # Calculate average IV
            ivs = [opt.implied_volatility for opt in options if opt.implied_volatility]
            
            if not ivs:
                return None
            
            avg_iv = np.mean(ivs)
            
            return float(avg_iv)
            
        except Exception as e:
            logger.error(f"Error getting IV from options: {e}")
            return None
    
    def calculate_iv_rank(
        self,
        symbol: str,
        current_iv: float,
        lookback_days: int = 252
    ) -> Optional[float]:
        """
        Calculate IV Rank
        
        IV Rank = (Current IV - 52-week Low IV) / (52-week High IV - 52-week Low IV) * 100
        
        Args:
            symbol: Symbol to analyze
            current_iv: Current implied volatility
            lookback_days: Days to look back (default: 252 = 1 year)
            
        Returns:
            IV Rank (0-100)
        """
        try:
            # Get historical IV data
            cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
            
            query = self.db.query(ImpliedVolatility).filter(
                ImpliedVolatility.symbol == symbol,
                ImpliedVolatility.timestamp >= cutoff_date
            ).order_by(ImpliedVolatility.timestamp.asc())
            
            iv_history = query.all()
            
            if len(iv_history) < 30:  # Need at least 30 days
                logger.warning(f"Insufficient IV history for {symbol}")
                return None
            
            # Get IV values
            iv_values = [iv.iv_30 for iv in iv_history if iv.iv_30]
            
            if not iv_values:
                return None
            
            iv_min = min(iv_values)
            iv_max = max(iv_values)
            
            if iv_max == iv_min:
                return 50.0  # If no range, return middle
            
            # Calculate IV Rank
            iv_rank = ((current_iv - iv_min) / (iv_max - iv_min)) * 100
            
            # Clamp to 0-100
            iv_rank = max(0, min(100, iv_rank))
            
            return float(iv_rank)
            
        except Exception as e:
            logger.error(f"Error calculating IV Rank: {e}")
            return None
    
    def calculate_iv_percentile(
        self,
        symbol: str,
        current_iv: float,
        lookback_days: int = 252
    ) -> Optional[float]:
        """
        Calculate IV Percentile
        
        IV Percentile = (# of days IV was below current) / (total days) * 100
        
        Args:
            symbol: Symbol to analyze
            current_iv: Current implied volatility
            lookback_days: Days to look back
            
        Returns:
            IV Percentile (0-100)
        """
        try:
            # Get historical IV data
            cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
            
            query = self.db.query(ImpliedVolatility).filter(
                ImpliedVolatility.symbol == symbol,
                ImpliedVolatility.timestamp >= cutoff_date
            ).order_by(ImpliedVolatility.timestamp.asc())
            
            iv_history = query.all()
            
            if len(iv_history) < 30:
                logger.warning(f"Insufficient IV history for {symbol}")
                return None
            
            # Get IV values
            iv_values = [iv.iv_30 for iv in iv_history if iv.iv_30]
            
            if not iv_values:
                return None
            
            # Count days below current IV
            days_below = sum(1 for iv in iv_values if iv < current_iv)
            total_days = len(iv_values)
            
            # Calculate percentile
            iv_percentile = (days_below / total_days) * 100
            
            return float(iv_percentile)
            
        except Exception as e:
            logger.error(f"Error calculating IV Percentile: {e}")
            return None
    
    def calculate_iv_metrics(
        self,
        symbol: str
    ) -> Dict[str, any]:
        """
        Calculate comprehensive IV metrics
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Dictionary with all IV metrics
        """
        try:
            logger.info(f"Calculating IV metrics for {symbol}")
            
            # Get current IV from options
            current_iv_30 = self.get_iv_from_options(symbol, dte_target=30)
            current_iv_60 = self.get_iv_from_options(symbol, dte_target=60)
            current_iv_90 = self.get_iv_from_options(symbol, dte_target=90)
            
            if current_iv_30 is None:
                logger.warning(f"No current IV data for {symbol}")
                return {}
            
            # Calculate historical volatility
            hv_10 = self.calculate_historical_volatility(symbol, days=10)
            hv_20 = self.calculate_historical_volatility(symbol, days=20)
            hv_30 = self.calculate_historical_volatility(symbol, days=30)
            
            # Calculate IV Rank and Percentile
            iv_rank = self.calculate_iv_rank(symbol, current_iv_30)
            iv_percentile = self.calculate_iv_percentile(symbol, current_iv_30)
            
            # Calculate IV/HV ratio
            iv_hv_ratio = None
            if hv_30:
                iv_hv_ratio = current_iv_30 / hv_30
            
            # Get IV statistics from history
            iv_stats = self._get_iv_statistics(symbol)
            
            metrics = {
                'symbol': symbol,
                'timestamp': datetime.utcnow(),
                'iv_30': current_iv_30,
                'iv_60': current_iv_60,
                'iv_90': current_iv_90,
                'hv_10': hv_10,
                'hv_20': hv_20,
                'hv_30': hv_30,
                'iv_rank': iv_rank,
                'iv_percentile': iv_percentile,
                'iv_hv_ratio': iv_hv_ratio,
                'iv_mean': iv_stats.get('mean'),
                'iv_std': iv_stats.get('std'),
                'iv_min': iv_stats.get('min'),
                'iv_max': iv_stats.get('max')
            }
            
            logger.info(f"IV metrics calculated: IV Rank={iv_rank}, IV Percentile={iv_percentile}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating IV metrics: {e}")
            return {}
    
    def _get_iv_statistics(
        self,
        symbol: str,
        lookback_days: int = 252
    ) -> Dict[str, float]:
        """Get IV statistics from historical data"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
            
            query = self.db.query(ImpliedVolatility).filter(
                ImpliedVolatility.symbol == symbol,
                ImpliedVolatility.timestamp >= cutoff_date
            )
            
            iv_history = query.all()
            
            if not iv_history:
                return {}
            
            iv_values = [iv.iv_30 for iv in iv_history if iv.iv_30]
            
            if not iv_values:
                return {}
            
            return {
                'mean': float(np.mean(iv_values)),
                'std': float(np.std(iv_values)),
                'min': float(np.min(iv_values)),
                'max': float(np.max(iv_values))
            }
            
        except Exception as e:
            logger.error(f"Error getting IV statistics: {e}")
            return {}
    
    def store_iv_metrics(
        self,
        metrics: Dict[str, any]
    ) -> bool:
        """
        Store IV metrics in database
        
        Args:
            metrics: IV metrics dictionary
            
        Returns:
            Success status
        """
        try:
            iv_record = ImpliedVolatility(
                symbol=metrics['symbol'],
                timestamp=metrics['timestamp'],
                iv_30=metrics.get('iv_30'),
                iv_60=metrics.get('iv_60'),
                iv_90=metrics.get('iv_90'),
                iv_mean=metrics.get('iv_mean'),
                iv_std=metrics.get('iv_std'),
                iv_min=metrics.get('iv_min'),
                iv_max=metrics.get('iv_max'),
                iv_rank=metrics.get('iv_rank'),
                iv_percentile=metrics.get('iv_percentile'),
                hv_10=metrics.get('hv_10'),
                hv_20=metrics.get('hv_20'),
                hv_30=metrics.get('hv_30'),
                iv_hv_ratio=metrics.get('iv_hv_ratio')
            )
            
            self.db.add(iv_record)
            self.db.commit()
            
            logger.info(f"Stored IV metrics for {metrics['symbol']}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing IV metrics: {e}")
            self.db.rollback()
            return False
    
    def get_iv_regime(
        self,
        iv_rank: Optional[float]
    ) -> str:
        """
        Classify IV regime based on IV Rank
        
        Args:
            iv_rank: IV Rank (0-100)
            
        Returns:
            IV regime classification
        """
        if iv_rank is None:
            return 'UNKNOWN'
        
        if iv_rank >= 75:
            return 'VERY_HIGH'
        elif iv_rank >= 50:
            return 'HIGH'
        elif iv_rank >= 25:
            return 'NORMAL'
        else:
            return 'LOW'
    
    def get_trading_recommendation(
        self,
        iv_rank: Optional[float],
        iv_percentile: Optional[float]
    ) -> Dict[str, str]:
        """
        Get trading recommendation based on IV metrics
        
        Args:
            iv_rank: IV Rank
            iv_percentile: IV Percentile
            
        Returns:
            Trading recommendation
        """
        if iv_rank is None or iv_percentile is None:
            return {
                'action': 'WAIT',
                'strategy': 'Insufficient data'
            }
        
        # High IV = Sell premium
        if iv_rank > 50 and iv_percentile > 50:
            return {
                'action': 'SELL_PREMIUM',
                'strategy': 'Credit spreads, iron condors, covered calls'
            }
        
        # Low IV = Buy options
        elif iv_rank < 30 and iv_percentile < 30:
            return {
                'action': 'BUY_OPTIONS',
                'strategy': 'Debit spreads, long calls/puts, calendars'
            }
        
        # Moderate IV
        else:
            return {
                'action': 'NEUTRAL',
                'strategy': 'Balanced strategies, wait for better setup'
            }

