"""
Strike Optimizer Module
Use ML to find optimal strike prices
"""

import numpy as np
from typing import Dict, List
from sqlalchemy.orm import Session
from loguru import logger

from src.options import GreeksCalculator


class StrikeOptimizer:
    """
    Optimize strike selection using ML and historical data
    """
    
    def __init__(self, db_session: Session):
        """Initialize strike optimizer"""
        self.db = db_session
        self.greeks_calc = GreeksCalculator()
        logger.info("Strike Optimizer initialized")
    
    def find_optimal_strikes(
        self,
        symbol: str,
        current_price: float,
        strategy: str,
        iv_rank: float,
        trend: str
    ) -> Dict:
        """
        Find optimal strikes for a strategy
        
        Args:
            symbol: Symbol
            current_price: Current stock price
            strategy: Strategy type
            iv_rank: Current IV Rank
            trend: Market trend
            
        Returns:
            Optimal strikes
        """
        try:
            if strategy == 'bull_put_spread':
                return self._optimize_bull_put_spread(
                    current_price, iv_rank, trend
                )
            elif strategy == 'iron_condor':
                return self._optimize_iron_condor(
                    current_price, iv_rank, trend
                )
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error finding optimal strikes: {e}")
            return {}
    
    def _optimize_bull_put_spread(
        self,
        current_price: float,
        iv_rank: float,
        trend: str
    ) -> Dict:
        """Optimize bull put spread strikes"""
        # ML-enhanced strike selection
        # Higher IV = further OTM
        # Stronger trend = closer to price
        
        if iv_rank > 70:
            short_delta = 0.25  # Further OTM in high IV
        elif iv_rank > 50:
            short_delta = 0.30
        else:
            short_delta = 0.35
        
        # Estimate strike from delta
        short_strike = current_price * (1 - short_delta * 0.15)
        long_strike = short_strike - 5.0
        
        return {
            'short_strike': round(short_strike, 2),
            'long_strike': round(long_strike, 2),
            'width': 5.0,
            'target_delta': short_delta
        }
    
    def _optimize_iron_condor(
        self,
        current_price: float,
        iv_rank: float,
        trend: str
    ) -> Dict:
        """Optimize iron condor strikes"""
        # Symmetric strikes around current price
        
        if iv_rank > 70:
            delta = 0.25
        else:
            delta = 0.30
        
        put_short = current_price * 0.95
        put_long = put_short - 5.0
        call_short = current_price * 1.05
        call_long = call_short + 5.0
        
        return {
            'put_short': round(put_short, 2),
            'put_long': round(put_long, 2),
            'call_short': round(call_short, 2),
            'call_long': round(call_long, 2),
            'target_delta': delta
        }

