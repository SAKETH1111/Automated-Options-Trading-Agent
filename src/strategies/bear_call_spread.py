"""
Bear Call Spread Strategy
Bearish credit spread strategy
"""

from typing import Dict, Optional
from .base import BaseStrategy


class BearCallSpread(BaseStrategy):
    """
    Bear Call Spread: Sell call, buy higher call
    Profits when price stays below short call
    """
    
    def __init__(self):
        super().__init__(
            name="Bear Call Spread",
            description="Bearish credit spread - profit when price stays below short call"
        )
        
        # Strategy parameters
        self.target_delta = 0.30  # Target delta for short call
        self.target_dte = 35  # Days to expiration
        self.min_credit = 0.25  # Minimum credit to collect
        self.max_width = 10.0  # Maximum spread width
        self.min_iv_rank = 50  # Minimum IV Rank
    
    def generate_signal(self, market_data: Dict) -> Optional[Dict]:
        """Generate bear call spread signal"""
        # Check if conditions are right
        if not self._check_conditions(market_data):
            return None
        
        # Find optimal strikes
        current_price = market_data.get('price', 0)
        
        # Short call above current price
        short_strike = current_price * 1.05  # 5% OTM
        long_strike = short_strike + 5.0  # $5 width
        
        # Estimate credit
        credit = self.min_credit + (5.0 * 0.05)
        
        max_profit = credit * 100
        max_loss = (5.0 - credit) * 100
        
        return {
            'strategy': 'bear_call_spread',
            'action': 'ENTRY',
            'short_strike': short_strike,
            'long_strike': long_strike,
            'credit': credit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'confidence': 0.70
        }
    
    def _check_conditions(self, market_data: Dict) -> bool:
        """Check if market conditions are suitable"""
        # Need bearish or neutral trend
        trend = market_data.get('trend', 'NEUTRAL')
        if trend == 'STRONG_UPTREND':
            return False
        
        # Need high IV
        iv_rank = market_data.get('iv_rank', 0)
        if iv_rank < self.min_iv_rank:
            return False
        
        return True

