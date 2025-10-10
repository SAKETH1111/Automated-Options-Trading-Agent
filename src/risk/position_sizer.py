"""Position sizing logic"""

import math
from typing import Dict, Optional

from loguru import logger


class PositionSizer:
    """Calculate optimal position size based on risk parameters"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.default_contracts = config.get("position_sizing", {}).get("default_contracts", 1)
        self.fixed_risk_per_trade_pct = config.get("position_sizing", {}).get("fixed_risk_per_trade_pct", 1.0)
        self.use_kelly = config.get("position_sizing", {}).get("kelly_criterion", False)
    
    def calculate_size(
        self,
        signal: Dict,
        account_balance: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> int:
        """
        Calculate position size (number of contracts)
        
        Args:
            signal: Trading signal with max_loss, max_profit
            account_balance: Current account balance
            win_rate: Historical win rate (for Kelly)
            avg_win: Average win size (for Kelly)
            avg_loss: Average loss size (for Kelly)
        
        Returns:
            Number of contracts to trade
        """
        try:
            max_loss = signal.get("max_loss", 0)
            max_profit = signal.get("max_profit", 0)
            
            if max_loss <= 0:
                logger.warning("Max loss is 0, using default size")
                return self.default_contracts
            
            if self.use_kelly and win_rate and avg_win and avg_loss:
                # Kelly Criterion sizing
                size = self._kelly_sizing(
                    account_balance, max_loss, max_profit,
                    win_rate, avg_win, avg_loss
                )
            else:
                # Fixed risk sizing
                size = self._fixed_risk_sizing(account_balance, max_loss)
            
            # Ensure minimum of 1 contract
            return max(1, size)
        
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.default_contracts
    
    def _fixed_risk_sizing(self, account_balance: float, max_loss_per_contract: float) -> int:
        """
        Fixed risk per trade
        Risk X% of account per trade
        """
        try:
            risk_amount = account_balance * (self.fixed_risk_per_trade_pct / 100)
            contracts = math.floor(risk_amount / max_loss_per_contract)
            
            logger.debug(f"Fixed risk sizing: ${risk_amount:.2f} risk / ${max_loss_per_contract:.2f} per contract = {contracts} contracts")
            
            return max(1, contracts)
        
        except Exception as e:
            logger.error(f"Error in fixed risk sizing: {e}")
            return self.default_contracts
    
    def _kelly_sizing(
        self,
        account_balance: float,
        max_loss_per_contract: float,
        max_profit_per_contract: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> int:
        """
        Kelly Criterion sizing (use fractional Kelly for safety)
        
        Kelly % = (W * R - L) / R
        Where:
        - W = win rate
        - R = avg win / avg loss ratio
        - L = loss rate (1 - W)
        """
        try:
            if avg_loss == 0:
                return self.default_contracts
            
            win_loss_ratio = avg_win / abs(avg_loss)
            loss_rate = 1 - win_rate
            
            kelly_pct = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
            
            # Use fractional Kelly (25% of Kelly) for safety
            fractional_kelly = kelly_pct * 0.25
            
            # Ensure positive and reasonable
            fractional_kelly = max(0.01, min(fractional_kelly, 0.05))  # Cap at 5%
            
            risk_amount = account_balance * fractional_kelly
            contracts = math.floor(risk_amount / max_loss_per_contract)
            
            logger.debug(f"Kelly sizing: {fractional_kelly*100:.2f}% of account = {contracts} contracts")
            
            return max(1, contracts)
        
        except Exception as e:
            logger.error(f"Error in Kelly sizing: {e}")
            return self.default_contracts
    
    def validate_size(
        self,
        contracts: int,
        max_loss_per_contract: float,
        account_balance: float,
        max_position_size_pct: float = 20.0
    ) -> int:
        """Validate and adjust position size if needed"""
        try:
            total_risk = contracts * max_loss_per_contract
            max_allowed_risk = account_balance * (max_position_size_pct / 100)
            
            if total_risk > max_allowed_risk:
                # Reduce size
                adjusted_contracts = math.floor(max_allowed_risk / max_loss_per_contract)
                logger.warning(f"Position size adjusted from {contracts} to {adjusted_contracts} due to risk limits")
                return max(1, adjusted_contracts)
            
            return contracts
        
        except Exception as e:
            logger.error(f"Error validating size: {e}")
            return contracts


