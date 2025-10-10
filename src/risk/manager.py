"""Risk management and portfolio constraints"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from loguru import logger
from sqlalchemy.orm import Session

from src.config.settings import get_config
from src.database.models import Trade, PerformanceMetric
from src.database.session import get_db


class RiskManager:
    """Manage risk across portfolio"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config()
        self.risk_config = self.config.get("trading", {}).get("risk", {})
        
        # Risk limits
        self.max_daily_loss_pct = self.risk_config.get("max_daily_loss_pct", 5.0)
        self.max_position_size_pct = self.risk_config.get("max_position_size_pct", 20.0)
        self.max_trades_per_day = self.risk_config.get("max_trades_per_day", 10)
        self.max_positions_per_symbol = self.risk_config.get("max_positions_per_symbol", 2)
        self.max_portfolio_heat = self.risk_config.get("max_portfolio_heat", 30.0)
        
        self.db = get_db()
        
        logger.info("Risk Manager initialized")
    
    def can_open_trade(
        self,
        signal: Dict,
        account_balance: float,
        current_positions: List[Dict]
    ) -> Dict:
        """
        Check if a trade can be opened given risk constraints
        
        Returns:
            Dict with "allowed" (bool) and "reason" (str)
        """
        try:
            symbol = signal.get("symbol")
            max_loss = signal.get("max_loss", 0)
            
            # Check 1: Daily trade limit
            if not self._check_daily_trade_limit():
                return {
                    "allowed": False,
                    "reason": "max_daily_trades_reached",
                }
            
            # Check 2: Daily loss limit
            if not self._check_daily_loss_limit(account_balance):
                return {
                    "allowed": False,
                    "reason": "max_daily_loss_reached",
                }
            
            # Check 3: Position size limit
            if not self._check_position_size(max_loss, account_balance):
                return {
                    "allowed": False,
                    "reason": "position_too_large",
                }
            
            # Check 4: Per-symbol position limit
            if not self._check_symbol_limit(symbol, current_positions):
                return {
                    "allowed": False,
                    "reason": "too_many_positions_in_symbol",
                }
            
            # Check 5: Portfolio heat
            if not self._check_portfolio_heat(max_loss, account_balance, current_positions):
                return {
                    "allowed": False,
                    "reason": "portfolio_heat_too_high",
                }
            
            # Check 6: Concentration risk
            if not self._check_concentration(symbol, max_loss, account_balance, current_positions):
                return {
                    "allowed": False,
                    "reason": "concentration_risk_too_high",
                }
            
            return {
                "allowed": True,
                "reason": "all_risk_checks_passed",
            }
        
        except Exception as e:
            logger.error(f"Error checking risk constraints: {e}")
            return {
                "allowed": False,
                "reason": f"error: {str(e)}",
            }
    
    def _check_daily_trade_limit(self) -> bool:
        """Check if daily trade limit has been reached"""
        try:
            with self.db.get_session() as session:
                today = datetime.now().date()
                today_start = datetime.combine(today, datetime.min.time())
                
                trades_today = session.query(Trade).filter(
                    Trade.timestamp_enter >= today_start
                ).count()
                
                return trades_today < self.max_trades_per_day
        
        except Exception as e:
            logger.error(f"Error checking daily trade limit: {e}")
            return False
    
    def _check_daily_loss_limit(self, account_balance: float) -> bool:
        """Check if daily loss limit has been reached"""
        try:
            with self.db.get_session() as session:
                today = datetime.now().date()
                today_start = datetime.combine(today, datetime.min.time())
                
                # Get all trades closed today
                closed_trades = session.query(Trade).filter(
                    Trade.timestamp_exit >= today_start,
                    Trade.status == "closed"
                ).all()
                
                total_pnl_today = sum(trade.pnl for trade in closed_trades)
                
                # Get unrealized P&L for open positions
                open_trades = session.query(Trade).filter(
                    Trade.status == "open"
                ).all()
                
                # Note: In production, calculate actual unrealized P&L from positions
                # For now, use a conservative estimate
                
                max_daily_loss = account_balance * (self.max_daily_loss_pct / 100)
                
                return total_pnl_today > -max_daily_loss
        
        except Exception as e:
            logger.error(f"Error checking daily loss limit: {e}")
            return False
    
    def _check_position_size(self, max_loss: float, account_balance: float) -> bool:
        """Check if position size is within limits"""
        max_position_size = account_balance * (self.max_position_size_pct / 100)
        return max_loss <= max_position_size
    
    def _check_symbol_limit(self, symbol: str, current_positions: List[Dict]) -> bool:
        """Check if too many positions in this symbol"""
        symbol_positions = [p for p in current_positions if p.get("symbol") == symbol]
        return len(symbol_positions) < self.max_positions_per_symbol
    
    def _check_portfolio_heat(
        self,
        new_max_loss: float,
        account_balance: float,
        current_positions: List[Dict]
    ) -> bool:
        """
        Check portfolio heat (total risk as % of account)
        Portfolio heat = sum of all max losses / account balance
        """
        try:
            # Calculate total current risk
            current_risk = sum(p.get("max_loss", 0) for p in current_positions)
            
            # Add new trade risk
            total_risk = current_risk + new_max_loss
            
            # Calculate as percentage
            heat_pct = (total_risk / account_balance) * 100
            
            logger.debug(f"Portfolio heat: {heat_pct:.2f}% (max: {self.max_portfolio_heat}%)")
            
            return heat_pct <= self.max_portfolio_heat
        
        except Exception as e:
            logger.error(f"Error checking portfolio heat: {e}")
            return False
    
    def _check_concentration(
        self,
        symbol: str,
        new_max_loss: float,
        account_balance: float,
        current_positions: List[Dict]
    ) -> bool:
        """Check if too much concentration in one symbol"""
        try:
            # Calculate total risk in this symbol
            symbol_risk = sum(
                p.get("max_loss", 0)
                for p in current_positions
                if p.get("symbol") == symbol
            )
            
            # Add new trade
            symbol_risk += new_max_loss
            
            # Max 30% of portfolio in one symbol
            max_symbol_concentration = account_balance * 0.30
            
            return symbol_risk <= max_symbol_concentration
        
        except Exception as e:
            logger.error(f"Error checking concentration: {e}")
            return False
    
    def calculate_position_metrics(self, positions: List[Dict]) -> Dict:
        """Calculate portfolio-level risk metrics"""
        try:
            if not positions:
                return {
                    "total_positions": 0,
                    "total_delta": 0.0,
                    "total_theta": 0.0,
                    "total_vega": 0.0,
                    "total_risk": 0.0,
                    "total_unrealized_pnl": 0.0,
                }
            
            total_delta = sum(p.get("delta_exposure", 0) for p in positions)
            total_theta = sum(p.get("theta_exposure", 0) for p in positions)
            total_vega = sum(p.get("vega_exposure", 0) for p in positions)
            total_risk = sum(p.get("max_loss", 0) for p in positions)
            total_unrealized = sum(p.get("unrealized_pnl", 0) for p in positions)
            
            return {
                "total_positions": len(positions),
                "total_delta": round(total_delta, 2),
                "total_theta": round(total_theta, 2),
                "total_vega": round(total_vega, 2),
                "total_risk": round(total_risk, 2),
                "total_unrealized_pnl": round(total_unrealized, 2),
            }
        
        except Exception as e:
            logger.error(f"Error calculating position metrics: {e}")
            return {}
    
    def get_risk_summary(self, account_balance: float) -> Dict:
        """Get current risk summary"""
        try:
            with self.db.get_session() as session:
                # Get open trades
                open_trades = session.query(Trade).filter(
                    Trade.status == "open"
                ).all()
                
                # Get today's closed trades
                today = datetime.now().date()
                today_start = datetime.combine(today, datetime.min.time())
                
                closed_today = session.query(Trade).filter(
                    Trade.timestamp_exit >= today_start,
                    Trade.status == "closed"
                ).all()
                
                trades_today = session.query(Trade).filter(
                    Trade.timestamp_enter >= today_start
                ).count()
                
                total_pnl_today = sum(trade.pnl for trade in closed_today)
                
                # Calculate metrics
                total_risk = sum(
                    trade.risk.get("max_loss", 0)
                    for trade in open_trades
                )
                
                portfolio_heat = (total_risk / account_balance * 100) if account_balance > 0 else 0
                
                return {
                    "account_balance": account_balance,
                    "open_positions": len(open_trades),
                    "trades_today": trades_today,
                    "pnl_today": round(total_pnl_today, 2),
                    "total_risk": round(total_risk, 2),
                    "portfolio_heat_pct": round(portfolio_heat, 2),
                    "daily_trades_remaining": max(0, self.max_trades_per_day - trades_today),
                }
        
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}


