"""
Risk Manager - Main interface for risk management
Wrapper that provides unified risk management interface
"""

from typing import Dict
from loguru import logger

from src.database.session import get_db
from src.risk_management.portfolio_risk import PortfolioRiskManager
from src.risk_management.pdt_compliance import PDTComplianceManager


class RiskManager:
    """
    Unified risk management interface
    Combines portfolio risk management and PDT compliance
    """
    
    def __init__(self, config: Dict):
        """
        Initialize risk manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.db = get_db()
        
        # Get risk config
        risk_config = config.get("trading", {}).get("risk", {})
        
        # Initialize portfolio risk manager
        self.portfolio_risk = PortfolioRiskManager(
            db_session=self.db,
            total_capital=risk_config.get("initial_capital", 10000)
        )
        
        logger.info("RiskManager initialized")
    
    def get_risk_summary(self, account_equity: float) -> Dict:
        """
        Get risk summary for the account
        
        Args:
            account_equity: Current account equity
            
        Returns:
            Risk summary dictionary
        """
        try:
            # Get portfolio risk metrics
            metrics = self.portfolio_risk.get_portfolio_risk_metrics()
            
            # Get risk config
            risk_config = self.config.get("trading", {}).get("risk", {})
            max_trades_per_day = risk_config.get("max_trades_per_day", 10)
            
            # Count trades today
            from src.database.models import Trade
            from datetime import datetime
            
            with self.db.get_session() as session:
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                trades_today = session.query(Trade).filter(
                    Trade.timestamp_enter >= today_start
                ).count()
            
            return {
                "total_risk": metrics.get("total_risk", 0),
                "total_risk_pct": metrics.get("total_risk_pct", 0),
                "available_risk": metrics.get("available_risk", account_equity * 0.10),
                "max_positions": metrics.get("limits", {}).get("max_positions", 10),
                "open_positions": metrics.get("open_positions", 0),
                "daily_trades_today": trades_today,
                "daily_trades_remaining": max(0, max_trades_per_day - trades_today),
                "max_daily_loss_pct": risk_config.get("max_daily_loss_pct", 5.0),
                "max_portfolio_heat": risk_config.get("max_portfolio_heat", 30.0),
            }
        
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {
                "total_risk": 0,
                "total_risk_pct": 0,
                "available_risk": account_equity * 0.10,
                "daily_trades_remaining": 1,
            }
    
    def check_trade_risk(self, trade_params: Dict, account_equity: float) -> tuple[bool, str]:
        """
        Check if a trade passes risk checks
        
        Args:
            trade_params: Trade parameters
            account_equity: Current account equity
            
        Returns:
            (can_trade, reason)
        """
        try:
            # Check portfolio risk
            can_open, reason = self.portfolio_risk.check_can_open_position(trade_params)
            
            if not can_open:
                return False, reason
            
            # Additional risk checks
            max_loss = abs(trade_params.get('max_loss', 0))
            risk_pct = (max_loss / account_equity * 100) if account_equity > 0 else 0
            
            risk_config = self.config.get("trading", {}).get("risk", {})
            max_position_size_pct = risk_config.get("max_position_size_pct", 20.0)
            
            if risk_pct > max_position_size_pct:
                return False, f"Position risk {risk_pct:.1f}% exceeds max {max_position_size_pct}%"
            
            return True, "Risk checks passed"
        
        except Exception as e:
            logger.error(f"Error checking trade risk: {e}")
            return False, f"Risk check error: {e}"

