"""
PDT (Pattern Day Trader) Compliance Module

Enforces PDT rules for accounts under $25,000:
- Maximum 3 day trades per 5 business days
- Day trade = opening and closing same position same day
- Must hold positions overnight
- Automatic trading suspension when limit reached
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.database.session import get_db
from src.database.models import Trade

logger = logging.getLogger(__name__)


class PDTStatus(Enum):
    """PDT compliance status"""
    COMPLIANT = "compliant"
    WARNING = "warning"  # 2 day trades used
    LIMIT_REACHED = "limit_reached"  # 3 day trades used
    SUSPENDED = "suspended"  # Trading suspended


@dataclass
class PDTInfo:
    """PDT compliance information"""
    account_balance: float
    is_pdt_account: bool
    day_trades_used: int
    max_day_trades: int
    days_remaining: int
    status: PDTStatus
    can_trade: bool
    suspension_reason: Optional[str] = None


@dataclass
class DayTrade:
    """Represents a day trade"""
    symbol: str
    strategy: str
    open_time: datetime
    close_time: datetime
    is_day_trade: bool
    trade_id: str


class PDTComplianceManager:
    """Manages PDT compliance for trading accounts"""
    
    def __init__(self, account_balance: float):
        self.account_balance = account_balance
        self.pdt_threshold = 25000.0  # $25,000 minimum for PDT exemption
        self.max_day_trades = 3  # Maximum day trades per 5 days
        self.db = get_db()
        
        logger.info(f"PDT Compliance Manager initialized for ${account_balance:,.2f} account")
    
    def is_pdt_account(self) -> bool:
        """Check if account is subject to PDT rules"""
        return self.account_balance < self.pdt_threshold
    
    def get_pdt_status(self) -> PDTInfo:
        """Get current PDT compliance status"""
        if not self.is_pdt_account():
            return PDTInfo(
                account_balance=self.account_balance,
                is_pdt_account=False,
                day_trades_used=0,
                max_day_trades=0,
                days_remaining=0,
                status=PDTStatus.COMPLIANT,
                can_trade=True
            )
        
        # Get day trades in last 5 business days
        day_trades_count = self._count_day_trades()
        days_remaining = self._get_business_days_remaining()
        
        # Determine status
        if day_trades_count >= self.max_day_trades:
            status = PDTStatus.LIMIT_REACHED
            can_trade = False
            suspension_reason = f"PDT limit reached: {day_trades_count}/{self.max_day_trades} day trades"
        elif day_trades_count >= 2:
            status = PDTStatus.WARNING
            can_trade = True
        else:
            status = PDTStatus.COMPLIANT
            can_trade = True
        
        return PDTInfo(
            account_balance=self.account_balance,
            is_pdt_account=True,
            day_trades_used=day_trades_count,
            max_day_trades=self.max_day_trades,
            days_remaining=days_remaining,
            status=status,
            can_trade=can_trade,
            suspension_reason=suspension_reason if not can_trade else None
        )
    
    def _count_day_trades(self) -> int:
        """Count day trades in last 5 business days"""
        try:
            # Get 5 business days ago
            cutoff_date = self._get_business_days_ago(5)
            
            with self.db.get_session() as session:
                # Get all trades in last 5 business days
                recent_trades = session.query(Trade).filter(
                    Trade.timestamp_enter >= cutoff_date,
                    Trade.status == 'closed'
                ).order_by(Trade.timestamp_enter.asc()).all()
                
                day_trades = 0
                processed_trades = set()
                
                for trade in recent_trades:
                    if trade.id in processed_trades:
                        continue
                    
                    # Check if this is a day trade
                    if self._is_day_trade(trade):
                        day_trades += 1
                        processed_trades.add(trade.id)
                        
                        logger.debug(f"Day trade detected: {trade.symbol} {trade.strategy} "
                                   f"on {trade.timestamp_enter.date()}")
                
                logger.info(f"Day trades in last 5 business days: {day_trades}")
                return day_trades
                
        except Exception as e:
            logger.error(f"Error counting day trades: {e}")
            return 0
    
    def _is_day_trade(self, trade: Trade) -> bool:
        """Check if a trade is a day trade"""
        if not trade.timestamp_enter or not trade.timestamp_exit:
            return False
        
        # Same day entry and exit = day trade
        entry_date = trade.timestamp_enter.date()
        exit_date = trade.timestamp_exit.date()
        
        return entry_date == exit_date
    
    def _get_business_days_ago(self, days: int) -> datetime:
        """Get datetime for N business days ago"""
        current_date = datetime.now()
        business_days = 0
        
        while business_days < days:
            current_date -= timedelta(days=1)
            # Skip weekends (Monday=0, Sunday=6)
            if current_date.weekday() < 5:
                business_days += 1
        
        return current_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _get_business_days_remaining(self) -> int:
        """Get business days remaining in 5-day window"""
        # This is simplified - in practice, you'd track the exact 5-day window
        # For now, return days until next Monday (fresh 5-day window)
        today = datetime.now()
        days_until_monday = (7 - today.weekday()) % 7
        if days_until_monday == 0 and today.weekday() != 0:  # Not Monday
            days_until_monday = 7
        
        return days_until_monday
    
    def can_open_position(self, symbol: str, strategy: str) -> Tuple[bool, str]:
        """Check if we can open a new position (PDT compliance)"""
        pdt_info = self.get_pdt_status()
        
        if not pdt_info.is_pdt_account:
            return True, "Account exempt from PDT rules"
        
        if not pdt_info.can_trade:
            return False, pdt_info.suspension_reason or "PDT limit reached"
        
        # Check if we already have a position today
        today_positions = self._get_today_positions()
        if today_positions > 0:
            return False, f"PDT account limited to 1 position per day (have {today_positions})"
        
        return True, "PDT compliant - can open position"
    
    def _get_today_positions(self) -> int:
        """Count positions opened today"""
        try:
            today = datetime.now().date()
            
            with self.db.get_session() as session:
                today_trades = session.query(Trade).filter(
                    Trade.timestamp_entry >= today,
                    Trade.status.in_(['open', 'closed'])
                ).count()
                
                return today_trades
                
        except Exception as e:
            logger.error(f"Error counting today's positions: {e}")
            return 0
    
    def can_close_position(self, trade_id: str) -> Tuple[bool, str]:
        """Check if we can close a position (PDT compliance)"""
        pdt_info = self.get_pdt_status()
        
        if not pdt_info.is_pdt_account:
            return True, "Account exempt from PDT rules"
        
        try:
            with self.db.get_session() as session:
                trade = session.query(Trade).filter(Trade.id == trade_id).first()
                if not trade:
                    return False, "Trade not found"
                
                # Check if this would be a day trade
                if self._is_day_trade(trade):
                    # Check if we can make another day trade
                    if pdt_info.day_trades_used >= self.max_day_trades:
                        return False, f"PDT limit reached: {pdt_info.day_trades_used}/{self.max_day_trades} day trades"
                    else:
                        return True, f"Day trade allowed: {pdt_info.day_trades_used + 1}/{self.max_day_trades}"
                else:
                    return True, "Not a day trade - safe to close"
                    
        except Exception as e:
            logger.error(f"Error checking close permission: {e}")
            return False, f"Error: {e}"
    
    def get_pdt_warnings(self) -> List[str]:
        """Get PDT compliance warnings"""
        warnings = []
        pdt_info = self.get_pdt_status()
        
        if not pdt_info.is_pdt_account:
            return warnings
        
        if pdt_info.status == PDTStatus.WARNING:
            warnings.append(f"⚠️ PDT Warning: {pdt_info.day_trades_used}/{pdt_info.max_day_trades} day trades used")
            warnings.append(f"📅 {pdt_info.days_remaining} business days until reset")
        
        elif pdt_info.status == PDTStatus.LIMIT_REACHED:
            warnings.append("🚨 PDT LIMIT REACHED: Trading suspended for 5 business days")
            warnings.append(f"📅 Reset in {pdt_info.days_remaining} business days")
        
        return warnings
    
    def get_recommended_strategy(self) -> Dict:
        """Get PDT-compliant strategy recommendations"""
        if not self.is_pdt_account():
            return {
                "dte_range": [14, 45],
                "max_positions": 5,
                "strategy": "Standard options trading"
            }
        
        pdt_info = self.get_pdt_status()
        
        if not pdt_info.can_trade:
            return {
                "dte_range": [0, 0],  # No trading
                "max_positions": 0,
                "strategy": "Trading suspended - PDT limit reached"
            }
        
        return {
            "dte_range": [21, 45],  # Minimum 3 weeks
            "max_positions": 1,  # Only 1 position per day
            "strategy": "Swing trading - hold overnight minimum",
            "warnings": self.get_pdt_warnings()
        }
    
    def log_pdt_status(self):
        """Log current PDT status for monitoring"""
        pdt_info = self.get_pdt_status()
        
        if pdt_info.is_pdt_account:
            logger.info(f"PDT Status: {pdt_info.day_trades_used}/{pdt_info.max_day_trades} day trades used")
            logger.info(f"PDT Status: {pdt_info.status.value}")
            
            if pdt_info.warnings:
                for warning in pdt_info.warnings:
                    logger.warning(warning)
        else:
            logger.info("Account exempt from PDT rules (>$25K)")


# Example usage and testing
if __name__ == "__main__":
    # Test with different account sizes
    test_accounts = [3000, 15000, 25000, 50000]
    
    for balance in test_accounts:
        print(f"\n=== Testing ${balance:,} Account ===")
        pdt_manager = PDTComplianceManager(balance)
        pdt_info = pdt_manager.get_pdt_status()
        
        print(f"PDT Account: {pdt_info.is_pdt_account}")
        print(f"Day Trades: {pdt_info.day_trades_used}/{pdt_info.max_day_trades}")
        print(f"Status: {pdt_info.status.value}")
        print(f"Can Trade: {pdt_info.can_trade}")
        
        if pdt_info.suspension_reason:
            print(f"Reason: {pdt_info.suspension_reason}")
        
        # Test strategy recommendations
        strategy = pdt_manager.get_recommended_strategy()
        print(f"Recommended DTE: {strategy['dte_range']}")
        print(f"Max Positions: {strategy['max_positions']}")
        print(f"Strategy: {strategy['strategy']}")
