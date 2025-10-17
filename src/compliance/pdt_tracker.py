"""
PDT Compliance Tracker for Pattern Day Trading Rule Enforcement

This module implements comprehensive PDT compliance tracking for accounts under $25,000,
ensuring adherence to the 3-day-trade limit in a rolling 5-business-day window.

Key Features:
- Day trade detection and counting
- Rolling 5-business-day window monitoring
- Emergency allowance for circuit breaker triggers
- Hold period enforcement (overnight minimum)
- Real-time alerts and blocking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import sqlite3
from pathlib import Path

@dataclass
class DayTrade:
    """Represents a day trade transaction"""
    symbol: str
    contract_type: str
    strike: float
    expiration: str
    action: str  # 'OPEN' or 'CLOSE'
    quantity: int
    price: float
    timestamp: datetime
    position_id: str

@dataclass
class PDTSnapshot:
    """Current PDT compliance status"""
    account_value: float
    day_trades_used: int
    day_trades_remaining: int
    rolling_window_start: datetime
    rolling_window_end: datetime
    is_pdt_violation: bool
    can_day_trade: bool
    emergency_trades_available: bool

class PDTTracker:
    """Pattern Day Trading compliance tracker"""
    
    def __init__(self, db_path: str = "trading_agent.db"):
        self.db_path = db_path
        self.max_day_trades = 3
        self.rolling_window_days = 5
        self.emergency_allowance = True
        
        # Initialize database
        self._init_database()
        
        logger.info(f"PDT Tracker initialized - Max day trades: {self.max_day_trades}")
    
    def _init_database(self):
        """Initialize PDT tracking tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Day trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS day_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                contract_type TEXT NOT NULL,
                strike REAL NOT NULL,
                expiration TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                is_emergency BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Position tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS position_tracking (
                position_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                contract_type TEXT NOT NULL,
                strike REAL NOT NULL,
                expiration TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                open_timestamp DATETIME NOT NULL,
                close_timestamp DATETIME,
                is_closed BOOLEAN DEFAULT FALSE,
                day_trade_count INTEGER DEFAULT 0
            )
        """)
        
        # PDT violations log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pdt_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                violation_type TEXT NOT NULL,
                day_trades_attempted INTEGER NOT NULL,
                day_trades_limit INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                position_id TEXT,
                description TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_position_open(self, position_id: str, symbol: str, contract_type: str, 
                           strike: float, expiration: str, quantity: int, 
                           price: float, timestamp: datetime) -> bool:
        """Record opening a position"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO position_tracking 
                (position_id, symbol, contract_type, strike, expiration, quantity, 
                 open_timestamp, close_timestamp, is_closed, day_trade_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, FALSE, 0)
            """, (position_id, symbol, contract_type, strike, expiration, 
                  quantity, timestamp))
            
            cursor.execute("""
                INSERT INTO day_trades 
                (position_id, symbol, contract_type, strike, expiration, action, 
                 quantity, price, timestamp)
                VALUES (?, ?, ?, ?, ?, 'OPEN', ?, ?, ?)
            """, (position_id, symbol, contract_type, strike, expiration, 
                  quantity, price, timestamp))
            
            conn.commit()
            logger.info(f"Recorded position open: {position_id} - {symbol} {contract_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording position open: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def record_position_close(self, position_id: str, price: float, 
                            timestamp: datetime, is_emergency: bool = False) -> bool:
        """Record closing a position and check for day trade"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get position details
            cursor.execute("""
                SELECT symbol, contract_type, strike, expiration, quantity, open_timestamp
                FROM position_tracking WHERE position_id = ?
            """, (position_id,))
            
            position = cursor.fetchone()
            if not position:
                logger.error(f"Position not found: {position_id}")
                return False
            
            symbol, contract_type, strike, expiration, quantity, open_timestamp = position
            
            # Check if this is a day trade (same day open/close)
            open_date = pd.to_datetime(open_timestamp).date()
            close_date = timestamp.date()
            is_day_trade = open_date == close_date
            
            # Update position tracking
            cursor.execute("""
                UPDATE position_tracking 
                SET close_timestamp = ?, is_closed = TRUE, day_trade_count = ?
                WHERE position_id = ?
            """, (timestamp, 1 if is_day_trade else 0, position_id))
            
            # Record day trade if applicable
            if is_day_trade:
                cursor.execute("""
                    INSERT INTO day_trades 
                    (position_id, symbol, contract_type, strike, expiration, action, 
                     quantity, price, timestamp, is_emergency)
                    VALUES (?, ?, ?, ?, ?, 'CLOSE', ?, ?, ?, ?)
                """, (position_id, symbol, contract_type, strike, expiration, 
                      quantity, price, timestamp, is_emergency))
                
                logger.info(f"Day trade recorded: {position_id} - {symbol} {contract_type}")
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error recording position close: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_current_pdt_status(self, account_value: float) -> PDTSnapshot:
        """Get current PDT compliance status"""
        # Calculate rolling window
        today = datetime.now().date()
        window_start = today - timedelta(days=self.rolling_window_days + 2)  # Buffer for weekends
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count day trades in rolling window
        cursor.execute("""
            SELECT COUNT(*) FROM day_trades 
            WHERE timestamp >= ? AND action = 'CLOSE'
        """, (window_start.strftime('%Y-%m-%d'),))
        
        day_trades_used = cursor.fetchone()[0]
        
        # Check for emergency trades
        cursor.execute("""
            SELECT COUNT(*) FROM day_trades 
            WHERE timestamp >= ? AND action = 'CLOSE' AND is_emergency = TRUE
        """, (window_start.strftime('%Y-%m-%d'),))
        
        emergency_trades = cursor.fetchone()[0]
        
        conn.close()
        
        # Calculate remaining trades
        day_trades_remaining = self.max_day_trades - day_trades_used
        
        # Determine if can day trade
        can_day_trade = day_trades_remaining > 0
        
        # PDT violation check
        is_pdt_violation = day_trades_used >= self.max_day_trades
        
        # Emergency allowance (circuit breaker triggers only)
        emergency_available = self.emergency_allowance and emergency_trades < 2
        
        return PDTSnapshot(
            account_value=account_value,
            day_trades_used=day_trades_used,
            day_trades_remaining=day_trades_remaining,
            rolling_window_start=window_start,
            rolling_window_end=today,
            is_pdt_violation=is_pdt_violation,
            can_day_trade=can_day_trade,
            emergency_trades_available=emergency_available
        )
    
    def can_execute_day_trade(self, is_emergency: bool = False) -> Tuple[bool, str]:
        """Check if a day trade can be executed"""
        status = self.get_current_pdt_status(0)  # Account value not needed for day trade check
        
        if is_emergency and status.emergency_trades_available:
            return True, "Emergency day trade allowed"
        
        if status.can_day_trade:
            return True, f"Day trade allowed ({status.day_trades_remaining} remaining)"
        
        if status.is_pdt_violation:
            return False, f"PDT violation: {status.day_trades_used}/{self.max_day_trades} day trades used"
        
        return False, "Day trade not permitted"
    
    def enforce_hold_period(self, position_id: str) -> Tuple[bool, str]:
        """Enforce minimum hold period (overnight)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT open_timestamp FROM position_tracking 
            WHERE position_id = ? AND is_closed = FALSE
        """, (position_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return False, "Position not found"
        
        open_time = pd.to_datetime(result[0])
        current_time = datetime.now()
        
        # Check if position was opened today
        if open_time.date() == current_time.date():
            return False, "Position must be held overnight (PDT compliance)"
        
        return True, "Hold period satisfied"
    
    def get_day_trade_history(self, days: int = 30) -> pd.DataFrame:
        """Get day trade history for analysis"""
        start_date = datetime.now() - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT dt.*, pt.open_timestamp 
            FROM day_trades dt
            JOIN position_tracking pt ON dt.position_id = pt.position_id
            WHERE dt.timestamp >= ? AND dt.action = 'CLOSE'
            ORDER BY dt.timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date.strftime('%Y-%m-%d'),))
        conn.close()
        
        return df
    
    def log_pdt_violation(self, violation_type: str, day_trades_attempted: int, 
                         position_id: Optional[str] = None, description: str = ""):
        """Log PDT violations for audit trail"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO pdt_violations 
            (violation_type, day_trades_attempted, day_trades_limit, timestamp, 
             position_id, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (violation_type, day_trades_attempted, self.max_day_trades, 
              datetime.now(), position_id, description))
        
        conn.commit()
        conn.close()
        
        logger.warning(f"PDT violation logged: {violation_type} - {description}")
    
    def get_compliance_report(self) -> Dict:
        """Generate comprehensive PDT compliance report"""
        status = self.get_current_pdt_status(0)
        history = self.get_day_trade_history(30)
        
        return {
            "current_status": {
                "day_trades_used": status.day_trades_used,
                "day_trades_remaining": status.day_trades_remaining,
                "can_day_trade": status.can_day_trade,
                "is_pdt_violation": status.is_pdt_violation,
                "rolling_window": f"{status.rolling_window_start} to {status.rolling_window_end}"
            },
            "recent_activity": {
                "total_day_trades_30d": len(history),
                "emergency_trades": len(history[history.get('is_emergency', False)]),
                "avg_day_trades_per_week": len(history) / 4.3
            },
            "recommendations": self._get_compliance_recommendations(status)
        }
    
    def _get_compliance_recommendations(self, status: PDTSnapshot) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        if status.day_trades_used >= 2:
            recommendations.append("Consider reducing day trading frequency")
        
        if status.is_pdt_violation:
            recommendations.append("CRITICAL: PDT violation - no more day trades allowed")
        
        if status.day_trades_remaining == 1:
            recommendations.append("Last day trade available - use carefully")
        
        if not status.can_day_trade:
            recommendations.append("Hold all positions overnight to avoid day trades")
        
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    # Initialize PDT tracker
    tracker = PDTTracker()
    
    # Test position tracking
    position_id = "TEST_001"
    symbol = "SPY"
    contract_type = "PUT"
    strike = 400.0
    expiration = "2024-01-19"
    quantity = 1
    price = 2.50
    
    # Record position open
    tracker.record_position_open(
        position_id, symbol, contract_type, strike, expiration, 
        quantity, price, datetime.now()
    )
    
    # Check PDT status
    status = tracker.get_current_pdt_status(5000)  # $5,000 account
    print(f"PDT Status: {status.day_trades_used}/{tracker.max_day_trades} day trades used")
    print(f"Can day trade: {status.can_day_trade}")
    
    # Test day trade check
    can_trade, reason = tracker.can_execute_day_trade()
    print(f"Can execute day trade: {can_trade} - {reason}")
    
    # Generate compliance report
    report = tracker.get_compliance_report()
    print(f"Compliance Report: {report}")
