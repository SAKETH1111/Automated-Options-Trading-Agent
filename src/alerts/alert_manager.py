"""
Alert Manager
Coordinate all alert types and manage notifications
"""

from typing import Dict
from datetime import datetime
from loguru import logger

from .email_alerts import EmailAlertManager
from .sms_alerts import SMSAlertManager


class AlertManager:
    """
    Main alert coordinator
    Manages email and SMS alerts
    """
    
    def __init__(self):
        """Initialize alert manager"""
        self.email = EmailAlertManager()
        self.sms = SMSAlertManager()
        
        logger.info("Alert Manager initialized")
    
    def notify_trade_executed(self, trade: Dict):
        """Notify when trade is executed"""
        self.email.send_trade_alert(trade)
        logger.info(f"Trade alert sent for {trade.get('symbol')}")
    
    def notify_circuit_breaker(self, reason: str, details: Dict):
        """Notify when circuit breaker trips"""
        self.email.send_circuit_breaker_alert(reason, details)
        self.sms.send_circuit_breaker_sms(reason)
        logger.warning(f"Circuit breaker alert sent: {reason}")
    
    def notify_position_event(self, position: Dict, event_type: str):
        """Notify for position events"""
        self.email.send_position_alert(position, event_type)
        
        # SMS only for critical events
        if event_type == 'STOP_LOSS':
            self.sms.send_large_loss_sms(abs(position.get('current_pnl', 0)))
    
    def send_daily_summary(self, summary: Dict):
        """Send daily summary"""
        self.email.send_daily_summary(summary)
        logger.info("Daily summary sent")
    
    def notify_system_error(self, error: str):
        """Notify for system errors"""
        self.sms.send_system_error_sms(error)
        logger.error(f"System error alert sent: {error}")
    
    # Compatibility wrappers for orchestrator
    def send_alert(self, alert_type: str, message: str, severity: str = "info"):
        """Generic alert sender (compatibility wrapper)"""
        logger.info(f"Alert [{severity}] {alert_type}: {message}")
        
        if severity in ["error", "critical"]:
            self.notify_system_error(message)
    
    def alert_system_error(self, error_type: str, message: str):
        """System error alert (compatibility wrapper)"""
        logger.error(f"System error [{error_type}]: {message}")
        self.notify_system_error(f"{error_type}: {message}")
    
    def alert_trade_executed(self, trade_id: str, symbol: str, strategy: str, contracts: int):
        """Trade executed alert (compatibility wrapper)"""
        trade_data = {
            'trade_id': trade_id,
            'symbol': symbol,
            'strategy': strategy,
            'contracts': contracts,
            'timestamp': datetime.now()
        }
        self.notify_trade_executed(trade_data)
    
    def alert_position_closed(self, trade_id: str, symbol: str, pnl: float, reason: str):
        """Position closed alert (compatibility wrapper)"""
        position_data = {
            'trade_id': trade_id,
            'symbol': symbol,
            'current_pnl': pnl,
            'exit_reason': reason,
            'timestamp': datetime.now()
        }
        event_type = 'STOP_LOSS' if pnl < 0 else 'TAKE_PROFIT'
        self.notify_position_event(position_data, event_type)

