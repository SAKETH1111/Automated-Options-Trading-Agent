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

