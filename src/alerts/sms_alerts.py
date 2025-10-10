"""
SMS Alert Manager
Send SMS notifications for critical events
"""

import os
from typing import Optional
from datetime import datetime
from loguru import logger


class SMSAlertManager:
    """
    Send SMS alerts for critical trading events
    Uses Twilio API (requires account setup)
    """
    
    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_number: Optional[str] = None,
        to_number: Optional[str] = None
    ):
        """
        Initialize SMS alert manager
        
        Args:
            account_sid: Twilio account SID
            auth_token: Twilio auth token
            from_number: Twilio phone number
            to_number: Recipient phone number
        """
        self.account_sid = account_sid or os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = auth_token or os.getenv('TWILIO_AUTH_TOKEN')
        self.from_number = from_number or os.getenv('TWILIO_FROM_NUMBER')
        self.to_number = to_number or os.getenv('TWILIO_TO_NUMBER')
        
        self.enabled = bool(self.account_sid and self.auth_token and 
                           self.from_number and self.to_number)
        
        if self.enabled:
            try:
                from twilio.rest import Client
                self.client = Client(self.account_sid, self.auth_token)
                logger.info(f"SMS alerts enabled: {self.to_number}")
            except ImportError:
                logger.warning("Twilio not installed. Run: pip install twilio")
                self.enabled = False
        else:
            logger.warning("SMS alerts disabled - missing configuration")
    
    def send_circuit_breaker_sms(self, reason: str):
        """Send SMS when circuit breaker trips"""
        if not self.enabled:
            return
        
        message = f"🚨 CIRCUIT BREAKER TRIPPED: {reason}. Trading paused. Check system immediately."
        self._send_sms(message)
    
    def send_large_loss_sms(self, loss_amount: float):
        """Send SMS for large losses"""
        if not self.enabled:
            return
        
        message = f"⚠️ Large Loss Alert: ${loss_amount:,.2f}. Review positions immediately."
        self._send_sms(message)
    
    def send_system_error_sms(self, error: str):
        """Send SMS for system errors"""
        if not self.enabled:
            return
        
        message = f"❌ System Error: {error[:100]}. Check logs immediately."
        self._send_sms(message)
    
    def _send_sms(self, message: str):
        """Send SMS message"""
        try:
            self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=self.to_number
            )
            
            logger.info(f"SMS sent: {message[:50]}...")
            
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")

