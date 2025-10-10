"""
Email Alert Manager
Send email notifications for trading events
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger


class EmailAlertManager:
    """
    Send email alerts for trading events
    Supports: Trade execution, circuit breaker, daily summaries
    """
    
    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None,
        recipient_email: Optional[str] = None
    ):
        """
        Initialize email alert manager
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP port
            sender_email: Sender email address
            sender_password: Sender email password/app password
            recipient_email: Recipient email address
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email or os.getenv('ALERT_EMAIL')
        self.sender_password = sender_password or os.getenv('ALERT_EMAIL_PASSWORD')
        self.recipient_email = recipient_email or os.getenv('ALERT_RECIPIENT_EMAIL')
        
        self.enabled = bool(self.sender_email and self.sender_password and self.recipient_email)
        
        if self.enabled:
            logger.info(f"Email alerts enabled: {self.recipient_email}")
        else:
            logger.warning("Email alerts disabled - missing configuration")
    
    def send_trade_alert(self, trade_details: Dict):
        """Send alert when trade is executed"""
        if not self.enabled:
            return
        
        subject = f"🔔 Trade Executed: {trade_details.get('strategy', 'Unknown')} on {trade_details.get('symbol', 'Unknown')}"
        
        body = f"""
Trade Execution Alert
=====================

Symbol: {trade_details.get('symbol')}
Strategy: {trade_details.get('strategy')}
Action: {trade_details.get('action', 'ENTRY')}

Entry Details:
- Price: ${trade_details.get('entry_price', 0):.2f}
- Strikes: {trade_details.get('strikes', [])}
- Max Profit: ${trade_details.get('max_profit', 0):.2f}
- Max Loss: ${trade_details.get('max_loss', 0):.2f}
- POP: {trade_details.get('pop', 0):.1%}

Market Conditions:
- Underlying Price: ${trade_details.get('underlying_price', 0):.2f}
- IV Rank: {trade_details.get('iv_rank', 0):.0f}
- Technical Signal: {trade_details.get('technical_signal', 'N/A')}

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
Automated Options Trading Agent
"""
        
        self._send_email(subject, body)
    
    def send_circuit_breaker_alert(self, reason: str, details: Dict):
        """Send alert when circuit breaker trips"""
        if not self.enabled:
            return
        
        subject = f"🚨 CIRCUIT BREAKER TRIPPED: {reason}"
        
        body = f"""
CIRCUIT BREAKER ALERT
=====================

⚠️  Trading has been PAUSED

Reason: {reason}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Details:
{self._format_dict(details)}

Action Required:
- Review the situation
- Check system logs
- Investigate cause
- Trading will auto-resume in 24 hours (or manual reset)

---
Automated Options Trading Agent
"""
        
        self._send_email(subject, body)
    
    def send_daily_summary(self, summary: Dict):
        """Send daily performance summary"""
        if not self.enabled:
            return
        
        subject = f"📊 Daily Summary - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = f"""
Daily Trading Summary
=====================

Date: {datetime.now().strftime('%Y-%m-%d')}

Performance:
- Trades Today: {summary.get('total_trades', 0)}
- Win Rate: {summary.get('win_rate', 0):.1%}
- Daily P&L: ${summary.get('total_pnl', 0):+,.2f}

Open Positions: {summary.get('open_positions', 0)}

Risk Metrics:
- Total Risk: ${summary.get('total_risk', 0):,.2f}
- Available Capital: ${summary.get('available_capital', 0):,.2f}
- Circuit Breaker: {summary.get('circuit_breaker_status', 'ACTIVE')}

System Status:
- Data Collection: {summary.get('data_collection_status', 'ACTIVE')}
- Trading Status: {summary.get('trading_status', 'ACTIVE')}

---
Automated Options Trading Agent
"""
        
        self._send_email(subject, body)
    
    def send_position_alert(self, position: Dict, alert_type: str):
        """Send alert for position events"""
        if not self.enabled:
            return
        
        if alert_type == 'PROFIT_TARGET':
            subject = f"✅ Profit Target Reached: {position.get('symbol')}"
        elif alert_type == 'STOP_LOSS':
            subject = f"⚠️ Stop Loss Triggered: {position.get('symbol')}"
        elif alert_type == 'EXPIRATION':
            subject = f"⏰ Position Expiring: {position.get('symbol')}"
        else:
            subject = f"🔔 Position Alert: {position.get('symbol')}"
        
        body = f"""
Position Alert
==============

Alert Type: {alert_type}
Symbol: {position.get('symbol')}
Strategy: {position.get('strategy')}

Position Details:
- Entry Date: {position.get('entry_date')}
- Days Held: {position.get('days_held', 0)}
- Current P&L: ${position.get('current_pnl', 0):+,.2f}
- Max Profit: ${position.get('max_profit', 0):.2f}
- Max Loss: ${position.get('max_loss', 0):.2f}

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
Automated Options Trading Agent
"""
        
        self._send_email(subject, body)
    
    def _send_email(self, subject: str, body: str):
        """Send email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Email sent: {subject}")
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
    
    def _format_dict(self, d: Dict) -> str:
        """Format dictionary for email"""
        return '\n'.join(f"- {k}: {v}" for k, v in d.items())

