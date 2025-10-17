"""
Real Money Trading Alert System
Critical alerting for live trading with multiple channels (SMS, Email, Slack, Webhook)
"""

import asyncio
import json
import smtplib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from loguru import logger
import aiohttp
from twilio.rest import Client as TwilioClient

from src.config.settings import get_config


class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class AlertChannel(Enum):
    CONSOLE = "CONSOLE"
    EMAIL = "EMAIL"
    SMS = "SMS"
    SLACK = "SLACK"
    WEBHOOK = "WEBHOOK"
    TELEGRAM = "TELEGRAM"


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    timestamp: datetime
    level: AlertLevel
    title: str
    message: str
    details: Dict[str, Any]
    channels: List[AlertChannel]
    user_id: Optional[str] = None
    trade_id: Optional[str] = None
    position_id: Optional[str] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


@dataclass
class AlertChannelConfig:
    """Alert channel configuration"""
    channel: AlertChannel
    enabled: bool
    config: Dict[str, Any]
    rate_limit_seconds: int = 60  # Minimum seconds between alerts
    max_alerts_per_hour: int = 10


class RealMoneyAlertSystem:
    """
    Comprehensive alerting system for real money trading
    
    Features:
    - Multi-channel alerts (SMS, Email, Slack, Webhook)
    - Rate limiting and throttling
    - Alert acknowledgment tracking
    - Critical alert escalation
    - Alert history and analytics
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or get_config()
        
        # Alert channels configuration
        self.channels = self._initialize_alert_channels()
        
        # Rate limiting
        self.alert_history = {}  # channel -> list of alert times
        self.rate_limits = {}    # channel -> last_alert_time
        
        # Alert acknowledgment
        self.pending_alerts = {}  # alert_id -> alert
        self.acknowledged_alerts = {}
        
        # Initialize external services
        self._initialize_external_services()
        
        logger.info("RealMoneyAlertSystem initialized")
    
    def _initialize_alert_channels(self) -> Dict[AlertChannel, AlertChannelConfig]:
        """Initialize alert channel configurations"""
        channels = {}
        
        # Console alerts (always enabled)
        channels[AlertChannel.CONSOLE] = AlertChannelConfig(
            channel=AlertChannel.CONSOLE,
            enabled=True,
            config={}
        )
        
        # Email alerts
        email_config = self.config.get('alerts', {}).get('email', {})
        channels[AlertChannel.EMAIL] = AlertChannelConfig(
            channel=AlertChannel.EMAIL,
            enabled=email_config.get('enabled', False),
            config=email_config,
            rate_limit_seconds=300,  # 5 minutes
            max_alerts_per_hour=5
        )
        
        # SMS alerts
        sms_config = self.config.get('alerts', {}).get('sms', {})
        channels[AlertChannel.SMS] = AlertChannelConfig(
            channel=AlertChannel.SMS,
            enabled=sms_config.get('enabled', False),
            config=sms_config,
            rate_limit_seconds=600,  # 10 minutes
            max_alerts_per_hour=3
        )
        
        # Slack alerts
        slack_config = self.config.get('alerts', {}).get('slack', {})
        channels[AlertChannel.SLACK] = AlertChannelConfig(
            channel=AlertChannel.SLACK,
            enabled=slack_config.get('enabled', False),
            config=slack_config,
            rate_limit_seconds=60,
            max_alerts_per_hour=20
        )
        
        # Webhook alerts
        webhook_config = self.config.get('alerts', {}).get('webhook', {})
        channels[AlertChannel.WEBHOOK] = AlertChannelConfig(
            channel=AlertChannel.WEBHOOK,
            enabled=webhook_config.get('enabled', False),
            config=webhook_config,
            rate_limit_seconds=30,
            max_alerts_per_hour=50
        )
        
        # Telegram alerts
        telegram_config = self.config.get('alerts', {}).get('telegram', {})
        channels[AlertChannel.TELEGRAM] = AlertChannelConfig(
            channel=AlertChannel.TELEGRAM,
            enabled=telegram_config.get('enabled', False),
            config=telegram_config,
            rate_limit_seconds=60,
            max_alerts_per_hour=10
        )
        
        return channels
    
    def _initialize_external_services(self):
        """Initialize external services (Twilio, etc.)"""
        try:
            # Initialize Twilio for SMS
            sms_config = self.channels[AlertChannel.SMS].config
            if self.channels[AlertChannel.SMS].enabled and sms_config.get('twilio_sid'):
                self.twilio_client = TwilioClient(
                    sms_config['twilio_sid'],
                    sms_config['twilio_token']
                )
            else:
                self.twilio_client = None
                
        except Exception as e:
            logger.error(f"Error initializing external services: {e}")
            self.twilio_client = None
    
    # Core Alert Methods
    
    async def send_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        details: Dict[str, Any] = None,
        channels: List[AlertChannel] = None,
        user_id: str = None,
        trade_id: str = None,
        position_id: str = None
    ) -> str:
        """
        Send alert through specified channels
        
        Returns:
            alert_id: Unique identifier for the alert
        """
        try:
            # Generate alert ID
            alert_id = f"alert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(title) % 10000}"
            
            # Determine channels if not specified
            if channels is None:
                channels = self._get_default_channels(level)
            
            # Create alert
            alert = Alert(
                alert_id=alert_id,
                timestamp=datetime.utcnow(),
                level=level,
                title=title,
                message=message,
                details=details or {},
                channels=channels,
                user_id=user_id,
                trade_id=trade_id,
                position_id=position_id
            )
            
            # Send through each channel
            for channel in channels:
                if self._should_send_alert(channel, level):
                    await self._send_to_channel(alert, channel)
                else:
                    logger.debug(f"Rate limited for channel {channel.value}")
            
            # Store alert
            self.pending_alerts[alert_id] = alert
            
            # Log alert
            logger.info(f"Alert sent: {alert_id} - {level.value} - {title}")
            
            return alert_id
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return None
    
    def _get_default_channels(self, level: AlertLevel) -> List[AlertChannel]:
        """Get default channels based on alert level"""
        if level == AlertLevel.EMERGENCY:
            return [AlertChannel.CONSOLE, AlertChannel.SMS, AlertChannel.EMAIL, AlertChannel.SLACK]
        elif level == AlertLevel.CRITICAL:
            return [AlertChannel.CONSOLE, AlertChannel.EMAIL, AlertChannel.SLACK]
        elif level == AlertLevel.ERROR:
            return [AlertChannel.CONSOLE, AlertChannel.EMAIL, AlertChannel.SLACK]
        elif level == AlertLevel.WARNING:
            return [AlertChannel.CONSOLE, AlertChannel.SLACK]
        else:  # INFO
            return [AlertChannel.CONSOLE]
    
    def _should_send_alert(self, channel: AlertChannel, level: AlertLevel) -> bool:
        """Check if alert should be sent based on rate limits"""
        try:
            channel_config = self.channels.get(channel)
            if not channel_config or not channel_config.enabled:
                return False
            
            now = datetime.utcnow()
            channel_key = channel.value
            
            # Check rate limit
            last_alert_time = self.rate_limits.get(channel_key)
            if last_alert_time:
                time_since_last = (now - last_alert_time).total_seconds()
                if time_since_last < channel_config.rate_limit_seconds:
                    return False
            
            # Check hourly limit
            hour_ago = now - timedelta(hours=1)
            recent_alerts = self.alert_history.get(channel_key, [])
            recent_alerts = [t for t in recent_alerts if t > hour_ago]
            
            if len(recent_alerts) >= channel_config.max_alerts_per_hour:
                return False
            
            # Update tracking
            self.rate_limits[channel_key] = now
            if channel_key not in self.alert_history:
                self.alert_history[channel_key] = []
            self.alert_history[channel_key].append(now)
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking alert rate limit: {e}")
            return False
    
    async def _send_to_channel(self, alert: Alert, channel: AlertChannel):
        """Send alert to specific channel"""
        try:
            if channel == AlertChannel.CONSOLE:
                await self._send_console_alert(alert)
            elif channel == AlertChannel.EMAIL:
                await self._send_email_alert(alert)
            elif channel == AlertChannel.SMS:
                await self._send_sms_alert(alert)
            elif channel == AlertChannel.SLACK:
                await self._send_slack_alert(alert)
            elif channel == AlertChannel.WEBHOOK:
                await self._send_webhook_alert(alert)
            elif channel == AlertChannel.TELEGRAM:
                await self._send_telegram_alert(alert)
                
        except Exception as e:
            logger.error(f"Error sending alert to {channel.value}: {e}")
    
    # Channel-specific implementations
    
    async def _send_console_alert(self, alert: Alert):
        """Send console alert"""
        try:
            # Color coding based on level
            colors = {
                AlertLevel.INFO: "🔵",
                AlertLevel.WARNING: "🟡",
                AlertLevel.ERROR: "🔴",
                AlertLevel.CRITICAL: "🚨",
                AlertLevel.EMERGENCY: "🚨🚨🚨"
            }
            
            emoji = colors.get(alert.level, "ℹ️")
            
            message = f"""
{emoji} {alert.level.value} ALERT - {alert.timestamp.strftime('%H:%M:%S')}
Title: {alert.title}
Message: {alert.message}
Alert ID: {alert.alert_id}
"""
            
            if alert.details:
                message += f"Details: {json.dumps(alert.details, indent=2)}\n"
            
            if alert.trade_id:
                message += f"Trade ID: {alert.trade_id}\n"
            
            if alert.position_id:
                message += f"Position ID: {alert.position_id}\n"
            
            print(message)
            logger.info(f"Console alert: {alert.title}")
            
        except Exception as e:
            logger.error(f"Error sending console alert: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        try:
            channel_config = self.channels[AlertChannel.EMAIL]
            config = channel_config.config
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = config['to_email']
            msg['Subject'] = f"{alert.level.value} ALERT: {alert.title}"
            
            # Create email body
            body = f"""
Trading Alert Notification

Level: {alert.level.value}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Title: {alert.title}
Message: {alert.message}

Alert ID: {alert.alert_id}
"""
            
            if alert.trade_id:
                body += f"Trade ID: {alert.trade_id}\n"
            
            if alert.position_id:
                body += f"Position ID: {alert.position_id}\n"
            
            if alert.details:
                body += f"\nDetails:\n{json.dumps(alert.details, indent=2)}\n"
            
            body += f"""
---
This is an automated alert from your options trading system.
Please acknowledge this alert if action is required.
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    async def _send_sms_alert(self, alert: Alert):
        """Send SMS alert"""
        try:
            if not self.twilio_client:
                logger.warning("Twilio client not initialized")
                return
            
            channel_config = self.channels[AlertChannel.SMS]
            config = channel_config.config
            
            # Create SMS message (max 160 chars)
            message = f"{alert.level.value}: {alert.title}"
            if len(message) > 140:
                message = message[:137] + "..."
            
            # Send SMS
            self.twilio_client.messages.create(
                body=message,
                from_=config['from_number'],
                to=config['to_number']
            )
            
            logger.info(f"SMS alert sent: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending SMS alert: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        try:
            channel_config = self.channels[AlertChannel.SLACK]
            config = channel_config.config
            
            # Color coding for Slack
            colors = {
                AlertLevel.INFO: "#36a64f",      # Green
                AlertLevel.WARNING: "#ff9800",   # Orange
                AlertLevel.ERROR: "#f44336",     # Red
                AlertLevel.CRITICAL: "#9c27b0",  # Purple
                AlertLevel.EMERGENCY: "#000000"  # Black
            }
            
            color = colors.get(alert.level, "#36a64f")
            
            # Create Slack message
            slack_message = {
                "channel": config['channel'],
                "username": "Trading Bot",
                "icon_emoji": ":robot_face:",
                "attachments": [
                    {
                        "color": color,
                        "title": f"{alert.level.value} ALERT: {alert.title}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Alert ID",
                                "value": alert.alert_id,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                                "short": True
                            }
                        ],
                        "footer": "Options Trading System",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            if alert.trade_id:
                slack_message["attachments"][0]["fields"].append({
                    "title": "Trade ID",
                    "value": alert.trade_id,
                    "short": True
                })
            
            if alert.position_id:
                slack_message["attachments"][0]["fields"].append({
                    "title": "Position ID",
                    "value": alert.position_id,
                    "short": True
                })
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(config['webhook_url'], json=slack_message) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent: {alert.alert_id}")
                    else:
                        logger.error(f"Slack alert failed: {response.status}")
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        try:
            channel_config = self.channels[AlertChannel.WEBHOOK]
            config = channel_config.config
            
            # Create webhook payload
            payload = {
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp.isoformat(),
                "level": alert.level.value,
                "title": alert.title,
                "message": alert.message,
                "details": alert.details,
                "trade_id": alert.trade_id,
                "position_id": alert.position_id,
                "user_id": alert.user_id
            }
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                headers = {'Content-Type': 'application/json'}
                if 'auth_token' in config:
                    headers['Authorization'] = f"Bearer {config['auth_token']}"
                
                async with session.post(
                    config['webhook_url'], 
                    json=payload, 
                    headers=headers
                ) as response:
                    if response.status in [200, 201]:
                        logger.info(f"Webhook alert sent: {alert.alert_id}")
                    else:
                        logger.error(f"Webhook alert failed: {response.status}")
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
    
    async def _send_telegram_alert(self, alert: Alert):
        """Send Telegram alert"""
        try:
            channel_config = self.channels[AlertChannel.TELEGRAM]
            config = channel_config.config
            
            # Create Telegram message
            message = f"""
🚨 *{alert.level.value} ALERT*

*{alert.title}*

{alert.message}

Alert ID: `{alert.alert_id}`
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            
            if alert.trade_id:
                message += f"Trade ID: `{alert.trade_id}`\n"
            
            if alert.position_id:
                message += f"Position ID: `{alert.position_id}`\n"
            
            # Send to Telegram
            telegram_url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
            payload = {
                'chat_id': config['chat_id'],
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(telegram_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Telegram alert sent: {alert.alert_id}")
                    else:
                        logger.error(f"Telegram alert failed: {response.status}")
            
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    # Specific Alert Types for Trading
    
    async def send_trade_executed_alert(
        self,
        trade_id: str,
        symbol: str,
        strategy: str,
        pnl: float,
        details: Dict[str, Any]
    ):
        """Send trade execution alert"""
        level = AlertLevel.INFO if pnl >= 0 else AlertLevel.WARNING
        
        await self.send_alert(
            level=level,
            title=f"Trade Executed: {symbol} {strategy}",
            message=f"Trade {trade_id} executed. P&L: ${pnl:.2f}",
            details=details,
            trade_id=trade_id,
            channels=[AlertChannel.CONSOLE, AlertChannel.SLACK]
        )
    
    async def send_position_closed_alert(
        self,
        position_id: str,
        symbol: str,
        pnl: float,
        reason: str,
        details: Dict[str, Any]
    ):
        """Send position closed alert"""
        level = AlertLevel.INFO if pnl >= 0 else AlertLevel.WARNING
        
        await self.send_alert(
            level=level,
            title=f"Position Closed: {symbol}",
            message=f"Position {position_id} closed. P&L: ${pnl:.2f}. Reason: {reason}",
            details=details,
            position_id=position_id,
            channels=[AlertChannel.CONSOLE, AlertChannel.SLACK]
        )
    
    async def send_daily_loss_alert(
        self,
        daily_loss: float,
        daily_loss_pct: float,
        limit_pct: float,
        details: Dict[str, Any]
    ):
        """Send daily loss limit alert"""
        level = AlertLevel.CRITICAL if daily_loss_pct >= limit_pct else AlertLevel.WARNING
        
        await self.send_alert(
            level=level,
            title="Daily Loss Alert",
            message=f"Daily loss: ${daily_loss:.2f} ({daily_loss_pct:.1f}%). Limit: {limit_pct}%",
            details=details,
            channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL, AlertChannel.SLACK]
        )
    
    async def send_circuit_breaker_alert(
        self,
        breaker_level: str,
        trigger_reason: str,
        action_taken: str,
        affected_positions: List[str]
    ):
        """Send circuit breaker alert"""
        await self.send_alert(
            level=AlertLevel.CRITICAL,
            title=f"Circuit Breaker: {breaker_level}",
            message=f"Circuit breaker triggered. Reason: {trigger_reason}. Action: {action_taken}",
            details={
                'breaker_level': breaker_level,
                'trigger_reason': trigger_reason,
                'action_taken': action_taken,
                'affected_positions': affected_positions
            },
            channels=[AlertChannel.CONSOLE, AlertChannel.SMS, AlertChannel.EMAIL, AlertChannel.SLACK]
        )
    
    async def send_system_error_alert(
        self,
        error_type: str,
        error_message: str,
        error_details: Dict[str, Any]
    ):
        """Send system error alert"""
        await self.send_alert(
            level=AlertLevel.ERROR,
            title=f"System Error: {error_type}",
            message=error_message,
            details=error_details,
            channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL, AlertChannel.SLACK]
        )
    
    async def send_data_feed_alert(
        self,
        feed_name: str,
        status: str,
        last_update: datetime,
        details: Dict[str, Any]
    ):
        """Send data feed status alert"""
        level = AlertLevel.ERROR if status == "DOWN" else AlertLevel.WARNING
        
        await self.send_alert(
            level=level,
            title=f"Data Feed {status}: {feed_name}",
            message=f"Data feed {feed_name} is {status}. Last update: {last_update}",
            details=details,
            channels=[AlertChannel.CONSOLE, AlertChannel.SLACK]
        )
    
    async def send_performance_alert(
        self,
        metric_name: str,
        current_value: float,
        threshold: float,
        direction: str,  # "above" or "below"
        details: Dict[str, Any]
    ):
        """Send performance metric alert"""
        level = AlertLevel.WARNING
        
        if direction == "above" and current_value > threshold:
            message = f"{metric_name} is {current_value:.2f} (above threshold {threshold:.2f})"
        elif direction == "below" and current_value < threshold:
            message = f"{metric_name} is {current_value:.2f} (below threshold {threshold:.2f})"
        else:
            return  # No alert needed
        
        await self.send_alert(
            level=level,
            title=f"Performance Alert: {metric_name}",
            message=message,
            details=details,
            channels=[AlertChannel.CONSOLE, AlertChannel.SLACK]
        )
    
    # Alert Management
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.pending_alerts:
                alert = self.pending_alerts[alert_id]
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.utcnow()
                
                self.acknowledged_alerts[alert_id] = alert
                del self.pending_alerts[alert_id]
                
                logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True
            else:
                logger.warning(f"Alert not found for acknowledgment: {alert_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    def get_pending_alerts(self) -> List[Alert]:
        """Get list of pending alerts"""
        return list(self.pending_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        all_alerts = list(self.pending_alerts.values()) + list(self.acknowledged_alerts.values())
        recent_alerts = [alert for alert in all_alerts if alert.timestamp > cutoff_time]
        
        return sorted(recent_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics"""
        alerts = self.get_alert_history(hours)
        
        stats = {
            'total_alerts': len(alerts),
            'by_level': {},
            'by_channel': {},
            'pending_count': len(self.pending_alerts),
            'acknowledged_count': len(self.acknowledged_alerts),
            'period_hours': hours
        }
        
        # Count by level
        for alert in alerts:
            level = alert.level.value
            stats['by_level'][level] = stats['by_level'].get(level, 0) + 1
            
            # Count by channel
            for channel in alert.channels:
                channel_name = channel.value
                stats['by_channel'][channel_name] = stats['by_channel'].get(channel_name, 0) + 1
        
        return stats


# Example usage
async def main():
    """Example usage of RealMoneyAlertSystem"""
    
    # Initialize alert system
    alert_system = RealMoneyAlertSystem()
    
    # Send different types of alerts
    await alert_system.send_trade_executed_alert(
        trade_id="trade_123",
        symbol="SPY",
        strategy="bull_put_spread",
        pnl=45.50,
        details={"strike": 500, "dte": 30}
    )
    
    await alert_system.send_circuit_breaker_alert(
        breaker_level="PORTFOLIO",
        trigger_reason="Daily loss limit exceeded",
        action_taken="Stop trading",
        affected_positions=["pos_1", "pos_2"]
    )
    
    await alert_system.send_performance_alert(
        metric_name="Sharpe Ratio",
        current_value=1.2,
        threshold=1.5,
        direction="below",
        details={"account_balance": 25000}
    )
    
    # Get statistics
    stats = alert_system.get_alert_statistics(24)
    print(f"Alert statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
