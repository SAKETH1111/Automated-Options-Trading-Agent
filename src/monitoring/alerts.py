"""Alert management system"""

from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger
import requests

from src.config.settings import get_settings


class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self):
        self.settings = get_settings()
        self.alert_email = self.settings.alert_email
        self.alert_webhook_url = self.settings.alert_webhook_url
        
        # Alert history to prevent spam
        self.recent_alerts = []
        self.max_recent_alerts = 100
        
        logger.info("Alert Manager initialized")
    
    def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "info",
        data: Optional[Dict] = None
    ):
        """
        Send an alert
        
        Args:
            alert_type: Type of alert (trade_executed, position_closed, error, etc.)
            message: Alert message
            severity: info, warning, error, critical
            data: Additional data
        """
        try:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "type": alert_type,
                "severity": severity,
                "message": message,
                "data": data or {},
            }
            
            # Log the alert
            log_func = getattr(logger, severity, logger.info)
            log_func(f"ALERT [{alert_type}]: {message}")
            
            # Store in recent alerts
            self.recent_alerts.append(alert)
            if len(self.recent_alerts) > self.max_recent_alerts:
                self.recent_alerts.pop(0)
            
            # Send to external channels if critical or error
            if severity in ["error", "critical"]:
                self._send_to_webhook(alert)
                # Could also send email here
        
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def _send_to_webhook(self, alert: Dict):
        """Send alert to webhook (e.g., Slack, Discord)"""
        try:
            if not self.alert_webhook_url:
                return
            
            payload = {
                "text": f"🚨 {alert['severity'].upper()}: {alert['message']}",
                "attachments": [
                    {
                        "color": self._get_color_for_severity(alert['severity']),
                        "fields": [
                            {
                                "title": "Type",
                                "value": alert['type'],
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert['timestamp'],
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(
                self.alert_webhook_url,
                json=payload,
                timeout=5
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to send webhook: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
    
    def _get_color_for_severity(self, severity: str) -> str:
        """Get color code for severity"""
        colors = {
            "info": "#36a64f",      # Green
            "warning": "#ff9900",    # Orange
            "error": "#ff0000",      # Red
            "critical": "#8b0000",   # Dark red
        }
        return colors.get(severity, "#cccccc")
    
    def alert_trade_executed(self, trade_id: str, symbol: str, strategy: str, contracts: int):
        """Alert when trade is executed"""
        self.send_alert(
            "trade_executed",
            f"Opened {strategy} position in {symbol} ({contracts} contracts)",
            "info",
            {"trade_id": trade_id, "symbol": symbol, "strategy": strategy, "contracts": contracts}
        )
    
    def alert_position_closed(self, trade_id: str, symbol: str, pnl: float, reason: str):
        """Alert when position is closed"""
        severity = "info" if pnl >= 0 else "warning"
        self.send_alert(
            "position_closed",
            f"Closed {symbol} position: ${pnl:.2f} ({reason})",
            severity,
            {"trade_id": trade_id, "symbol": symbol, "pnl": pnl, "reason": reason}
        )
    
    def alert_daily_loss_limit(self, current_loss: float, limit: float):
        """Alert when approaching or hitting daily loss limit"""
        self.send_alert(
            "daily_loss_limit",
            f"Daily loss: ${current_loss:.2f} (limit: ${limit:.2f})",
            "critical",
            {"current_loss": current_loss, "limit": limit}
        )
    
    def alert_system_error(self, error_type: str, error_message: str):
        """Alert on system errors"""
        self.send_alert(
            "system_error",
            f"{error_type}: {error_message}",
            "error",
            {"error_type": error_type, "error_message": error_message}
        )
    
    def get_recent_alerts(self, count: int = 20) -> List[Dict]:
        """Get recent alerts"""
        return self.recent_alerts[-count:]













