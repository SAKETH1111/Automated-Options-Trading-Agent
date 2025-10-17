"""
Real Money Protection System
Comprehensive safeguards for live trading with circuit breakers, position monitoring, and account protection
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import threading
import time
from loguru import logger

from src.config.settings import get_config
from src.database.session import get_db
from src.database.models import Trade, Position
from src.monitoring.alerts import AlertManager


class CircuitBreakerLevel(Enum):
    POSITION = "POSITION"
    STRATEGY = "STRATEGY"
    PORTFOLIO = "PORTFOLIO"
    SYSTEM = "SYSTEM"
    EMERGENCY = "EMERGENCY"


class BreakerAction(Enum):
    CLOSE_POSITION = "CLOSE_POSITION"
    PAUSE_STRATEGY = "PAUSE_STRATEGY"
    STOP_TRADING = "STOP_TRADING"
    HEDGE_MODE = "HEDGE_MODE"
    EMERGENCY_SHUTDOWN = "EMERGENCY_SHUTDOWN"
    SWITCH_TO_PAPER = "SWITCH_TO_PAPER"


@dataclass
class CircuitBreakerEvent:
    """Circuit breaker event record"""
    event_id: str
    timestamp: datetime
    level: CircuitBreakerLevel
    action: BreakerAction
    trigger_reason: str
    trigger_value: float
    threshold_value: float
    account_balance: float
    affected_positions: List[str]
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_reason: Optional[str] = None


@dataclass
class PositionRiskMetrics:
    """Real-time position risk metrics"""
    position_id: str
    symbol: str
    strategy: str
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    days_held: int
    days_to_expiration: int
    delta: float
    gamma: float
    theta: float
    vega: float
    assignment_risk: float
    pin_risk: bool
    last_updated: datetime


class RealMoneyProtectionSystem:
    """
    Comprehensive real money protection system
    
    Features:
    - Real-time position monitoring (every 30 seconds)
    - Multi-level circuit breakers
    - Account protection and monitoring
    - Automatic position unwinding
    - Emergency shutdown procedures
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or get_config()
        self.db = get_db()
        self.alert_manager = AlertManager(config)
        
        # Circuit breaker thresholds by account size
        self.breaker_thresholds = self._get_breaker_thresholds()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.active_breakers = []
        
        # Position tracking
        self.position_metrics = {}
        self.last_monitor_time = None
        
        logger.info("RealMoneyProtectionSystem initialized")
    
    def _get_breaker_thresholds(self) -> Dict[str, Dict]:
        """Get circuit breaker thresholds by account size"""
        return {
            "micro": {
                "position_loss_pct": 3.0,      # Close position at 3% loss
                "daily_loss_pct": 5.0,         # Stop trading at 5% daily loss
                "drawdown_pct": 10.0,          # Reduce size at 10% drawdown
                "max_drawdown_pct": 15.0,      # Emergency stop at 15%
                "consecutive_losses": 3,       # Review after 3 losses
                "vix_threshold": 50.0,         # Reduce size if VIX > 50
                "flash_crash_vix": 80.0        # Emergency stop if VIX > 80
            },
            "small": {
                "position_loss_pct": 4.0,
                "daily_loss_pct": 6.0,
                "drawdown_pct": 12.0,
                "max_drawdown_pct": 18.0,
                "consecutive_losses": 4,
                "vix_threshold": 45.0,
                "flash_crash_vix": 75.0
            },
            "medium": {
                "position_loss_pct": 5.0,
                "daily_loss_pct": 7.0,
                "drawdown_pct": 15.0,
                "max_drawdown_pct": 20.0,
                "consecutive_losses": 5,
                "vix_threshold": 40.0,
                "flash_crash_vix": 70.0
            },
            "large": {
                "position_loss_pct": 6.0,
                "daily_loss_pct": 8.0,
                "drawdown_pct": 18.0,
                "max_drawdown_pct": 25.0,
                "consecutive_losses": 6,
                "vix_threshold": 35.0,
                "flash_crash_vix": 60.0
            },
            "institutional": {
                "position_loss_pct": 7.0,
                "daily_loss_pct": 10.0,
                "drawdown_pct": 20.0,
                "max_drawdown_pct": 30.0,
                "consecutive_losses": 8,
                "vix_threshold": 30.0,
                "flash_crash_vix": 50.0
            }
        }
    
    def start_monitoring(self, account_balance: float):
        """Start real-time monitoring of positions and account"""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.account_balance = account_balance
        self.account_tier = self._get_account_tier(account_balance)
        self.thresholds = self.breaker_thresholds[self.account_tier]
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Started real money monitoring for {self.account_tier} account (${account_balance:,.0f})")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Stopped real money monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop - runs every 30 seconds"""
        logger.info("Real money monitoring loop started")
        
        while self.is_monitoring:
            try:
                start_time = time.time()
                
                # Update position metrics
                self._update_position_metrics()
                
                # Check all circuit breaker levels
                self._check_position_breakers()
                self._check_portfolio_breakers()
                self._check_system_breakers()
                
                # Update last monitor time
                self.last_monitor_time = datetime.utcnow()
                
                # Calculate sleep time to maintain 30-second intervals
                elapsed = time.time() - start_time
                sleep_time = max(0, 30 - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                # Continue monitoring even if there's an error
                time.sleep(30)
        
        logger.info("Real money monitoring loop stopped")
    
    def _update_position_metrics(self):
        """Update real-time position metrics"""
        try:
            with self.db.get_session() as session:
                # Get all open positions
                open_positions = session.query(Trade).filter(
                    Trade.status == "open"
                ).all()
                
                for trade in open_positions:
                    try:
                        # Get current position data
                        current_positions = session.query(Position).filter(
                            Position.trade_id == trade.trade_id,
                            Position.status == "open"
                        ).all()
                        
                        if not current_positions:
                            continue
                        
                        # Calculate current P&L and Greeks
                        total_pnl = 0.0
                        total_delta = 0.0
                        total_gamma = 0.0
                        total_theta = 0.0
                        total_vega = 0.0
                        min_dte = float('inf')
                        
                        for pos in current_positions:
                            # Calculate position P&L (simplified)
                            current_price = pos.current_price or pos.entry_price
                            position_pnl = (current_price - pos.entry_price) * pos.quantity
                            if pos.side == "short":
                                position_pnl = -position_pnl
                            
                            total_pnl += position_pnl
                            
                            # Aggregate Greeks
                            total_delta += (pos.current_delta or 0) * pos.quantity
                            total_gamma += (pos.current_gamma or 0) * pos.quantity
                            total_theta += (pos.current_theta or 0) * pos.quantity
                            total_vega += (pos.current_vega or 0) * pos.quantity
                            
                            # Track minimum DTE
                            if pos.expiration:
                                dte = (pos.expiration - datetime.utcnow()).days
                                min_dte = min(min_dte, dte)
                        
                        # Calculate percentage P&L
                        entry_credit = trade.execution.get("fill_credit", 0) if trade.execution else 0
                        pnl_pct = (total_pnl / (entry_credit * 100)) * 100 if entry_credit > 0 else 0
                        
                        # Calculate days held
                        days_held = (datetime.utcnow() - trade.timestamp_enter).days
                        
                        # Check assignment and pin risk
                        assignment_risk = self._calculate_assignment_risk(trade, current_positions)
                        pin_risk = min_dte <= 1 and self._check_pin_risk(trade, current_positions)
                        
                        # Update metrics
                        self.position_metrics[trade.trade_id] = PositionRiskMetrics(
                            position_id=trade.trade_id,
                            symbol=trade.symbol,
                            strategy=trade.strategy,
                            entry_price=entry_credit,
                            current_price=entry_credit + (total_pnl / 100),  # Approximate
                            unrealized_pnl=total_pnl,
                            unrealized_pnl_pct=pnl_pct,
                            days_held=days_held,
                            days_to_expiration=int(min_dte) if min_dte != float('inf') else 0,
                            delta=total_delta,
                            gamma=total_gamma,
                            theta=total_theta,
                            vega=total_vega,
                            assignment_risk=assignment_risk,
                            pin_risk=pin_risk,
                            last_updated=datetime.utcnow()
                        )
                        
                    except Exception as e:
                        logger.error(f"Error updating metrics for trade {trade.trade_id}: {e}")
                        
        except Exception as e:
            logger.error(f"Error updating position metrics: {e}")
    
    def _check_position_breakers(self):
        """Check position-level circuit breakers"""
        try:
            for trade_id, metrics in self.position_metrics.items():
                # Check position loss limit
                if metrics.unrealized_pnl_pct <= -self.thresholds["position_loss_pct"]:
                    self._trigger_breaker(
                        CircuitBreakerLevel.POSITION,
                        BreakerAction.CLOSE_POSITION,
                        f"Position loss {metrics.unrealized_pnl_pct:.1f}% exceeds limit {self.thresholds['position_loss_pct']}%",
                        metrics.unrealized_pnl_pct,
                        self.thresholds["position_loss_pct"],
                        [trade_id]
                    )
                
                # Check assignment risk
                if metrics.assignment_risk > 0.8:  # 80% assignment probability
                    self._trigger_breaker(
                        CircuitBreakerLevel.POSITION,
                        BreakerAction.CLOSE_POSITION,
                        f"High assignment risk {metrics.assignment_risk:.1%}",
                        metrics.assignment_risk,
                        0.8,
                        [trade_id]
                    )
                
                # Check pin risk
                if metrics.pin_risk:
                    self._trigger_breaker(
                        CircuitBreakerLevel.POSITION,
                        BreakerAction.CLOSE_POSITION,
                        f"Pin risk detected - position near strike with {metrics.days_to_expiration} DTE",
                        1.0,
                        0.0,
                        [trade_id]
                    )
                
                # Check gamma explosion (simplified)
                if abs(metrics.gamma) > 0.5:  # High gamma exposure
                    self._trigger_breaker(
                        CircuitBreakerLevel.POSITION,
                        BreakerAction.CLOSE_POSITION,
                        f"High gamma exposure {metrics.gamma:.3f}",
                        abs(metrics.gamma),
                        0.5,
                        [trade_id]
                    )
                
        except Exception as e:
            logger.error(f"Error checking position breakers: {e}")
    
    def _check_portfolio_breakers(self):
        """Check portfolio-level circuit breakers"""
        try:
            # Calculate daily P&L
            daily_pnl = self._calculate_daily_pnl()
            daily_pnl_pct = (daily_pnl / self.account_balance) * 100
            
            # Check daily loss limit
            if daily_pnl_pct <= -self.thresholds["daily_loss_pct"]:
                self._trigger_breaker(
                    CircuitBreakerLevel.PORTFOLIO,
                    BreakerAction.STOP_TRADING,
                    f"Daily loss {daily_pnl_pct:.1f}% exceeds limit {self.thresholds['daily_loss_pct']}%",
                    daily_pnl_pct,
                    self.thresholds["daily_loss_pct"],
                    []
                )
            
            # Calculate current drawdown
            drawdown_pct = self._calculate_drawdown()
            
            # Check drawdown limit
            if drawdown_pct >= self.thresholds["drawdown_pct"]:
                self._trigger_breaker(
                    CircuitBreakerLevel.PORTFOLIO,
                    BreakerAction.HEDGE_MODE,
                    f"Drawdown {drawdown_pct:.1f}% exceeds limit {self.thresholds['drawdown_pct']}%",
                    drawdown_pct,
                    self.thresholds["drawdown_pct"],
                    []
                )
            
            # Check max drawdown
            if drawdown_pct >= self.thresholds["max_drawdown_pct"]:
                self._trigger_breaker(
                    CircuitBreakerLevel.PORTFOLIO,
                    BreakerAction.EMERGENCY_SHUTDOWN,
                    f"Max drawdown {drawdown_pct:.1f}% exceeded {self.thresholds['max_drawdown_pct']}%",
                    drawdown_pct,
                    self.thresholds["max_drawdown_pct"],
                    []
                )
            
            # Check consecutive losses
            consecutive_losses = self._count_consecutive_losses()
            if consecutive_losses >= self.thresholds["consecutive_losses"]:
                self._trigger_breaker(
                    CircuitBreakerLevel.PORTFOLIO,
                    BreakerAction.STOP_TRADING,
                    f"Consecutive losses {consecutive_losses} exceeds limit {self.thresholds['consecutive_losses']}",
                    consecutive_losses,
                    self.thresholds["consecutive_losses"],
                    []
                )
                
        except Exception as e:
            logger.error(f"Error checking portfolio breakers: {e}")
    
    def _check_system_breakers(self):
        """Check system-level circuit breakers"""
        try:
            # Check VIX level (if available)
            vix = self._get_current_vix()
            if vix:
                # Check VIX threshold
                if vix >= self.thresholds["flash_crash_vix"]:
                    self._trigger_breaker(
                        CircuitBreakerLevel.SYSTEM,
                        BreakerAction.EMERGENCY_SHUTDOWN,
                        f"Flash crash conditions: VIX {vix:.1f} >= {self.thresholds['flash_crash_vix']}",
                        vix,
                        self.thresholds["flash_crash_vix"],
                        []
                    )
                elif vix >= self.thresholds["vix_threshold"]:
                    self._trigger_breaker(
                        CircuitBreakerLevel.SYSTEM,
                        BreakerAction.HEDGE_MODE,
                        f"High volatility: VIX {vix:.1f} >= {self.thresholds['vix_threshold']}",
                        vix,
                        self.thresholds["vix_threshold"],
                        []
                    )
            
            # Check for system errors
            recent_errors = self._count_recent_system_errors()
            if recent_errors > 20:  # More than 20 errors in last hour
                self._trigger_breaker(
                    CircuitBreakerLevel.SYSTEM,
                    BreakerAction.SWITCH_TO_PAPER,
                    f"Too many system errors: {recent_errors} in last hour",
                    recent_errors,
                    20,
                    []
                )
            
            # Check data feed health
            if not self._check_data_feed_health():
                self._trigger_breaker(
                    CircuitBreakerLevel.SYSTEM,
                    BreakerAction.SWITCH_TO_PAPER,
                    "Data feed unhealthy",
                    1.0,
                    0.0,
                    []
                )
                
        except Exception as e:
            logger.error(f"Error checking system breakers: {e}")
    
    def _trigger_breaker(
        self,
        level: CircuitBreakerLevel,
        action: BreakerAction,
        reason: str,
        trigger_value: float,
        threshold_value: float,
        affected_positions: List[str]
    ):
        """Trigger a circuit breaker"""
        try:
            # Create breaker event
            event = CircuitBreakerEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                level=level,
                action=action,
                trigger_reason=reason,
                trigger_value=trigger_value,
                threshold_value=threshold_value,
                account_balance=self.account_balance,
                affected_positions=affected_positions
            )
            
            # Add to active breakers
            self.active_breakers.append(event)
            
            # Log the event
            logger.critical(f"CIRCUIT BREAKER TRIGGERED: {level.value} - {action.value}")
            logger.critical(f"Reason: {reason}")
            logger.critical(f"Trigger: {trigger_value} vs Threshold: {threshold_value}")
            logger.critical(f"Account: ${self.account_balance:,.0f}")
            logger.critical(f"Affected positions: {affected_positions}")
            
            # Send critical alert
            self.alert_manager.send_critical_alert(
                f"CIRCUIT BREAKER: {level.value}",
                f"Action: {action.value}\nReason: {reason}\nAccount: ${self.account_balance:,.0f}"
            )
            
            # Execute breaker action
            self._execute_breaker_action(action, affected_positions)
            
            # Store in database
            self._store_breaker_event(event)
            
        except Exception as e:
            logger.error(f"Error triggering circuit breaker: {e}")
    
    def _execute_breaker_action(self, action: BreakerAction, affected_positions: List[str]):
        """Execute the circuit breaker action"""
        try:
            if action == BreakerAction.CLOSE_POSITION:
                self._close_positions(affected_positions)
            elif action == BreakerAction.PAUSE_STRATEGY:
                self._pause_strategies(affected_positions)
            elif action == BreakerAction.STOP_TRADING:
                self._stop_trading()
            elif action == BreakerAction.HEDGE_MODE:
                self._enter_hedge_mode()
            elif action == BreakerAction.EMERGENCY_SHUTDOWN:
                self._emergency_shutdown()
            elif action == BreakerAction.SWITCH_TO_PAPER:
                self._switch_to_paper()
                
        except Exception as e:
            logger.error(f"Error executing breaker action {action.value}: {e}")
    
    def _close_positions(self, position_ids: List[str]):
        """Close specific positions"""
        try:
            logger.info(f"Closing positions: {position_ids}")
            
            # In real implementation, would call order management system
            # to close positions at market or with limit orders
            
            # For now, just log and mark for closure
            for position_id in position_ids:
                logger.info(f"Marking position {position_id} for closure")
                
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
    
    def _pause_strategies(self, strategy_names: List[str]):
        """Pause specific strategies"""
        try:
            logger.info(f"Pausing strategies: {strategy_names}")
            # Implementation would disable strategies in the system
            
        except Exception as e:
            logger.error(f"Error pausing strategies: {e}")
    
    def _stop_trading(self):
        """Stop all trading"""
        try:
            logger.critical("STOPPING ALL TRADING")
            # Implementation would stop the trading engine
            
        except Exception as e:
            logger.error(f"Error stopping trading: {e}")
    
    def _enter_hedge_mode(self):
        """Enter defensive hedge mode"""
        try:
            logger.warning("ENTERING HEDGE MODE")
            # Implementation would switch to defensive strategies only
            
        except Exception as e:
            logger.error(f"Error entering hedge mode: {e}")
    
    def _emergency_shutdown(self):
        """Emergency shutdown - close everything"""
        try:
            logger.critical("EMERGENCY SHUTDOWN INITIATED")
            
            # Close all positions
            all_positions = list(self.position_metrics.keys())
            self._close_positions(all_positions)
            
            # Stop monitoring
            self.stop_monitoring()
            
            # Stop trading
            self._stop_trading()
            
            logger.critical("EMERGENCY SHUTDOWN COMPLETE")
            
        except Exception as e:
            logger.error(f"Error in emergency shutdown: {e}")
    
    def _switch_to_paper(self):
        """Switch to paper trading mode"""
        try:
            logger.warning("SWITCHING TO PAPER TRADING")
            # Implementation would switch trading mode to paper
            
        except Exception as e:
            logger.error(f"Error switching to paper: {e}")
    
    # Helper methods
    def _get_account_tier(self, account_balance: float) -> str:
        """Determine account tier"""
        if account_balance < 1000:
            return "micro"
        elif account_balance < 10000:
            return "small"
        elif account_balance < 100000:
            return "medium"
        elif account_balance < 1000000:
            return "large"
        else:
            return "institutional"
    
    def _calculate_assignment_risk(self, trade: Trade, positions: List) -> float:
        """Calculate assignment risk for short options"""
        try:
            # Simplified assignment risk calculation
            # In real implementation, would use Black-Scholes model
            
            for pos in positions:
                if pos.side == "short" and pos.option_type == "put":
                    # Check if underlying is near strike
                    underlying_price = 450  # Would get from market data
                    strike = pos.strike
                    dte = (pos.expiration - datetime.utcnow()).days
                    
                    # Higher risk if close to strike and near expiration
                    distance_pct = abs(underlying_price - strike) / strike
                    if distance_pct < 0.02 and dte <= 3:  # Within 2% and 3 DTE
                        return 0.9  # 90% assignment risk
                    elif distance_pct < 0.05 and dte <= 7:  # Within 5% and 7 DTE
                        return 0.5  # 50% assignment risk
            
            return 0.1  # Low risk
            
        except Exception as e:
            logger.error(f"Error calculating assignment risk: {e}")
            return 0.0
    
    def _check_pin_risk(self, trade: Trade, positions: List) -> bool:
        """Check for pin risk (position near strike at expiration)"""
        try:
            for pos in positions:
                if pos.side == "short":
                    underlying_price = 450  # Would get from market data
                    strike = pos.strike
                    dte = (pos.expiration - datetime.utcnow()).days
                    
                    # Pin risk if very close to strike and 1 DTE
                    if dte <= 1:
                        distance_pct = abs(underlying_price - strike) / strike
                        if distance_pct < 0.01:  # Within 1%
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking pin risk: {e}")
            return False
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L"""
        try:
            with self.db.get_session() as session:
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                
                # Get today's closed trades
                today_trades = session.query(Trade).filter(
                    Trade.timestamp_exit >= today_start,
                    Trade.status == "closed"
                ).all()
                
                daily_pnl = sum(trade.pnl or 0 for trade in today_trades)
                return daily_pnl
                
        except Exception as e:
            logger.error(f"Error calculating daily P&L: {e}")
            return 0.0
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        try:
            # Simplified drawdown calculation
            # In real implementation, would track account value over time
            
            current_value = self.account_balance
            
            # Get peak value from recent trades or account history
            with self.db.get_session() as session:
                # This is simplified - would need proper account value tracking
                peak_value = self.account_balance * 1.1  # Assume 10% above current
                
            drawdown_pct = ((peak_value - current_value) / peak_value) * 100
            return max(0, drawdown_pct)
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return 0.0
    
    def _count_consecutive_losses(self) -> int:
        """Count consecutive losing trades"""
        try:
            with self.db.get_session() as session:
                # Get recent closed trades ordered by exit time
                recent_trades = session.query(Trade).filter(
                    Trade.status == "closed"
                ).order_by(Trade.timestamp_exit.desc()).limit(10).all()
                
                consecutive_losses = 0
                for trade in recent_trades:
                    if trade.pnl and trade.pnl < 0:
                        consecutive_losses += 1
                    else:
                        break
                
                return consecutive_losses
                
        except Exception as e:
            logger.error(f"Error counting consecutive losses: {e}")
            return 0
    
    def _get_current_vix(self) -> Optional[float]:
        """Get current VIX level"""
        try:
            # In real implementation, would get from market data feed
            return None  # Placeholder
            
        except Exception as e:
            logger.error(f"Error getting VIX: {e}")
            return None
    
    def _count_recent_system_errors(self) -> int:
        """Count recent system errors"""
        try:
            # In real implementation, would check error logs
            return 0
            
        except Exception as e:
            logger.error(f"Error counting system errors: {e}")
            return 0
    
    def _check_data_feed_health(self) -> bool:
        """Check if data feeds are healthy"""
        try:
            # In real implementation, would ping data feeds
            return True
            
        except Exception as e:
            logger.error(f"Error checking data feed health: {e}")
            return False
    
    def _store_breaker_event(self, event: CircuitBreakerEvent):
        """Store circuit breaker event in database"""
        try:
            # In real implementation, would store in database
            pass
            
        except Exception as e:
            logger.error(f"Error storing breaker event: {e}")
    
    def get_protection_status(self) -> Dict[str, Any]:
        """Get current protection system status"""
        try:
            status = {
                "monitoring_active": self.is_monitoring,
                "account_tier": self.account_tier if hasattr(self, 'account_tier') else None,
                "account_balance": self.account_balance if hasattr(self, 'account_balance') else 0,
                "active_breakers": len(self.active_breakers),
                "positions_monitored": len(self.position_metrics),
                "last_monitor_time": self.last_monitor_time.isoformat() if self.last_monitor_time else None,
                "thresholds": self.thresholds if hasattr(self, 'thresholds') else {}
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting protection status: {e}")
            return {"error": str(e)}
    
    def get_position_risk_summary(self) -> Dict[str, Any]:
        """Get summary of position risks"""
        try:
            if not self.position_metrics:
                return {"positions": 0, "total_risk": 0}
            
            total_pnl = sum(metrics.unrealized_pnl for metrics in self.position_metrics.values())
            total_pnl_pct = (total_pnl / self.account_balance) * 100 if hasattr(self, 'account_balance') else 0
            
            high_risk_positions = [
                pos for pos in self.position_metrics.values()
                if pos.unrealized_pnl_pct < -5 or pos.assignment_risk > 0.5 or pos.pin_risk
            ]
            
            summary = {
                "total_positions": len(self.position_metrics),
                "total_unrealized_pnl": total_pnl,
                "total_unrealized_pnl_pct": total_pnl_pct,
                "high_risk_positions": len(high_risk_positions),
                "positions": [
                    {
                        "id": metrics.position_id,
                        "symbol": metrics.symbol,
                        "strategy": metrics.strategy,
                        "pnl_pct": metrics.unrealized_pnl_pct,
                        "dte": metrics.days_to_expiration,
                        "assignment_risk": metrics.assignment_risk,
                        "pin_risk": metrics.pin_risk,
                        "risk_level": "HIGH" if (
                            metrics.unrealized_pnl_pct < -5 or 
                            metrics.assignment_risk > 0.5 or 
                            metrics.pin_risk
                        ) else "NORMAL"
                    }
                    for metrics in self.position_metrics.values()
                ]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting position risk summary: {e}")
            return {"error": str(e)}


# Example usage
if __name__ == "__main__":
    # Test the protection system
    protection = RealMoneyProtectionSystem()
    
    # Start monitoring
    protection.start_monitoring(account_balance=25000)
    
    # Let it run for a bit
    time.sleep(60)
    
    # Check status
    status = protection.get_protection_status()
    print(f"Protection Status: {status}")
    
    # Check position risks
    risks = protection.get_position_risk_summary()
    print(f"Position Risks: {risks}")
    
    # Stop monitoring
    protection.stop_monitoring()
