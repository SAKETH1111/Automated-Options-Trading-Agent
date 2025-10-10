"""
Circuit Breaker Module
Advanced circuit breakers for trading protection
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from loguru import logger

from src.database.models import Trade, IndexTickData


class CircuitBreaker:
    """
    Advanced circuit breaker system
    Pauses trading during extreme conditions
    """
    
    def __init__(
        self,
        db_session: Session,
        total_capital: float = 10000.0
    ):
        """
        Initialize circuit breaker
        
        Args:
            db_session: Database session
            total_capital: Total trading capital
        """
        self.db = db_session
        self.total_capital = total_capital
        
        # Circuit breaker thresholds
        self.daily_loss_limit_pct = 0.03  # 3% daily loss
        self.max_drawdown_limit_pct = 0.15  # 15% max drawdown
        self.extreme_volatility_threshold = 3.0  # 3x normal volatility
        self.max_consecutive_losses = 5
        
        # State
        self.is_tripped = False
        self.trip_reason = None
        self.trip_timestamp = None
        self.reset_time_hours = 24  # Reset after 24 hours
        
        logger.info("Circuit Breaker initialized")
    
    def check_circuit_breaker(self) -> Dict:
        """
        Check if circuit breaker should trip
        
        Returns:
            Dictionary with status and reason
        """
        result = {
            'tripped': False,
            'reason': None,
            'can_trade': True,
            'warnings': []
        }
        
        try:
            # Check if already tripped and should reset
            if self.is_tripped:
                if self._should_reset():
                    self._reset()
                    result['warnings'].append('Circuit breaker reset')
                else:
                    result['tripped'] = True
                    result['reason'] = self.trip_reason
                    result['can_trade'] = False
                    return result
            
            # Check 1: Daily loss limit
            daily_loss = self._get_daily_loss()
            daily_loss_pct = abs(daily_loss / self.total_capital)
            
            if daily_loss < 0 and daily_loss_pct >= self.daily_loss_limit_pct:
                self._trip('DAILY_LOSS_LIMIT', 
                          f"Daily loss ${daily_loss:.2f} ({daily_loss_pct:.1%}) exceeded limit")
                result['tripped'] = True
                result['reason'] = self.trip_reason
                result['can_trade'] = False
                return result
            
            # Check 2: Max drawdown
            drawdown_pct = self._get_current_drawdown()
            
            if drawdown_pct >= self.max_drawdown_limit_pct:
                self._trip('MAX_DRAWDOWN',
                          f"Drawdown {drawdown_pct:.1%} exceeded limit {self.max_drawdown_limit_pct:.1%}")
                result['tripped'] = True
                result['reason'] = self.trip_reason
                result['can_trade'] = False
                return result
            
            # Check 3: Extreme volatility
            for symbol in ['SPY', 'QQQ']:
                vol_ratio = self._get_volatility_ratio(symbol)
                
                if vol_ratio >= self.extreme_volatility_threshold:
                    self._trip('EXTREME_VOLATILITY',
                              f"{symbol} volatility {vol_ratio:.1f}x normal - pausing trading")
                    result['tripped'] = True
                    result['reason'] = self.trip_reason
                    result['can_trade'] = False
                    return result
            
            # Check 4: Consecutive losses
            consecutive_losses = self._get_consecutive_losses()
            
            if consecutive_losses >= self.max_consecutive_losses:
                self._trip('CONSECUTIVE_LOSSES',
                          f"{consecutive_losses} consecutive losses - pausing to review")
                result['tripped'] = True
                result['reason'] = self.trip_reason
                result['can_trade'] = False
                return result
            
            # Add warnings for approaching limits
            if daily_loss_pct >= self.daily_loss_limit_pct * 0.75:
                result['warnings'].append(f"Approaching daily loss limit: {daily_loss_pct:.1%}")
            
            if drawdown_pct >= self.max_drawdown_limit_pct * 0.75:
                result['warnings'].append(f"Approaching max drawdown: {drawdown_pct:.1%}")
            
            if consecutive_losses >= self.max_consecutive_losses - 2:
                result['warnings'].append(f"Consecutive losses: {consecutive_losses}")
            
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
        
        return result
    
    def _trip(self, reason: str, message: str):
        """Trip the circuit breaker"""
        self.is_tripped = True
        self.trip_reason = reason
        self.trip_timestamp = datetime.utcnow()
        
        logger.warning(f"🚨 CIRCUIT BREAKER TRIPPED: {message}")
    
    def _reset(self):
        """Reset the circuit breaker"""
        logger.info(f"Circuit breaker reset (was tripped for {self.trip_reason})")
        
        self.is_tripped = False
        self.trip_reason = None
        self.trip_timestamp = None
    
    def _should_reset(self) -> bool:
        """Check if circuit breaker should reset"""
        if not self.is_tripped or not self.trip_timestamp:
            return False
        
        hours_since_trip = (datetime.utcnow() - self.trip_timestamp).total_seconds() / 3600
        
        return hours_since_trip >= self.reset_time_hours
    
    def _get_daily_loss(self) -> float:
        """Get today's realized loss"""
        try:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            
            closed_today = self.db.query(Trade).filter(
                Trade.timestamp_exit >= today,
                Trade.status == 'closed'
            ).all()
            
            return sum(t.pnl for t in closed_today)
            
        except Exception as e:
            logger.error(f"Error getting daily loss: {e}")
            return 0.0
    
    def _get_current_drawdown(self) -> float:
        """Get current drawdown percentage"""
        try:
            all_trades = self.db.query(Trade).filter(
                Trade.status == 'closed'
            ).order_by(Trade.timestamp_exit.asc()).all()
            
            if not all_trades:
                return 0.0
            
            equity = self.total_capital
            peak = equity
            max_dd_pct = 0
            
            for trade in all_trades:
                equity += trade.pnl
                
                if equity > peak:
                    peak = equity
                
                dd_pct = ((peak - equity) / peak) if peak > 0 else 0
                
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
            
            return max_dd_pct
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return 0.0
    
    def _get_volatility_ratio(self, symbol: str) -> float:
        """Get current volatility vs historical"""
        try:
            cutoff = datetime.utcnow() - timedelta(days=30)
            
            data = self.db.query(IndexTickData).filter(
                IndexTickData.symbol == symbol,
                IndexTickData.timestamp >= cutoff
            ).all()
            
            if len(data) < 100:
                return 1.0
            
            prices = [d.price for d in data]
            returns = np.diff(np.log(prices))
            
            current_vol = np.std(returns[-100:])
            historical_vol = np.std(returns)
            
            return current_vol / historical_vol if historical_vol > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Error getting volatility ratio: {e}")
            return 1.0
    
    def _get_consecutive_losses(self) -> int:
        """Get number of consecutive losses"""
        try:
            recent_trades = self.db.query(Trade).filter(
                Trade.status == 'closed'
            ).order_by(Trade.timestamp_exit.desc()).limit(20).all()
            
            consecutive = 0
            for trade in recent_trades:
                if trade.pnl < 0:
                    consecutive += 1
                else:
                    break
            
            return consecutive
            
        except Exception as e:
            logger.error(f"Error getting consecutive losses: {e}")
            return 0
    
    def force_trip(self, reason: str):
        """Manually trip circuit breaker"""
        self._trip('MANUAL', reason)
    
    def force_reset(self):
        """Manually reset circuit breaker"""
        self._reset()

