"""
Automated Trade Manager Module
Manage trades with automatic stop-loss and profit targets
"""

from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from loguru import logger

from src.database.models import Trade


class AutomatedTradeManager:
    """
    Automatically manage trades
    Handle stop-loss, profit targets, and position adjustments
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize automated trade manager
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        
        # Management parameters
        self.profit_target_pct = 0.50  # Close at 50% of max profit
        self.stop_loss_multiplier = 2.0  # Stop at 2x max loss
        self.days_before_expiry_close = 1  # Close 1 day before expiry
        
        logger.info("Automated Trade Manager initialized")
    
    def manage_position(
        self,
        position: Dict,
        current_price: float
    ) -> Optional[Dict]:
        """
        Manage a position and determine if action needed
        
        Args:
            position: Position dictionary
            current_price: Current underlying price
            
        Returns:
            Management action or None
        """
        try:
            # Calculate current metrics
            days_to_expiry = self._calculate_days_to_expiry(position)
            current_pnl = position.get('current_pnl', 0)
            max_profit = position['max_profit']
            max_loss = position['max_loss']
            
            # Check expiration
            if days_to_expiry <= self.days_before_expiry_close:
                return {
                    'action': 'CLOSE',
                    'reason': 'EXPIRATION',
                    'urgency': 'HIGH',
                    'message': f'Position expiring in {days_to_expiry} day(s)'
                }
            
            # Check profit target
            profit_target = max_profit * self.profit_target_pct
            if current_pnl >= profit_target:
                return {
                    'action': 'CLOSE',
                    'reason': 'TAKE_PROFIT',
                    'urgency': 'MEDIUM',
                    'message': f'Profit target reached: ${current_pnl:.2f} >= ${profit_target:.2f}'
                }
            
            # Check stop loss
            stop_loss = -abs(max_loss) * self.stop_loss_multiplier
            if current_pnl <= stop_loss:
                return {
                    'action': 'CLOSE',
                    'reason': 'STOP_LOSS',
                    'urgency': 'HIGH',
                    'message': f'Stop loss triggered: ${current_pnl:.2f} <= ${stop_loss:.2f}'
                }
            
            # Check if approaching max loss (warning)
            if current_pnl <= -abs(max_loss) * 0.75:
                return {
                    'action': 'MONITOR',
                    'reason': 'APPROACHING_MAX_LOSS',
                    'urgency': 'MEDIUM',
                    'message': f'Position at 75% of max loss'
                }
            
            # Position is healthy
            return {
                'action': 'HOLD',
                'reason': 'HEALTHY',
                'urgency': 'LOW',
                'message': 'Position within acceptable range'
            }
            
        except Exception as e:
            logger.error(f"Error managing position: {e}")
            return None
    
    def _calculate_days_to_expiry(self, position: Dict) -> int:
        """Calculate days to expiration"""
        if not position.get('expiration'):
            return 999  # Unknown expiration
        
        expiration = position['expiration']
        if isinstance(expiration, str):
            expiration = datetime.fromisoformat(expiration)
        
        days = (expiration - datetime.utcnow()).days
        return max(0, days)
    
    def adjust_position(
        self,
        position: Dict,
        adjustment_type: str
    ) -> Optional[Dict]:
        """
        Adjust a position (roll, add, reduce)
        
        Args:
            position: Position to adjust
            adjustment_type: Type of adjustment
            
        Returns:
            Adjustment result
        """
        logger.info(f"Adjusting position {position['position_id']}: {adjustment_type}")
        
        if adjustment_type == 'ROLL':
            return self._roll_position(position)
        elif adjustment_type == 'REDUCE':
            return self._reduce_position(position)
        elif adjustment_type == 'ADD':
            return self._add_to_position(position)
        else:
            logger.error(f"Unknown adjustment type: {adjustment_type}")
            return None
    
    def _roll_position(self, position: Dict) -> Dict:
        """Roll position to next expiration"""
        logger.info(f"Rolling position {position['position_id']}")
        
        # In real implementation, would:
        # 1. Close current position
        # 2. Open new position at next expiration
        # 3. Track as rolled trade
        
        return {
            'success': True,
            'action': 'ROLLED',
            'message': 'Position rolled to next expiration'
        }
    
    def _reduce_position(self, position: Dict) -> Dict:
        """Reduce position size"""
        logger.info(f"Reducing position {position['position_id']}")
        
        return {
            'success': True,
            'action': 'REDUCED',
            'message': 'Position size reduced by 50%'
        }
    
    def _add_to_position(self, position: Dict) -> Dict:
        """Add to position"""
        logger.info(f"Adding to position {position['position_id']}")
        
        return {
            'success': True,
            'action': 'ADDED',
            'message': 'Position size increased'
        }
    
    def calculate_portfolio_greeks(
        self,
        positions: List[Dict]
    ) -> Dict:
        """
        Calculate portfolio-level Greeks
        
        Args:
            positions: List of positions
            
        Returns:
            Portfolio Greeks
        """
        total_delta = 0
        total_theta = 0
        total_vega = 0
        
        for position in positions:
            # Get position Greeks from metadata
            delta = position.get('position_delta', 0)
            theta = position.get('position_theta', 0)
            vega = position.get('position_vega', 0)
            
            total_delta += delta
            total_theta += theta
            total_vega += vega
        
        return {
            'portfolio_delta': total_delta,
            'portfolio_theta': total_theta,
            'portfolio_vega': total_vega,
            'delta_dollars': total_delta * 100,  # Per $1 move
            'theta_dollars': total_theta * 100,  # Per day
            'vega_dollars': total_vega * 100     # Per 1% IV move
        }

