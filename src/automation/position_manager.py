"""
Automated Position Manager Module
Track and monitor open positions in real-time
"""

from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from loguru import logger

from src.database.models import Trade, Position


class AutomatedPositionManager:
    """
    Automatically manage open positions
    Track P&L, Greeks, and position health
    """
    
    def __init__(self, db_session: Session, alpaca_client):
        """
        Initialize automated position manager
        
        Args:
            db_session: Database session
            alpaca_client: Alpaca client
        """
        self.db = db_session
        self.alpaca = alpaca_client
        logger.info("Automated Position Manager initialized")
    
    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions from database
        
        Returns:
            List of open positions
        """
        try:
            with self.db.get_session() as session:
                open_trades = session.query(Trade).filter(
                    Trade.status == 'open'
                ).all()
            
            positions = []
            
            for trade in open_trades:
                position = {
                    'position_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'strategy_type': trade.strategy,
                    'entry_date': trade.timestamp_enter,
                    'expiration': trade.params.get('expiration') if isinstance(trade.params, dict) else None,
                    'strikes': trade.params.get('strikes') if isinstance(trade.params, dict) else [],
                    'max_profit': trade.risk.get('max_profit') if isinstance(trade.risk, dict) else 0,
                    'max_loss': trade.risk.get('max_loss') if isinstance(trade.risk, dict) else 0,
                    'entry_credit': trade.execution.get('entry_credit') if isinstance(trade.execution, dict) else 0,
                    'current_pnl': trade.pnl,
                    'days_held': (datetime.utcnow() - trade.timestamp_enter).days
                }
                
                positions.append(position)
            
            logger.info(f"Retrieved {len(positions)} open positions")
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []
    
    def update_position_values(
        self,
        positions: List[Dict]
    ) -> List[Dict]:
        """
        Update current values for all positions
        
        Args:
            positions: List of positions
            
        Returns:
            Updated positions
        """
        for position in positions:
            try:
                # Get current market data
                symbol = position['symbol']
                stock_data = self.alpaca.get_stock_data(symbol)
                
                if not stock_data:
                    continue
                
                current_price = stock_data['price']
                
                # Estimate current P&L (simplified)
                # In real implementation, would get current option prices
                days_held = position['days_held']
                max_profit = position['max_profit']
                
                # Simulate time decay benefit
                time_decay_factor = min(1.0, days_held / 35.0)  # Assume 35 DTE
                estimated_pnl = max_profit * time_decay_factor * 0.5  # Simplified
                
                position['current_price'] = current_price
                position['current_pnl'] = estimated_pnl
                position['current_pnl_pct'] = (estimated_pnl / abs(position['max_loss'])) * 100 if position['max_loss'] else 0
                
            except Exception as e:
                logger.error(f"Error updating position {position['position_id']}: {e}")
                continue
        
        return positions
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get portfolio summary
        
        Returns:
            Portfolio summary dictionary
        """
        try:
            positions = self.get_open_positions()
            positions = self.update_position_values(positions)
            
            total_positions = len(positions)
            total_risk = sum(abs(p['max_loss']) for p in positions)
            total_potential_profit = sum(p['max_profit'] for p in positions)
            current_pnl = sum(p.get('current_pnl', 0) for p in positions)
            
            # Get account info
            account = self.alpaca.get_account()
            
            summary = {
                'timestamp': datetime.utcnow(),
                'account_equity': float(account.get('equity', 0)),
                'cash': float(account.get('cash', 0)),
                'buying_power': float(account.get('buying_power', 0)),
                'total_positions': total_positions,
                'total_risk': total_risk,
                'total_potential_profit': total_potential_profit,
                'current_pnl': current_pnl,
                'positions': positions
            }
            
            logger.info(f"Portfolio: {total_positions} positions, ${current_pnl:,.2f} P&L")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def check_position_health(
        self,
        position: Dict
    ) -> Dict:
        """
        Check health of a position
        
        Args:
            position: Position dictionary
            
        Returns:
            Health status
        """
        health = {
            'position_id': position['position_id'],
            'status': 'HEALTHY',
            'warnings': [],
            'actions': []
        }
        
        # Check days to expiration
        if position.get('expiration'):
            days_to_expiry = (position['expiration'] - datetime.utcnow()).days
            
            if days_to_expiry <= 0:
                health['status'] = 'EXPIRED'
                health['actions'].append('CLOSE_IMMEDIATELY')
            elif days_to_expiry <= 1:
                health['status'] = 'CRITICAL'
                health['warnings'].append('Expiring soon')
                health['actions'].append('CONSIDER_CLOSING')
        
        # Check P&L
        current_pnl = position.get('current_pnl', 0)
        max_profit = position['max_profit']
        max_loss = position['max_loss']
        
        # Check if at profit target
        if current_pnl >= max_profit * 0.50:
            health['warnings'].append('At profit target')
            health['actions'].append('TAKE_PROFIT')
        
        # Check if approaching max loss
        if current_pnl <= -abs(max_loss) * 0.75:
            health['status'] = 'AT_RISK'
            health['warnings'].append('Approaching max loss')
            health['actions'].append('CONSIDER_CLOSING')
        
        # Check if at max loss
        if current_pnl <= -abs(max_loss):
            health['status'] = 'CRITICAL'
            health['warnings'].append('At or beyond max loss')
            health['actions'].append('CLOSE_IMMEDIATELY')
        
        return health
    
    def get_positions_needing_action(self) -> List[Dict]:
        """
        Get positions that need action
        
        Returns:
            List of positions with required actions
        """
        positions = self.get_open_positions()
        positions = self.update_position_values(positions)
        
        needs_action = []
        
        for position in positions:
            health = self.check_position_health(position)
            
            if health['actions']:
                needs_action.append({
                    'position': position,
                    'health': health
                })
        
        logger.info(f"{len(needs_action)} positions need action")
        
        return needs_action

