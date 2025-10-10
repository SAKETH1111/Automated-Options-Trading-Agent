"""
Portfolio Risk Manager Module
Portfolio-level risk management and diversification
"""

from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from loguru import logger

from src.database.models import Trade


class PortfolioRiskManager:
    """
    Manage portfolio-level risk
    Enforce position limits, diversification, and portfolio stop-losses
    """
    
    def __init__(
        self,
        db_session: Session,
        total_capital: float = 10000.0
    ):
        """
        Initialize portfolio risk manager
        
        Args:
            db_session: Database session
            total_capital: Total trading capital
        """
        self.db = db_session
        self.total_capital = total_capital
        
        # Risk limits
        self.max_positions = 10
        self.max_risk_per_position = 0.02  # 2% per position
        self.max_total_risk = 0.10  # 10% total portfolio risk
        self.max_symbol_concentration = 0.30  # 30% per symbol
        self.max_strategy_concentration = 0.40  # 40% per strategy
        
        # Portfolio stop-loss
        self.daily_loss_limit = 0.03  # 3% daily loss limit
        self.max_drawdown_limit = 0.15  # 15% max drawdown
        
        logger.info(f"Portfolio Risk Manager initialized with ${total_capital:,.2f}")
    
    def check_can_open_position(
        self,
        proposed_trade: Dict
    ) -> Dict:
        """
        Check if we can open a new position
        
        Args:
            proposed_trade: Proposed trade details
            
        Returns:
            Dictionary with approval status and reasons
        """
        result = {
            'approved': True,
            'reasons': [],
            'warnings': []
        }
        
        try:
            # Get current open positions
            open_positions = self._get_open_positions()
            
            # Check 1: Max positions limit
            if len(open_positions) >= self.max_positions:
                result['approved'] = False
                result['reasons'].append(f"At max positions limit ({self.max_positions})")
                return result
            
            # Check 2: Position size limit
            position_risk = abs(proposed_trade.get('max_loss', 0))
            max_position_risk = self.total_capital * self.max_risk_per_position
            
            if position_risk > max_position_risk:
                result['approved'] = False
                result['reasons'].append(
                    f"Position risk ${position_risk:.2f} exceeds limit ${max_position_risk:.2f}"
                )
                return result
            
            # Check 3: Total portfolio risk
            current_total_risk = sum(abs(p.get('max_loss', 0)) for p in open_positions)
            new_total_risk = current_total_risk + position_risk
            max_total_risk = self.total_capital * self.max_total_risk
            
            if new_total_risk > max_total_risk:
                result['approved'] = False
                result['reasons'].append(
                    f"Total risk ${new_total_risk:.2f} would exceed limit ${max_total_risk:.2f}"
                )
                return result
            
            # Check 4: Symbol concentration
            symbol = proposed_trade.get('symbol')
            symbol_risk = sum(
                abs(p.get('max_loss', 0))
                for p in open_positions
                if p.get('symbol') == symbol
            )
            symbol_risk += position_risk
            max_symbol_risk = self.total_capital * self.max_symbol_concentration
            
            if symbol_risk > max_symbol_risk:
                result['approved'] = False
                result['reasons'].append(
                    f"Symbol concentration ${symbol_risk:.2f} exceeds limit ${max_symbol_risk:.2f}"
                )
                return result
            
            # Check 5: Strategy concentration
            strategy = proposed_trade.get('strategy_type')
            strategy_risk = sum(
                abs(p.get('max_loss', 0))
                for p in open_positions
                if p.get('strategy') == strategy
            )
            strategy_risk += position_risk
            max_strategy_risk = self.total_capital * self.max_strategy_concentration
            
            if strategy_risk > max_strategy_risk:
                result['warnings'].append(
                    f"Strategy concentration high: ${strategy_risk:.2f}"
                )
            
            # Check 6: Daily loss limit
            daily_loss = self._get_daily_loss()
            if daily_loss <= -self.total_capital * self.daily_loss_limit:
                result['approved'] = False
                result['reasons'].append(
                    f"Daily loss limit reached: ${daily_loss:.2f}"
                )
                return result
            
            # Check 7: Max drawdown
            current_drawdown_pct = self._get_current_drawdown_pct()
            if current_drawdown_pct >= self.max_drawdown_limit:
                result['approved'] = False
                result['reasons'].append(
                    f"Max drawdown limit reached: {current_drawdown_pct:.1%}"
                )
                return result
            
            # All checks passed
            if result['warnings']:
                logger.warning(f"Position approved with warnings: {result['warnings']}")
            else:
                logger.info("Position approved - all risk checks passed")
            
        except Exception as e:
            logger.error(f"Error checking position approval: {e}")
            result['approved'] = False
            result['reasons'].append(f"Error: {str(e)}")
        
        return result
    
    def get_portfolio_risk_metrics(self) -> Dict:
        """
        Get current portfolio risk metrics
        
        Returns:
            Dictionary with risk metrics
        """
        try:
            open_positions = self._get_open_positions()
            
            # Calculate total risk
            total_risk = sum(abs(p.get('max_loss', 0)) for p in open_positions)
            total_risk_pct = (total_risk / self.total_capital) * 100
            
            # Calculate by symbol
            by_symbol = {}
            for position in open_positions:
                symbol = position.get('symbol', 'UNKNOWN')
                if symbol not in by_symbol:
                    by_symbol[symbol] = {'count': 0, 'risk': 0}
                by_symbol[symbol]['count'] += 1
                by_symbol[symbol]['risk'] += abs(position.get('max_loss', 0))
            
            # Calculate by strategy
            by_strategy = {}
            for position in open_positions:
                strategy = position.get('strategy', 'UNKNOWN')
                if strategy not in by_strategy:
                    by_strategy[strategy] = {'count': 0, 'risk': 0}
                by_strategy[strategy]['count'] += 1
                by_strategy[strategy]['risk'] += abs(position.get('max_loss', 0))
            
            # Get daily loss
            daily_loss = self._get_daily_loss()
            daily_loss_pct = (daily_loss / self.total_capital) * 100
            
            # Get drawdown
            drawdown_pct = self._get_current_drawdown_pct()
            
            return {
                'timestamp': datetime.utcnow(),
                'total_capital': self.total_capital,
                'total_positions': len(open_positions),
                'total_risk': total_risk,
                'total_risk_pct': total_risk_pct,
                'available_risk': self.total_capital * self.max_total_risk - total_risk,
                'by_symbol': by_symbol,
                'by_strategy': by_strategy,
                'daily_loss': daily_loss,
                'daily_loss_pct': daily_loss_pct,
                'current_drawdown_pct': drawdown_pct,
                'limits': {
                    'max_positions': self.max_positions,
                    'max_risk_per_position': self.max_risk_per_position,
                    'max_total_risk': self.max_total_risk,
                    'daily_loss_limit': self.daily_loss_limit,
                    'max_drawdown_limit': self.max_drawdown_limit
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio risk metrics: {e}")
            return {}
    
    def _get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            trades = self.db.query(Trade).filter(
                Trade.status == 'open'
            ).all()
            
            positions = []
            for trade in trades:
                positions.append({
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'strategy': trade.strategy,
                    'max_loss': trade.risk.get('max_loss') if isinstance(trade.risk, dict) else 0,
                    'max_profit': trade.risk.get('max_profit') if isinstance(trade.risk, dict) else 0,
                    'current_pnl': trade.pnl
                })
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []
    
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
    
    def _get_current_drawdown_pct(self) -> float:
        """Get current drawdown percentage"""
        try:
            # Get all closed trades
            all_trades = self.db.query(Trade).filter(
                Trade.status == 'closed'
            ).order_by(Trade.timestamp_exit.asc()).all()
            
            if not all_trades:
                return 0.0
            
            # Calculate equity curve
            equity = self.total_capital
            peak = equity
            max_dd_pct = 0
            
            for trade in all_trades:
                equity += trade.pnl
                
                if equity > peak:
                    peak = equity
                
                dd_pct = ((peak - equity) / peak) * 100 if peak > 0 else 0
                
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
            
            return max_dd_pct
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return 0.0

