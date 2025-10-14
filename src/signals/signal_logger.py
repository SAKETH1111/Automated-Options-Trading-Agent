"""
Signal Logger
Logs all signals to database for tracking and analysis
"""

from datetime import datetime
from typing import Dict, List
from loguru import logger

from src.database.session import get_db
from src.database.models import OptionsOpportunity


class SignalLogger:
    """Log signals to database for tracking"""
    
    def __init__(self):
        self.db = get_db()
        logger.info("SignalLogger initialized")
    
    def log_signal(self, signal: Dict, executed: bool = False) -> str:
        """
        Log a signal to the database
        
        Args:
            signal: Signal dictionary
            executed: Whether the signal was executed as a trade
            
        Returns:
            Opportunity ID
        """
        try:
            with self.db.get_session() as session:
                opportunity = OptionsOpportunity(
                    symbol=signal['symbol'],
                    timestamp=datetime.now(),
                    strategy_type=signal.get('strategy_name', 'unknown'),
                    opportunity_score=signal.get('signal_quality', 0),
                    confidence=signal.get('probability_of_profit', 0),
                    strikes=[leg['strike'] for leg in signal.get('legs', [])],
                    expiration=datetime.fromisoformat(signal['params']['expiration']) if signal.get('params', {}).get('expiration') else None,
                    dte=signal.get('params', {}).get('dte', 0),
                    entry_credit=signal.get('expected_credit', 0),
                    max_profit=signal.get('max_profit', 0),
                    max_loss=signal.get('max_loss', 0),
                    position_delta=signal.get('delta_exposure', 0),
                    position_theta=signal.get('theta_exposure', 0),
                    position_vega=signal.get('vega_exposure', 0),
                    pop=signal.get('probability_of_profit', 0),
                    risk_reward_ratio=signal.get('risk_reward_ratio', 0),
                    iv_rank=signal.get('market_snapshot', {}).get('iv_rank', 0),
                    underlying_price=signal.get('market_snapshot', {}).get('price', 0),
                    reasons=[signal.get('reason', ''), signal.get('notes', '')],
                    status='executed' if executed else 'identified',
                )
                
                session.add(opportunity)
                session.commit()
                
                logger.info(f"Signal logged: {signal['symbol']} - {signal.get('strategy_name')} (Quality: {signal.get('signal_quality', 0):.0f}/100)")
                
                return opportunity.opportunity_id
        
        except Exception as e:
            logger.error(f"Error logging signal: {e}")
            return None
    
    def log_signals_batch(self, signals: List[Dict]) -> int:
        """
        Log multiple signals at once
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            Number of signals logged
        """
        count = 0
        for signal in signals:
            if self.log_signal(signal):
                count += 1
        
        return count
    
    def get_recent_signals(self, limit: int = 20) -> List[OptionsOpportunity]:
        """Get recent signals from database"""
        try:
            with self.db.get_session() as session:
                signals = session.query(OptionsOpportunity)\
                    .order_by(OptionsOpportunity.timestamp.desc())\
                    .limit(limit)\
                    .all()
                
                return signals
        
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return []
    
    def get_signal_stats(self) -> Dict:
        """Get signal generation statistics"""
        try:
            with self.db.get_session() as session:
                from sqlalchemy import func
                
                # Total signals
                total = session.query(func.count(OptionsOpportunity.opportunity_id)).scalar()
                
                # Executed signals
                executed = session.query(func.count(OptionsOpportunity.opportunity_id))\
                    .filter(OptionsOpportunity.status == 'executed')\
                    .scalar()
                
                # By symbol
                by_symbol = session.query(
                    OptionsOpportunity.symbol,
                    func.count(OptionsOpportunity.symbol).label('count')
                ).group_by(OptionsOpportunity.symbol).all()
                
                return {
                    'total_signals': total or 0,
                    'executed_signals': executed or 0,
                    'execution_rate': (executed / total * 100) if total > 0 else 0,
                    'by_symbol': {symbol: count for symbol, count in by_symbol}
                }
        
        except Exception as e:
            logger.error(f"Error getting signal stats: {e}")
            return {}

