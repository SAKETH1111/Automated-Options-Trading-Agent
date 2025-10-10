"""Position monitoring and exit management"""

from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger
from sqlalchemy.orm import Session

from src.database.models import Trade, Position
from src.database.session import get_db
from src.market_data.collector import MarketDataCollector
from src.strategies import BullPutSpreadStrategy, CashSecuredPutStrategy, IronCondorStrategy


class PositionMonitor:
    """Monitor open positions and trigger exits"""
    
    def __init__(self, market_data: Optional[MarketDataCollector] = None):
        self.market_data = market_data or MarketDataCollector()
        self.db = get_db()
        
        # Strategy instances for exit logic
        self.strategies = {}
        
        logger.info("Position Monitor initialized")
    
    def load_strategy(self, strategy_name: str, config: Dict):
        """Load strategy instance"""
        if strategy_name == "Bull Put Spread":
            self.strategies[strategy_name] = BullPutSpreadStrategy(config)
        elif strategy_name == "Cash Secured Put":
            self.strategies[strategy_name] = CashSecuredPutStrategy(config)
        elif strategy_name == "Iron Condor":
            self.strategies[strategy_name] = IronCondorStrategy(config)
    
    def monitor_positions(self) -> List[Dict]:
        """
        Monitor all open positions and check exit conditions
        
        Returns:
            List of exit signals
        """
        try:
            exit_signals = []
            
            with self.db.get_session() as session:
                open_trades = session.query(Trade).filter_by(status="open").all()
                
                logger.info(f"Monitoring {len(open_trades)} open positions")
                
                for trade in open_trades:
                    # Get current positions
                    positions = session.query(Position).filter_by(
                        trade_id=trade.trade_id,
                        status="open"
                    ).all()
                    
                    if not positions:
                        continue
                    
                    # Update position data
                    self._update_position_data(positions)
                    
                    # Calculate current P&L
                    current_pnl = sum(p.unrealized_pnl for p in positions)
                    
                    entry_credit = trade.execution.get("fill_credit", 0) * trade.risk.get("size", 1)
                    current_pnl_pct = (current_pnl / (entry_credit * 100)) * 100 if entry_credit > 0 else 0
                    
                    # Update trade P&L
                    trade.pnl = current_pnl
                    trade.pnl_pct = current_pnl_pct
                    trade.days_held = (datetime.now() - trade.timestamp_enter).days
                    
                    # Check exit conditions
                    strategy_name = trade.strategy
                    if strategy_name in self.strategies:
                        strategy = self.strategies[strategy_name]
                        
                        # Check if should exit
                        exit_signal = strategy.should_exit(
                            trade.to_dict(),
                            [p.to_dict() for p in positions],
                            current_pnl,
                            current_pnl_pct
                        )
                        
                        if exit_signal:
                            exit_signals.append({
                                "trade_id": trade.trade_id,
                                "symbol": trade.symbol,
                                "strategy": strategy_name,
                                "action": exit_signal["action"],
                                "reason": exit_signal["reason"],
                                "pnl": current_pnl,
                                "pnl_pct": current_pnl_pct,
                            })
                        
                        # Check if should roll
                        # Get options chain for rolling
                        options_chain = self.market_data.get_options_chain_enriched(
                            trade.symbol, target_dte=30
                        )
                        
                        roll_signal = strategy.should_roll(
                            trade.to_dict(),
                            [p.to_dict() for p in positions],
                            options_chain
                        )
                        
                        if roll_signal:
                            exit_signals.append({
                                "trade_id": trade.trade_id,
                                "symbol": trade.symbol,
                                "strategy": strategy_name,
                                "action": "roll",
                                "reason": roll_signal["reason"],
                                "pnl": current_pnl,
                                "pnl_pct": current_pnl_pct,
                            })
                
                session.commit()
            
            if exit_signals:
                logger.info(f"Generated {len(exit_signals)} exit signals")
            
            return exit_signals
        
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
            return []
    
    def _update_position_data(self, positions: List[Position]):
        """Update current position data from market"""
        try:
            for position in positions:
                # Get current option data
                # In production, fetch real-time data
                # For now, use simplified update
                
                # Calculate unrealized P&L
                if position.side == "short":
                    unrealized_pnl = (position.entry_price - position.current_price) * position.quantity * 100
                else:
                    unrealized_pnl = (position.current_price - position.entry_price) * position.quantity * 100
                
                position.unrealized_pnl = unrealized_pnl
                position.updated_at = datetime.now()
        
        except Exception as e:
            logger.error(f"Error updating position data: {e}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary with all positions"""
        try:
            with self.db.get_session() as session:
                open_trades = session.query(Trade).filter_by(status="open").all()
                
                total_pnl = sum(trade.pnl for trade in open_trades)
                total_risk = sum(trade.risk.get("max_loss", 0) for trade in open_trades)
                
                positions_by_symbol = {}
                for trade in open_trades:
                    symbol = trade.symbol
                    if symbol not in positions_by_symbol:
                        positions_by_symbol[symbol] = []
                    
                    positions_by_symbol[symbol].append({
                        "trade_id": trade.trade_id,
                        "strategy": trade.strategy,
                        "pnl": trade.pnl,
                        "pnl_pct": trade.pnl_pct,
                        "days_held": trade.days_held,
                        "max_loss": trade.risk.get("max_loss", 0),
                    })
                
                return {
                    "total_positions": len(open_trades),
                    "total_pnl": round(total_pnl, 2),
                    "total_risk": round(total_risk, 2),
                    "positions_by_symbol": positions_by_symbol,
                    "timestamp": datetime.now().isoformat(),
                }
        
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}


# Helper method to convert SQLAlchemy model to dict
def _model_to_dict(model) -> Dict:
    """Convert SQLAlchemy model to dictionary"""
    result = {}
    for column in model.__table__.columns:
        value = getattr(model, column.name)
        if isinstance(value, datetime):
            value = value.isoformat()
        result[column.name] = value
    return result


# Add to_dict methods to models
Trade.to_dict = lambda self: _model_to_dict(self)
Position.to_dict = lambda self: _model_to_dict(self)


