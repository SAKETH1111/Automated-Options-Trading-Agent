"""Trade execution engine"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger
from sqlalchemy.orm import Session

from src.brokers.alpaca_client import AlpacaClient
from src.database.models import Trade, Position
from src.database.session import get_db
from src.risk.manager import RiskManager
from src.risk.position_sizer import PositionSizer


class TradeExecutor:
    """Execute trades and manage orders"""
    
    def __init__(
        self,
        alpaca_client: Optional[AlpacaClient] = None,
        risk_manager: Optional[RiskManager] = None,
        config: Optional[Dict] = None
    ):
        self.alpaca = alpaca_client or AlpacaClient()
        self.risk_manager = risk_manager or RiskManager(config)
        self.config = config or {}
        self.position_sizer = PositionSizer(self.config)
        self.db = get_db()
        
        logger.info("Trade Executor initialized")
    
    def execute_signal(self, signal: Dict) -> Optional[str]:
        """
        Execute a trading signal
        
        Args:
            signal: Trading signal dictionary
        
        Returns:
            Trade ID if successful, None otherwise
        """
        try:
            # Get account info
            account = self.alpaca.get_account()
            account_balance = account["equity"]
            
            # Get current positions
            current_positions = self._get_current_positions()
            
            # Check risk constraints
            risk_check = self.risk_manager.can_open_trade(
                signal, account_balance, current_positions
            )
            
            if not risk_check["allowed"]:
                logger.warning(f"Trade rejected: {risk_check['reason']}")
                return None
            
            # Calculate position size
            position_size = self.position_sizer.calculate_size(
                signal, account_balance
            )
            
            # Validate size
            position_size = self.position_sizer.validate_size(
                position_size,
                signal["max_loss"],
                account_balance,
                self.risk_manager.max_position_size_pct
            )
            
            logger.info(f"Executing signal for {signal['symbol']} with {position_size} contracts")
            
            # Execute legs
            execution_result = self._execute_legs(signal["legs"], position_size)
            
            if not execution_result["success"]:
                logger.error("Failed to execute trade")
                return None
            
            # Create trade record
            trade_id = self._create_trade_record(
                signal, position_size, execution_result, account_balance
            )
            
            logger.info(f"Trade {trade_id} executed successfully")
            
            return trade_id
        
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return None
    
    def _execute_legs(self, legs: List[Dict], contracts: int) -> Dict:
        """Execute all legs of a spread"""
        try:
            executed_legs = []
            total_credit = 0.0
            
            for leg in legs:
                # Prepare order
                side = "sell" if leg["side"] == "short" else "buy"
                qty = contracts
                symbol = leg["option_symbol"]
                limit_price = leg["price"]
                
                # For spreads, adjust limit slightly for better fill
                if side == "sell":
                    limit_price *= 0.95  # Accept 95% of mid for sells
                else:
                    limit_price *= 1.05  # Pay 105% of mid for buys
                
                # Place limit order
                order = self.alpaca.place_limit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    limit_price=round(limit_price, 2),
                    time_in_force="day"
                )
                
                if not order:
                    logger.error(f"Failed to place order for {symbol}")
                    # Cancel previous orders if any leg fails
                    self._cancel_orders([leg["order_id"] for leg in executed_legs if "order_id" in leg])
                    return {"success": False, "reason": "order_placement_failed"}
                
                executed_legs.append({
                    "leg": leg,
                    "order_id": order["order_id"],
                    "limit_price": limit_price,
                })
                
                # Track credit/debit
                if side == "sell":
                    total_credit += limit_price * qty * 100
                else:
                    total_credit -= limit_price * qty * 100
            
            # Wait for fills (simplified - in production, use async monitoring)
            import time
            time.sleep(2)
            
            # Check order status
            filled_legs = []
            for executed_leg in executed_legs:
                order_status = self.alpaca.get_order(executed_leg["order_id"])
                
                if order_status and order_status["status"] == "filled":
                    filled_legs.append({
                        "leg": executed_leg["leg"],
                        "order_id": executed_leg["order_id"],
                        "fill_price": order_status["filled_avg_price"],
                        "filled_qty": order_status["filled_qty"],
                    })
                else:
                    logger.warning(f"Order {executed_leg['order_id']} not filled yet")
            
            # For now, require all legs to fill
            if len(filled_legs) != len(legs):
                logger.error("Not all legs filled")
                # Cancel unfilled orders
                for executed_leg in executed_legs:
                    order_status = self.alpaca.get_order(executed_leg["order_id"])
                    if order_status and order_status["status"] != "filled":
                        self.alpaca.cancel_order(executed_leg["order_id"])
                
                return {"success": False, "reason": "partial_fill"}
            
            # Calculate actual credit received
            actual_credit = 0.0
            for filled_leg in filled_legs:
                side = filled_leg["leg"]["side"]
                fill_price = filled_leg["fill_price"]
                qty = filled_leg["filled_qty"]
                
                if side == "short":
                    actual_credit += fill_price * qty * 100
                else:
                    actual_credit -= fill_price * qty * 100
            
            return {
                "success": True,
                "filled_legs": filled_legs,
                "net_credit": actual_credit,
            }
        
        except Exception as e:
            logger.error(f"Error executing legs: {e}")
            return {"success": False, "reason": str(e)}
    
    def _cancel_orders(self, order_ids: List[str]):
        """Cancel list of orders"""
        for order_id in order_ids:
            try:
                self.alpaca.cancel_order(order_id)
            except Exception as e:
                logger.error(f"Error cancelling order {order_id}: {e}")
    
    def _create_trade_record(
        self,
        signal: Dict,
        contracts: int,
        execution_result: Dict,
        account_balance: float
    ) -> str:
        """Create trade record in database"""
        try:
            trade_id = str(uuid.uuid4())
            
            # Calculate actual execution metrics
            net_credit = execution_result["net_credit"]
            max_loss = signal["max_loss"] * contracts
            
            # Prepare trade record
            trade = Trade(
                trade_id=trade_id,
                timestamp_enter=datetime.now(),
                symbol=signal["symbol"],
                strategy=signal["strategy_name"],
                params=signal["params"],
                market_snapshot=signal["market_snapshot"],
                execution={
                    "limit_credit": signal["expected_credit"],
                    "fill_credit": net_credit / (contracts * 100),  # Per contract
                    "slippage": (signal["expected_credit"] - net_credit / (contracts * 100)) / signal["expected_credit"] * 100 if signal["expected_credit"] > 0 else 0,
                    "contracts": contracts,
                },
                risk={
                    "size": contracts,
                    "risk_pct": (max_loss / account_balance) * 100,
                    "max_loss": max_loss,
                },
                status="open",
            )
            
            # Create position records
            with self.db.get_session() as session:
                session.add(trade)
                
                # Add positions
                for filled_leg in execution_result["filled_legs"]:
                    leg = filled_leg["leg"]
                    
                    position = Position(
                        position_id=str(uuid.uuid4()),
                        trade_id=trade_id,
                        symbol=signal["symbol"],
                        option_symbol=leg["option_symbol"],
                        option_type=leg["option_type"],
                        strike=leg["strike"],
                        expiration=datetime.fromisoformat(leg["expiration"]),
                        side=leg["side"],
                        quantity=contracts,
                        entry_price=filled_leg["fill_price"],
                        entry_time=datetime.now(),
                        entry_delta=leg["delta"],
                        current_price=filled_leg["fill_price"],
                        current_delta=leg["delta"],
                        status="open",
                    )
                    
                    session.add(position)
                
                session.commit()
            
            return trade_id
        
        except Exception as e:
            logger.error(f"Error creating trade record: {e}")
            raise
    
    def close_trade(self, trade_id: str, reason: str) -> bool:
        """Close an existing trade"""
        try:
            with self.db.get_session() as session:
                trade = session.query(Trade).filter_by(trade_id=trade_id).first()
                
                if not trade:
                    logger.error(f"Trade {trade_id} not found")
                    return False
                
                if trade.status != "open":
                    logger.warning(f"Trade {trade_id} is not open")
                    return False
                
                # Get positions
                positions = session.query(Position).filter_by(
                    trade_id=trade_id, status="open"
                ).all()
                
                if not positions:
                    logger.error(f"No open positions for trade {trade_id}")
                    return False
                
                # Close each position
                total_pnl = 0.0
                
                for position in positions:
                    # Get current price
                    # In production, use actual market data
                    current_price = position.current_price  # Placeholder
                    
                    # Calculate P&L
                    if position.side == "short":
                        pnl = (position.entry_price - current_price) * position.quantity * 100
                    else:
                        pnl = (current_price - position.entry_price) * position.quantity * 100
                    
                    # Place closing order
                    close_side = "buy" if position.side == "short" else "sell"
                    
                    order = self.alpaca.place_market_order(
                        symbol=position.option_symbol,
                        qty=position.quantity,
                        side=close_side
                    )
                    
                    if order:
                        # Update position
                        position.exit_price = current_price
                        position.exit_time = datetime.now()
                        position.realized_pnl = pnl
                        position.status = "closed"
                        
                        total_pnl += pnl
                
                # Update trade
                trade.timestamp_exit = datetime.now()
                trade.pnl = total_pnl
                trade.pnl_pct = (total_pnl / abs(trade.risk["max_loss"])) * 100 if trade.risk["max_loss"] != 0 else 0
                trade.days_held = (datetime.now() - trade.timestamp_enter).days
                trade.exit_reason = reason
                trade.status = "closed"
                
                session.commit()
                
                logger.info(f"Trade {trade_id} closed with P&L: ${total_pnl:.2f} ({trade.pnl_pct:.1f}%)")
                
                return True
        
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
            return False
    
    def _get_current_positions(self) -> List[Dict]:
        """Get current open positions with risk metrics"""
        try:
            with self.db.get_session() as session:
                open_trades = session.query(Trade).filter_by(status="open").all()
                
                positions = []
                for trade in open_trades:
                    positions.append({
                        "trade_id": trade.trade_id,
                        "symbol": trade.symbol,
                        "strategy": trade.strategy,
                        "max_loss": trade.risk.get("max_loss", 0),
                        "unrealized_pnl": trade.pnl,
                        "delta_exposure": trade.market_snapshot.get("delta_exposure", 0),
                        "theta_exposure": trade.market_snapshot.get("theta_exposure", 0),
                        "vega_exposure": trade.market_snapshot.get("vega_exposure", 0),
                    })
                
                return positions
        
        except Exception as e:
            logger.error(f"Error getting current positions: {e}")
            return []


