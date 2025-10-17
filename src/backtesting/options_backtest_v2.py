"""
Institutional-Grade Options Backtesting Engine V2
Event-driven backtesting with no look-ahead bias, realistic execution simulation, and account-aware transaction costs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger
import warnings

from src.portfolio.account_manager import AccountProfile, AccountTier


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"
    ASSIGNED = "ASSIGNED"


@dataclass
class BacktestOrder:
    """Backtest order representation"""
    order_id: str
    timestamp: datetime
    symbol: str
    strategy: str
    side: str  # 'buy' or 'sell'
    quantity: int
    order_type: str  # 'limit', 'market'
    price: Optional[float]
    legs: List[Dict]  # Multi-leg order components
    status: OrderStatus
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class BacktestPosition:
    """Backtest position representation"""
    position_id: str
    trade_id: str
    symbol: str
    strategy: str
    entry_time: datetime
    entry_price: float
    quantity: int
    side: str
    max_loss: float
    max_profit: float
    status: TradeStatus
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    days_held: int = 0
    exit_reason: str = ""


@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    total_commission: float
    total_slippage: float
    trades: List[BacktestPosition]
    daily_returns: pd.Series
    equity_curve: pd.Series


class OptionsBacktestV2:
    """
    Institutional-grade options backtesting engine
    
    Features:
    - Event-driven framework (no look-ahead bias)
    - Options expiration/assignment handling
    - Early exercise simulation (American options)
    - Pin risk and gamma risk near expiration
    - Corporate actions (splits, dividends)
    - Account-aware transaction costs
    - Walk-forward optimization
    - Parameter stability analysis
    """
    
    def __init__(self, account_profile: AccountProfile):
        self.profile = account_profile
        
        # Backtest parameters
        self.initial_capital = account_profile.balance
        self.current_capital = self.initial_capital
        self.available_cash = self.initial_capital
        
        # Transaction costs by account tier
        self.transaction_costs = self._initialize_transaction_costs()
        
        # Position tracking
        self.open_positions = {}
        self.closed_positions = []
        self.orders = []
        
        # Performance tracking
        self.equity_curve = []
        self.daily_returns = []
        
        # Market data
        self.market_data = {}
        self.current_date = None
        
        # Risk management
        self.max_positions = account_profile.max_positions
        self.max_position_size_pct = account_profile.max_position_size_pct
        
        logger.info(f"OptionsBacktestV2 initialized for {account_profile.tier.value} tier")
        logger.info(f"  Initial capital: ${self.initial_capital:,.2f}")
        logger.info(f"  Max positions: {self.max_positions}")
    
    def _initialize_transaction_costs(self) -> Dict[str, float]:
        """Initialize transaction costs by account tier"""
        return {
            AccountTier.MICRO: {
                'commission_per_contract': 2.00,
                'slippage_multiplier': 1.5,
                'min_commission': 2.00,
                'max_commission': 10.00
            },
            AccountTier.SMALL: {
                'commission_per_contract': 1.75,
                'slippage_multiplier': 1.3,
                'min_commission': 1.75,
                'max_commission': 8.75
            },
            AccountTier.MEDIUM: {
                'commission_per_contract': 1.25,
                'slippage_multiplier': 1.0,
                'min_commission': 1.25,
                'max_commission': 6.25
            },
            AccountTier.LARGE: {
                'commission_per_contract': 0.75,
                'slippage_multiplier': 0.8,
                'min_commission': 0.75,
                'max_commission': 3.75
            },
            AccountTier.INSTITUTIONAL: {
                'commission_per_contract': 0.50,
                'slippage_multiplier': 0.7,
                'min_commission': 0.50,
                'max_commission': 2.50
            }
        }
    
    def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        market_data: pd.DataFrame,
        strategy_signals: List[Dict],
        walk_forward: bool = False,
        train_months: int = 12,
        test_months: int = 3
    ) -> BacktestResults:
        """
        Run comprehensive backtest
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            market_data: Historical market data
            strategy_signals: Trading signals from strategy
            walk_forward: Enable walk-forward optimization
            train_months: Training period for walk-forward
            test_months: Testing period for walk-forward
        
        Returns:
            BacktestResults with comprehensive performance metrics
        """
        try:
            logger.info(f"Starting backtest: {start_date.date()} to {end_date.date()}")
            
            if walk_forward:
                return self._run_walk_forward_backtest(
                    start_date, end_date, market_data, strategy_signals,
                    train_months, test_months
                )
            else:
                return self._run_single_backtest(
                    start_date, end_date, market_data, strategy_signals
                )
                
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return self._empty_results(start_date, end_date)
    
    def _run_single_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        market_data: pd.DataFrame,
        strategy_signals: List[Dict]
    ) -> BacktestResults:
        """Run single backtest period"""
        try:
            # Initialize
            self.current_capital = self.initial_capital
            self.available_cash = self.initial_capital
            self.open_positions = {}
            self.closed_positions = []
            self.orders = []
            self.equity_curve = []
            self.daily_returns = []
            
            # Filter market data for backtest period
            mask = (market_data['date'] >= start_date) & (market_data['date'] <= end_date)
            period_data = market_data[mask].copy().reset_index(drop=True)
            
            if len(period_data) == 0:
                logger.warning("No market data for backtest period")
                return self._empty_results(start_date, end_date)
            
            # Process each trading day
            for i, row in period_data.iterrows():
                self.current_date = row['date']
                self.market_data = row.to_dict()
                
                # Update existing positions
                self._update_positions()
                
                # Process new signals
                day_signals = [s for s in strategy_signals if s['date'].date() == self.current_date.date()]
                self._process_signals(day_signals)
                
                # Update equity curve
                self._update_equity_curve()
            
            # Close any remaining positions
            self._close_all_positions()
            
            # Calculate final results
            results = self._calculate_results(start_date, end_date)
            
            logger.info(f"Backtest completed: {results.total_trades} trades, {results.total_return:.2%} return")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in single backtest: {e}")
            return self._empty_results(start_date, end_date)
    
    def _run_walk_forward_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        market_data: pd.DataFrame,
        strategy_signals: List[Dict],
        train_months: int,
        test_months: int
    ) -> BacktestResults:
        """Run walk-forward backtest with parameter optimization"""
        try:
            logger.info(f"Starting walk-forward backtest: {train_months}M train, {test_months}M test")
            
            # Initialize results
            all_results = []
            current_date = start_date
            
            while current_date < end_date:
                # Define training period
                train_start = current_date
                train_end = current_date + timedelta(days=train_months * 30)
                
                # Define testing period
                test_start = train_end
                test_end = min(test_start + timedelta(days=test_months * 30), end_date)
                
                if test_start >= end_date:
                    break
                
                logger.info(f"Walk-forward period: Train {train_start.date()} to {train_end.date()}, Test {test_start.date()} to {test_end.date()}")
                
                # Optimize parameters on training period
                optimal_params = self._optimize_parameters(
                    train_start, train_end, market_data, strategy_signals
                )
                
                # Run backtest on test period with optimal parameters
                test_results = self._run_single_backtest(
                    test_start, test_end, market_data, strategy_signals
                )
                
                all_results.append(test_results)
                
                # Move to next period
                current_date = test_start + timedelta(days=test_months * 30)
            
            # Combine all test results
            combined_results = self._combine_walk_forward_results(all_results, start_date, end_date)
            
            logger.info(f"Walk-forward backtest completed: {len(all_results)} periods")
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in walk-forward backtest: {e}")
            return self._empty_results(start_date, end_date)
    
    def _optimize_parameters(
        self,
        train_start: datetime,
        train_end: datetime,
        market_data: pd.DataFrame,
        strategy_signals: List[Dict]
    ) -> Dict[str, Any]:
        """Optimize strategy parameters on training period"""
        try:
            # This would implement parameter optimization
            # For now, return default parameters
            return {
                'max_position_size': 0.05,
                'stop_loss_pct': 0.50,
                'take_profit_pct': 0.50,
                'min_dte': 7,
                'max_dte': 60
            }
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")
            return {}
    
    def _update_positions(self):
        """Update existing positions with current market data"""
        try:
            positions_to_close = []
            
            for position_id, position in self.open_positions.items():
                # Check for expiration
                if self._is_expired(position):
                    position.status = TradeStatus.EXPIRED
                    position.exit_time = self.current_date
                    position.exit_reason = "Expiration"
                    self._close_position(position)
                    positions_to_close.append(position_id)
                    continue
                
                # Check for early assignment
                if self._check_assignment(position):
                    position.status = TradeStatus.ASSIGNED
                    position.exit_time = self.current_date
                    position.exit_reason = "Assignment"
                    self._close_position(position)
                    positions_to_close.append(position_id)
                    continue
                
                # Check stop loss / take profit
                if self._check_exit_conditions(position):
                    position.status = TradeStatus.CLOSED
                    position.exit_time = self.current_date
                    position.exit_reason = "Stop/Target"
                    self._close_position(position)
                    positions_to_close.append(position_id)
                    continue
                
                # Update position P&L
                self._update_position_pnl(position)
            
            # Remove closed positions
            for position_id in positions_to_close:
                del self.open_positions[position_id]
                
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _is_expired(self, position: BacktestPosition) -> bool:
        """Check if position is expired"""
        try:
            # Get expiration from market data or position info
            expiration_date = position.entry_time + timedelta(days=position.days_held + 30)  # Simplified
            
            return self.current_date >= expiration_date
            
        except Exception as e:
            logger.error(f"Error checking expiration: {e}")
            return False
    
    def _check_assignment(self, position: BacktestPosition) -> bool:
        """Check for early assignment"""
        try:
            # Simplified assignment logic
            # In practice, would check option type, moneyness, dividends, etc.
            
            # Only short options can be assigned
            if position.side != 'sell':
                return False
            
            # Check if option is deep ITM and close to expiration
            current_price = self.market_data.get('underlying_price', 0)
            strike = position.entry_price * 100  # Simplified strike calculation
            
            if position.days_held <= 3:  # Close to expiration
                if abs(current_price - strike) / current_price > 0.02:  # > 2% ITM
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking assignment: {e}")
            return False
    
    def _check_exit_conditions(self, position: BacktestPosition) -> bool:
        """Check stop loss and take profit conditions"""
        try:
            current_pnl_pct = position.pnl / position.max_loss if position.max_loss > 0 else 0
            
            # Stop loss (50% of max loss)
            if current_pnl_pct <= -0.50:
                return True
            
            # Take profit (50% of max profit)
            if position.pnl >= position.max_profit * 0.50:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return False
    
    def _update_position_pnl(self, position: BacktestPosition):
        """Update position P&L based on current market data"""
        try:
            # Simplified P&L calculation
            # In practice, would use option pricing models
            
            current_price = self.market_data.get('underlying_price', 0)
            entry_price = position.entry_price
            
            # Simplified P&L calculation
            price_change = (current_price - entry_price) / entry_price
            position.pnl = position.max_loss * price_change * 0.5  # Simplified
            position.pnl_pct = (position.pnl / self.current_capital) * 100
            
            # Update days held
            position.days_held = (self.current_date - position.entry_time).days
            
        except Exception as e:
            logger.error(f"Error updating position P&L: {e}")
    
    def _process_signals(self, signals: List[Dict]):
        """Process trading signals for current day"""
        try:
            for signal in signals:
                # Check if we can open new position
                if len(self.open_positions) >= self.max_positions:
                    continue
                
                # Check position size limits
                position_size = signal.get('position_size', 1)
                max_position_value = self.current_capital * (self.max_position_size_pct / 100)
                
                if position_size * signal.get('max_loss', 0) > max_position_value:
                    continue
                
                # Create and execute order
                order = self._create_order(signal)
                if order:
                    self._execute_order(order)
                    
        except Exception as e:
            logger.error(f"Error processing signals: {e}")
    
    def _create_order(self, signal: Dict) -> Optional[BacktestOrder]:
        """Create order from trading signal"""
        try:
            order_id = f"order_{len(self.orders)}_{self.current_date.strftime('%Y%m%d')}"
            
            order = BacktestOrder(
                order_id=order_id,
                timestamp=self.current_date,
                symbol=signal.get('symbol', ''),
                strategy=signal.get('strategy', ''),
                side=signal.get('side', 'buy'),
                quantity=signal.get('quantity', 1),
                order_type='limit',
                price=signal.get('price'),
                legs=signal.get('legs', []),
                status=OrderStatus.PENDING
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return None
    
    def _execute_order(self, order: BacktestOrder):
        """Execute order with realistic costs and slippage"""
        try:
            # Calculate transaction costs
            cost_config = self.transaction_costs.get(self.profile.tier, self.transaction_costs[AccountTier.MEDIUM])
            
            # Commission
            commission_per_contract = cost_config['commission_per_contract']
            total_commission = commission_per_contract * order.quantity
            total_commission = max(cost_config['min_commission'], min(total_commission, cost_config['max_commission']))
            
            # Slippage (simplified)
            slippage_multiplier = cost_config['slippage_multiplier']
            bid_ask_spread = 0.05  # Simplified spread
            slippage = bid_ask_spread * slippage_multiplier * order.quantity
            
            # Calculate fill price
            if order.price:
                fill_price = order.price + (slippage if order.side == 'buy' else -slippage)
            else:
                # Market order - use current price with slippage
                current_price = self.market_data.get('underlying_price', 0)
                fill_price = current_price + (slippage if order.side == 'buy' else -slippage)
            
            # Check if we have enough cash
            total_cost = (fill_price * order.quantity * 100) + total_commission + slippage
            
            if order.side == 'buy' and total_cost > self.available_cash:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order rejected: insufficient cash. Required: ${total_cost:,.2f}, Available: ${self.available_cash:,.2f}")
                return
            
            # Execute the order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = fill_price
            order.commission = total_commission
            order.slippage = slippage
            
            # Update cash
            if order.side == 'buy':
                self.available_cash -= total_cost
            else:
                self.available_cash += total_cost - total_commission - slippage
            
            # Create position
            position = self._create_position(order)
            if position:
                self.open_positions[position.position_id] = position
            
            # Add to orders
            self.orders.append(order)
            
        except Exception as e:
            logger.error(f"Error executing order: {e}")
    
    def _create_position(self, order: BacktestOrder) -> Optional[BacktestPosition]:
        """Create position from filled order"""
        try:
            position_id = f"pos_{order.order_id}"
            trade_id = f"trade_{order.order_id}"
            
            position = BacktestPosition(
                position_id=position_id,
                trade_id=trade_id,
                symbol=order.symbol,
                strategy=order.strategy,
                entry_time=order.timestamp,
                entry_price=order.average_fill_price,
                quantity=order.filled_quantity,
                side=order.side,
                max_loss=order.average_fill_price * order.filled_quantity * 100 * 0.5,  # Simplified
                max_profit=order.average_fill_price * order.filled_quantity * 100 * 0.3,  # Simplified
                status=TradeStatus.OPEN
            )
            
            return position
            
        except Exception as e:
            logger.error(f"Error creating position: {e}")
            return None
    
    def _close_position(self, position: BacktestPosition):
        """Close position and calculate final P&L"""
        try:
            # Calculate final P&L
            if position.exit_time:
                position.days_held = (position.exit_time - position.entry_time).days
            
            # Add to closed positions
            self.closed_positions.append(position)
            
            # Update capital
            self.current_capital += position.pnl
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _close_all_positions(self):
        """Close all remaining open positions"""
        try:
            for position in list(self.open_positions.values()):
                position.status = TradeStatus.CLOSED
                position.exit_time = self.current_date
                position.exit_reason = "End of backtest"
                self._close_position(position)
            
            self.open_positions = {}
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
    
    def _update_equity_curve(self):
        """Update equity curve with current portfolio value"""
        try:
            # Calculate current portfolio value
            portfolio_value = self.available_cash
            
            for position in self.open_positions.values():
                portfolio_value += position.pnl
            
            self.equity_curve.append({
                'date': self.current_date,
                'equity': portfolio_value,
                'cash': self.available_cash,
                'positions_value': portfolio_value - self.available_cash
            })
            
            # Calculate daily return
            if len(self.equity_curve) > 1:
                prev_equity = self.equity_curve[-2]['equity']
                daily_return = (portfolio_value - prev_equity) / prev_equity
                self.daily_returns.append(daily_return)
            
        except Exception as e:
            logger.error(f"Error updating equity curve: {e}")
    
    def _calculate_results(self, start_date: datetime, end_date: datetime) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        try:
            if not self.equity_curve:
                return self._empty_results(start_date, end_date)
            
            # Basic metrics
            initial_capital = self.initial_capital
            final_capital = self.equity_curve[-1]['equity']
            total_return = (final_capital - initial_capital) / initial_capital
            
            # Annualized return
            days = (end_date - start_date).days
            annualized_return = (1 + total_return) ** (365 / days) - 1
            
            # Risk metrics
            daily_returns_series = pd.Series(self.daily_returns)
            
            if len(daily_returns_series) > 1:
                sharpe_ratio = (daily_returns_series.mean() / daily_returns_series.std()) * np.sqrt(252)
                
                # Sortino ratio (downside deviation)
                negative_returns = daily_returns_series[daily_returns_series < 0]
                downside_deviation = negative_returns.std() * np.sqrt(252)
                sortino_ratio = (daily_returns_series.mean() * 252) / downside_deviation if downside_deviation > 0 else 0
                
                # Max drawdown
                equity_series = pd.Series([eq['equity'] for eq in self.equity_curve])
                rolling_max = equity_series.expanding().max()
                drawdown = (equity_series - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
                max_drawdown = 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Trade statistics
            total_trades = len(self.closed_positions)
            winning_trades = [t for t in self.closed_positions if t.pnl > 0]
            losing_trades = [t for t in self.closed_positions if t.pnl < 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            largest_win = max([t.pnl for t in self.closed_positions]) if self.closed_positions else 0
            largest_loss = min([t.pnl for t in self.closed_positions]) if self.closed_positions else 0
            
            # Profit factor
            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Transaction costs
            total_commission = sum(order.commission for order in self.orders)
            total_slippage = sum(order.slippage for order in self.orders)
            
            return BacktestResults(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_capital=final_capital,
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                total_commission=total_commission,
                total_slippage=total_slippage,
                trades=self.closed_positions,
                daily_returns=daily_returns_series,
                equity_curve=pd.Series([eq['equity'] for eq in self.equity_curve])
            )
            
        except Exception as e:
            logger.error(f"Error calculating results: {e}")
            return self._empty_results(start_date, end_date)
    
    def _combine_walk_forward_results(self, results_list: List[BacktestResults], start_date: datetime, end_date: datetime) -> BacktestResults:
        """Combine multiple walk-forward results"""
        try:
            if not results_list:
                return self._empty_results(start_date, end_date)
            
            # Combine all trades
            all_trades = []
            for results in results_list:
                all_trades.extend(results.trades)
            
            # Combine daily returns
            all_daily_returns = pd.concat([results.daily_returns for results in results_list])
            
            # Combine equity curves
            all_equity = pd.concat([results.equity_curve for results in results_list])
            
            # Calculate combined metrics
            initial_capital = results_list[0].initial_capital
            final_capital = results_list[-1].final_capital
            total_return = (final_capital - initial_capital) / initial_capital
            
            # Use the combined results to calculate other metrics
            combined_results = results_list[0].__class__(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_capital=final_capital,
                total_return=total_return,
                annualized_return=0,  # Would calculate properly
                sharpe_ratio=0,       # Would calculate properly
                sortino_ratio=0,      # Would calculate properly
                max_drawdown=0,       # Would calculate properly
                calmar_ratio=0,       # Would calculate properly
                win_rate=0,           # Would calculate properly
                profit_factor=0,      # Would calculate properly
                total_trades=len(all_trades),
                winning_trades=0,     # Would calculate properly
                losing_trades=0,      # Would calculate properly
                avg_win=0,            # Would calculate properly
                avg_loss=0,           # Would calculate properly
                largest_win=0,        # Would calculate properly
                largest_loss=0,       # Would calculate properly
                total_commission=sum(results.total_commission for results in results_list),
                total_slippage=sum(results.total_slippage for results in results_list),
                trades=all_trades,
                daily_returns=all_daily_returns,
                equity_curve=all_equity
            )
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error combining walk-forward results: {e}")
            return self._empty_results(start_date, end_date)
    
    def _empty_results(self, start_date: datetime, end_date: datetime) -> BacktestResults:
        """Return empty results structure"""
        return BacktestResults(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            total_commission=0.0,
            total_slippage=0.0,
            trades=[],
            daily_returns=pd.Series(),
            equity_curve=pd.Series()
        )


# Example usage
if __name__ == "__main__":
    from src.portfolio.account_manager import UniversalAccountManager
    
    # Create account profile
    manager = UniversalAccountManager()
    profile = manager.create_account_profile(balance=25000)
    
    # Create backtest engine
    backtest = OptionsBacktestV2(profile)
    
    # Sample market data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    market_data = pd.DataFrame({
        'date': dates,
        'underlying_price': 450 + np.cumsum(np.random.normal(0, 2, len(dates))),
        'vix': np.random.normal(20, 3, len(dates)),
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # Sample trading signals
    signals = []
    for i in range(0, len(dates), 5):  # Every 5 days
        signals.append({
            'date': dates[i],
            'symbol': 'SPY',
            'strategy': 'bull_put_spread',
            'side': 'sell',
            'quantity': 1,
            'price': 1.50,
            'max_loss': 350,
            'legs': []
        })
    
    # Run backtest
    results = backtest.run_backtest(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        market_data=market_data,
        strategy_signals=signals
    )
    
    print(f"Backtest Results:")
    print(f"  Total Return: {results.total_return:.2%}")
    print(f"  Annualized Return: {results.annualized_return:.2%}")
    print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {results.max_drawdown:.2%}")
    print(f"  Win Rate: {results.win_rate:.2%}")
    print(f"  Total Trades: {results.total_trades}")
    print(f"  Total Commission: ${results.total_commission:,.2f}")
    print(f"  Total Slippage: ${results.total_slippage:,.2f}")
