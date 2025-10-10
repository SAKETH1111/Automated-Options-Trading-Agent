"""
Backtesting Engine
Core engine for testing trading strategies on historical data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class BacktestTrade:
    """Represents a trade in the backtest"""
    entry_date: datetime
    exit_date: Optional[datetime] = None
    symbol: str = ""
    strategy: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 1
    max_profit: float = 0.0
    max_loss: float = 0.0
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    days_held: int = 0
    exit_reason: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    trades: List[BacktestTrade] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    avg_days_held: float = 0.0
    starting_capital: float = 10000.0
    ending_capital: float = 10000.0
    metadata: Dict = field(default_factory=dict)


class BacktestEngine:
    """
    Core backtesting engine
    Replays historical data and tests trading strategies
    """
    
    def __init__(
        self,
        starting_capital: float = 10000.0,
        commission: float = 0.65,
        slippage: float = 0.01
    ):
        """
        Initialize backtest engine
        
        Args:
            starting_capital: Starting capital for backtest
            commission: Commission per contract
            slippage: Slippage per trade (as decimal)
        """
        self.starting_capital = starting_capital
        self.commission = commission
        self.slippage = slippage
        self.current_capital = starting_capital
        self.trades: List[BacktestTrade] = []
        self.open_positions: List[BacktestTrade] = []
        self.equity_curve: List[Dict] = []
        
        logger.info(f"Backtest Engine initialized with ${starting_capital:,.2f}")
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        strategy_params: Dict = None
    ) -> BacktestResult:
        """
        Run backtest on historical data
        
        Args:
            data: Historical price data (DataFrame with timestamp, price, etc.)
            strategy_func: Strategy function that generates signals
            strategy_params: Parameters for the strategy
            
        Returns:
            BacktestResult with performance metrics
        """
        logger.info("Starting backtest...")
        
        if strategy_params is None:
            strategy_params = {}
        
        # Reset state
        self.current_capital = self.starting_capital
        self.trades = []
        self.open_positions = []
        self.equity_curve = []
        
        # Iterate through historical data
        for i in range(len(data)):
            current_bar = data.iloc[i]
            timestamp = current_bar.get('timestamp', datetime.now())
            
            # Update open positions
            self._update_open_positions(current_bar)
            
            # Check for exit signals
            self._check_exit_signals(current_bar)
            
            # Check for entry signals (only if we have capital)
            if self.current_capital > 0:
                signal = strategy_func(data.iloc[:i+1], strategy_params)
                
                if signal and signal.get('action') in ['BUY', 'SELL']:
                    self._execute_trade(signal, current_bar)
            
            # Record equity
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.current_capital + self._calculate_open_position_value(),
                'cash': self.current_capital,
                'open_positions': len(self.open_positions)
            })
        
        # Close any remaining positions
        self._close_all_positions(data.iloc[-1])
        
        # Calculate performance metrics
        result = self._calculate_results()
        
        logger.info(f"Backtest complete: {result.total_trades} trades, "
                   f"{result.win_rate:.1%} win rate, ${result.total_pnl:,.2f} P&L")
        
        return result
    
    def _execute_trade(self, signal: Dict, current_bar: pd.Series):
        """Execute a trade based on signal"""
        try:
            # Extract trade details from signal
            trade = BacktestTrade(
                entry_date=current_bar.get('timestamp', datetime.now()),
                symbol=signal.get('symbol', ''),
                strategy=signal.get('strategy', ''),
                entry_price=signal.get('entry_price', 0.0),
                quantity=signal.get('quantity', 1),
                max_profit=signal.get('max_profit', 0.0),
                max_loss=signal.get('max_loss', 0.0),
                metadata=signal.get('metadata', {})
            )
            
            # Calculate cost with commission and slippage
            entry_cost = trade.entry_price * trade.quantity * 100  # Options are per 100 shares
            total_cost = entry_cost + (self.commission * trade.quantity)
            
            # Apply slippage
            if signal.get('action') == 'BUY':
                total_cost *= (1 + self.slippage)
            else:
                total_cost *= (1 - self.slippage)
            
            # Check if we have enough capital
            required_capital = abs(trade.max_loss) if trade.max_loss else total_cost
            
            if required_capital > self.current_capital:
                logger.debug(f"Insufficient capital for trade: ${required_capital:,.2f} > ${self.current_capital:,.2f}")
                return
            
            # Execute trade
            self.current_capital -= required_capital
            self.open_positions.append(trade)
            
            logger.debug(f"Opened {trade.strategy} on {trade.symbol} at ${trade.entry_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _update_open_positions(self, current_bar: pd.Series):
        """Update open position values"""
        # This would update mark-to-market values
        # For now, we'll handle this in exit logic
        pass
    
    def _check_exit_signals(self, current_bar: pd.Series):
        """Check if any open positions should be closed"""
        current_price = current_bar.get('price', 0)
        current_date = current_bar.get('timestamp', datetime.now())
        
        positions_to_close = []
        
        for position in self.open_positions:
            # Check expiration
            if position.metadata.get('expiration'):
                if current_date >= position.metadata['expiration']:
                    positions_to_close.append((position, 'EXPIRATION', 0.0))
                    continue
            
            # Check stop loss
            if position.max_loss and position.realized_pnl <= -abs(position.max_loss) * 0.9:
                positions_to_close.append((position, 'STOP_LOSS', position.max_loss))
                continue
            
            # Check take profit (50% of max profit)
            if position.max_profit and position.realized_pnl >= position.max_profit * 0.5:
                positions_to_close.append((position, 'TAKE_PROFIT', position.max_profit * 0.5))
                continue
        
        # Close positions
        for position, reason, pnl in positions_to_close:
            self._close_position(position, current_bar, reason, pnl)
    
    def _close_position(
        self,
        position: BacktestTrade,
        current_bar: pd.Series,
        reason: str,
        pnl: float = None
    ):
        """Close a position"""
        try:
            position.exit_date = current_bar.get('timestamp', datetime.now())
            position.exit_reason = reason
            
            # Calculate P&L
            if pnl is not None:
                position.realized_pnl = pnl
            else:
                # Default: assume max profit at expiration if no loss
                position.realized_pnl = position.max_profit
            
            # Calculate percentage
            if position.max_loss:
                position.realized_pnl_pct = (position.realized_pnl / abs(position.max_loss)) * 100
            
            # Calculate days held
            position.days_held = (position.exit_date - position.entry_date).days
            
            # Return capital plus P&L
            returned_capital = abs(position.max_loss) + position.realized_pnl
            self.current_capital += returned_capital
            
            # Move to closed trades
            self.trades.append(position)
            self.open_positions.remove(position)
            
            logger.debug(f"Closed {position.strategy}: ${position.realized_pnl:.2f} ({reason})")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _close_all_positions(self, final_bar: pd.Series):
        """Close all remaining positions at end of backtest"""
        for position in list(self.open_positions):
            self._close_position(position, final_bar, 'END_OF_BACKTEST')
    
    def _calculate_open_position_value(self) -> float:
        """Calculate total value of open positions"""
        return sum(pos.max_loss for pos in self.open_positions)
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest performance metrics"""
        if not self.trades:
            return BacktestResult(
                starting_capital=self.starting_capital,
                ending_capital=self.current_capital
            )
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.realized_pnl > 0)
        losing_trades = sum(1 for t in self.trades if t.realized_pnl < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t.realized_pnl for t in self.trades)
        wins = [t.realized_pnl for t in self.trades if t.realized_pnl > 0]
        losses = [t.realized_pnl for t in self.trades if t.realized_pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Sharpe ratio (simplified)
        returns = [t.realized_pnl_pct for t in self.trades if t.realized_pnl_pct]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Drawdown
        equity_values = [e['equity'] for e in self.equity_curve]
        max_drawdown, max_drawdown_pct = self._calculate_max_drawdown(equity_values)
        
        # Returns
        ending_capital = self.current_capital
        total_return = ending_capital - self.starting_capital
        total_return_pct = (total_return / self.starting_capital) * 100
        
        # Average days held
        avg_days_held = np.mean([t.days_held for t in self.trades]) if self.trades else 0
        
        return BacktestResult(
            trades=self.trades,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            total_return=total_return,
            total_return_pct=total_return_pct,
            avg_days_held=avg_days_held,
            starting_capital=self.starting_capital,
            ending_capital=ending_capital
        )
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> tuple:
        """Calculate maximum drawdown"""
        if not equity_curve:
            return 0.0, 0.0
        
        peak = equity_curve[0]
        max_dd = 0
        max_dd_pct = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            drawdown = peak - value
            drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0
            
            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_pct = drawdown_pct
        
        return max_dd, max_dd_pct

