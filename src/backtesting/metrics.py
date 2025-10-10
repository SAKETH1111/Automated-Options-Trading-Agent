"""
Performance Metrics Module
Calculate comprehensive performance metrics for backtests
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime
from loguru import logger


class PerformanceMetrics:
    """
    Calculate trading performance metrics
    Includes: Sharpe, Sortino, Calmar, Win Rate, Profit Factor, etc.
    """
    
    def __init__(self):
        """Initialize performance metrics calculator"""
        self.risk_free_rate = 0.05  # 5% annual
        logger.info("Performance Metrics calculator initialized")
    
    def calculate_all_metrics(
        self,
        trades: List,
        equity_curve: List[Dict],
        starting_capital: float
    ) -> Dict:
        """
        Calculate all performance metrics
        
        Args:
            trades: List of BacktestTrade objects
            equity_curve: List of equity snapshots
            starting_capital: Starting capital
            
        Returns:
            Dictionary with all metrics
        """
        if not trades:
            return self._empty_metrics()
        
        metrics = {}
        
        # Basic metrics
        metrics.update(self._calculate_basic_metrics(trades))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(trades, equity_curve, starting_capital))
        
        # Return metrics
        metrics.update(self._calculate_return_metrics(trades, equity_curve, starting_capital))
        
        # Consistency metrics
        metrics.update(self._calculate_consistency_metrics(trades))
        
        return metrics
    
    def _calculate_basic_metrics(self, trades: List) -> Dict:
        """Calculate basic trading metrics"""
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.realized_pnl > 0)
        losing_trades = sum(1 for t in trades if t.realized_pnl < 0)
        breakeven_trades = total_trades - winning_trades - losing_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        loss_rate = losing_trades / total_trades if total_trades > 0 else 0
        
        # P&L
        total_pnl = sum(t.realized_pnl for t in trades)
        wins = [t.realized_pnl for t in trades if t.realized_pnl > 0]
        losses = [t.realized_pnl for t in trades if t.realized_pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'breakeven_trades': breakeven_trades,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }
    
    def _calculate_risk_metrics(
        self,
        trades: List,
        equity_curve: List[Dict],
        starting_capital: float
    ) -> Dict:
        """Calculate risk-adjusted metrics"""
        # Extract returns
        returns = [t.realized_pnl_pct for t in trades if t.realized_pnl_pct]
        
        if not returns:
            return {
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0
            }
        
        # Sharpe Ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (mean_return - self.risk_free_rate) / std_return if std_return > 0 else 0
        
        # Sortino Ratio (only downside deviation)
        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns) if downside_returns else std_return
        sortino_ratio = (mean_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Max Drawdown
        equity_values = [e['equity'] for e in equity_curve]
        max_dd, max_dd_pct = self._calculate_max_drawdown(equity_values)
        
        # Calmar Ratio
        annual_return = mean_return * 252  # Annualize
        calmar_ratio = annual_return / max_dd_pct if max_dd_pct > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'volatility': std_return
        }
    
    def _calculate_return_metrics(
        self,
        trades: List,
        equity_curve: List[Dict],
        starting_capital: float
    ) -> Dict:
        """Calculate return metrics"""
        if not equity_curve:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'cagr': 0,
                'avg_monthly_return': 0
            }
        
        ending_capital = equity_curve[-1]['equity']
        total_return = ending_capital - starting_capital
        total_return_pct = (total_return / starting_capital) * 100
        
        # Calculate CAGR (if we have date range)
        if len(equity_curve) > 1:
            start_date = equity_curve[0]['timestamp']
            end_date = equity_curve[-1]['timestamp']
            years = (end_date - start_date).days / 365.0
            
            if years > 0:
                cagr = (((ending_capital / starting_capital) ** (1 / years)) - 1) * 100
            else:
                cagr = 0
        else:
            cagr = 0
        
        # Average monthly return (simplified)
        avg_monthly_return = total_return_pct / 12 if len(equity_curve) > 30 else 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'cagr': cagr,
            'avg_monthly_return': avg_monthly_return,
            'ending_capital': ending_capital
        }
    
    def _calculate_consistency_metrics(self, trades: List) -> Dict:
        """Calculate consistency metrics"""
        if not trades:
            return {
                'avg_days_held': 0,
                'consecutive_wins': 0,
                'consecutive_losses': 0,
                'avg_win_streak': 0,
                'avg_loss_streak': 0
            }
        
        # Average days held
        avg_days_held = np.mean([t.days_held for t in trades])
        
        # Streaks
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        win_streaks = []
        loss_streaks = []
        
        for trade in trades:
            if trade.realized_pnl > 0:
                current_win_streak += 1
                if current_loss_streak > 0:
                    loss_streaks.append(current_loss_streak)
                    current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif trade.realized_pnl < 0:
                current_loss_streak += 1
                if current_win_streak > 0:
                    win_streaks.append(current_win_streak)
                    current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        avg_win_streak = np.mean(win_streaks) if win_streaks else 0
        avg_loss_streak = np.mean(loss_streaks) if loss_streaks else 0
        
        return {
            'avg_days_held': avg_days_held,
            'consecutive_wins': max_win_streak,
            'consecutive_losses': max_loss_streak,
            'avg_win_streak': avg_win_streak,
            'avg_loss_streak': avg_loss_streak
        }
    
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
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics"""
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }

