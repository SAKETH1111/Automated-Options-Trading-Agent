"""
Backtest Reporter Module
Generate comprehensive backtest reports and visualizations
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
from loguru import logger

from .engine import BacktestResult, BacktestTrade


class BacktestReporter:
    """
    Generate backtest reports and visualizations
    """
    
    def __init__(self):
        """Initialize backtest reporter"""
        logger.info("Backtest Reporter initialized")
    
    def generate_text_report(self, result: BacktestResult, strategy_name: str = "") -> str:
        """
        Generate comprehensive text report
        
        Args:
            result: Backtest result
            strategy_name: Name of strategy
            
        Returns:
            Formatted text report
        """
        report = []
        report.append("=" * 70)
        report.append(f"  📊 BACKTEST REPORT{' - ' + strategy_name if strategy_name else ''}")
        report.append("=" * 70)
        report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        # Capital Summary
        report.append("\n💰 CAPITAL SUMMARY:")
        report.append(f"  Starting Capital: ${result.starting_capital:,.2f}")
        report.append(f"  Ending Capital:   ${result.ending_capital:,.2f}")
        report.append(f"  Total Return:     ${result.total_return:,.2f} ({result.total_return_pct:+.2f}%)")
        
        # Trade Statistics
        report.append("\n📊 TRADE STATISTICS:")
        report.append(f"  Total Trades:     {result.total_trades}")
        report.append(f"  Winning Trades:   {result.winning_trades} ({result.win_rate:.1%})")
        report.append(f"  Losing Trades:    {result.losing_trades}")
        report.append(f"  Average Days Held: {result.avg_days_held:.1f} days")
        
        # P&L Analysis
        report.append("\n💵 P&L ANALYSIS:")
        report.append(f"  Total P&L:        ${result.total_pnl:,.2f}")
        report.append(f"  Average Win:      ${result.avg_win:,.2f}")
        report.append(f"  Average Loss:     ${result.avg_loss:,.2f}")
        report.append(f"  Profit Factor:    {result.profit_factor:.2f}")
        
        # Risk Metrics
        report.append("\n📉 RISK METRICS:")
        report.append(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        report.append(f"  Max Drawdown:     ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.2f}%)")
        
        # Performance Grade
        grade = self._calculate_grade(result)
        report.append(f"\n🎯 PERFORMANCE GRADE: {grade}")
        
        # Recommendations
        recommendations = self._generate_recommendations(result)
        if recommendations:
            report.append("\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                report.append(f"  • {rec}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def generate_trade_log(self, trades: List[BacktestTrade]) -> str:
        """Generate detailed trade log"""
        if not trades:
            return "No trades executed"
        
        log = []
        log.append("\n📋 TRADE LOG:")
        log.append("=" * 100)
        log.append(f"{'#':<4} {'Date':<12} {'Strategy':<20} {'Entry':<8} {'Exit':<8} {'P&L':<12} {'Days':<5} {'Reason':<15}")
        log.append("=" * 100)
        
        for i, trade in enumerate(trades, 1):
            entry_date = trade.entry_date.strftime('%Y-%m-%d')
            exit_date = trade.exit_date.strftime('%Y-%m-%d') if trade.exit_date else 'OPEN'
            pnl_str = f"${trade.realized_pnl:+,.2f}"
            
            log.append(
                f"{i:<4} {entry_date:<12} {trade.strategy:<20} "
                f"${trade.entry_price:<7.2f} ${trade.exit_price:<7.2f} "
                f"{pnl_str:<12} {trade.days_held:<5} {trade.exit_reason:<15}"
            )
        
        log.append("=" * 100)
        
        return "\n".join(log)
    
    def generate_monthly_summary(self, trades: List[BacktestTrade]) -> str:
        """Generate monthly performance summary"""
        if not trades:
            return "No trades to summarize"
        
        # Group trades by month
        monthly_data = {}
        
        for trade in trades:
            if not trade.exit_date:
                continue
            
            month_key = trade.exit_date.strftime('%Y-%m')
            
            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    'trades': 0,
                    'wins': 0,
                    'pnl': 0
                }
            
            monthly_data[month_key]['trades'] += 1
            if trade.realized_pnl > 0:
                monthly_data[month_key]['wins'] += 1
            monthly_data[month_key]['pnl'] += trade.realized_pnl
        
        # Generate report
        report = []
        report.append("\n📅 MONTHLY SUMMARY:")
        report.append("=" * 70)
        report.append(f"{'Month':<10} {'Trades':<8} {'Wins':<8} {'Win Rate':<12} {'P&L':<15}")
        report.append("=" * 70)
        
        for month, data in sorted(monthly_data.items()):
            win_rate = data['wins'] / data['trades'] if data['trades'] > 0 else 0
            report.append(
                f"{month:<10} {data['trades']:<8} {data['wins']:<8} "
                f"{win_rate:<11.1%} ${data['pnl']:+,.2f}"
            )
        
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def _calculate_grade(self, result: BacktestResult) -> str:
        """Calculate performance grade"""
        score = 0
        
        # Win rate (30 points)
        if result.win_rate >= 0.70:
            score += 30
        elif result.win_rate >= 0.60:
            score += 25
        elif result.win_rate >= 0.50:
            score += 15
        else:
            score += 5
        
        # Sharpe ratio (30 points)
        if result.sharpe_ratio >= 2.0:
            score += 30
        elif result.sharpe_ratio >= 1.5:
            score += 25
        elif result.sharpe_ratio >= 1.0:
            score += 15
        else:
            score += 5
        
        # Profit factor (20 points)
        if result.profit_factor >= 2.0:
            score += 20
        elif result.profit_factor >= 1.5:
            score += 15
        elif result.profit_factor >= 1.2:
            score += 10
        else:
            score += 5
        
        # Max drawdown (20 points)
        if result.max_drawdown_pct <= 10:
            score += 20
        elif result.max_drawdown_pct <= 20:
            score += 15
        elif result.max_drawdown_pct <= 30:
            score += 10
        else:
            score += 5
        
        # Assign grade
        if score >= 90:
            return "A+ (Excellent)"
        elif score >= 80:
            return "A (Very Good)"
        elif score >= 70:
            return "B (Good)"
        elif score >= 60:
            return "C (Acceptable)"
        else:
            return "D (Needs Improvement)"
    
    def _generate_recommendations(self, result: BacktestResult) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        # Win rate recommendations
        if result.win_rate < 0.60:
            recommendations.append("Win rate below 60% - consider tightening entry criteria")
        
        # Sharpe ratio recommendations
        if result.sharpe_ratio < 1.0:
            recommendations.append("Sharpe ratio below 1.0 - strategy may be too risky")
        
        # Profit factor recommendations
        if result.profit_factor < 1.5:
            recommendations.append("Profit factor below 1.5 - improve risk/reward ratio")
        
        # Drawdown recommendations
        if result.max_drawdown_pct > 20:
            recommendations.append("Max drawdown over 20% - implement better risk management")
        
        # Positive recommendations
        if result.win_rate >= 0.70 and result.sharpe_ratio >= 1.5:
            recommendations.append("Excellent performance - strategy is ready for paper trading")
        
        return recommendations
    
    def export_to_csv(self, trades: List[BacktestTrade], filename: str):
        """Export trades to CSV"""
        if not trades:
            logger.warning("No trades to export")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'symbol': t.symbol,
            'strategy': t.strategy,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'quantity': t.quantity,
            'realized_pnl': t.realized_pnl,
            'realized_pnl_pct': t.realized_pnl_pct,
            'days_held': t.days_held,
            'exit_reason': t.exit_reason
        } for t in trades])
        
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(trades)} trades to {filename}")

