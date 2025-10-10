"""
Performance Tracker Module
Track real-time P&L and generate performance reports
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
from loguru import logger

from src.database.models import Trade


class PerformanceTracker:
    """
    Track trading performance in real-time
    Calculate daily/weekly/monthly P&L and metrics
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize performance tracker
        
        Args:
            db_session: Database session
        """
        self.db = db_session
        logger.info("Performance Tracker initialized")
    
    def get_daily_pnl(self, date: Optional[datetime] = None) -> Dict:
        """
        Get P&L for a specific day
        
        Args:
            date: Date to check (default: today)
            
        Returns:
            Daily P&L summary
        """
        if date is None:
            date = datetime.utcnow()
        
        try:
            # Get trades closed today
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            
            closed_trades = self.db.query(Trade).filter(
                Trade.timestamp_exit >= start_of_day,
                Trade.timestamp_exit < end_of_day,
                Trade.status == 'closed'
            ).all()
            
            total_pnl = sum(t.pnl for t in closed_trades)
            winning_trades = sum(1 for t in closed_trades if t.pnl > 0)
            losing_trades = sum(1 for t in closed_trades if t.pnl < 0)
            
            return {
                'date': date.strftime('%Y-%m-%d'),
                'total_trades': len(closed_trades),
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / len(closed_trades) if closed_trades else 0,
                'total_pnl': total_pnl,
                'trades': closed_trades
            }
            
        except Exception as e:
            logger.error(f"Error getting daily P&L: {e}")
            return {}
    
    def get_weekly_pnl(self) -> Dict:
        """Get P&L for current week"""
        try:
            # Get start of week
            today = datetime.utcnow()
            start_of_week = today - timedelta(days=today.weekday())
            start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
            
            closed_trades = self.db.query(Trade).filter(
                Trade.timestamp_exit >= start_of_week,
                Trade.status == 'closed'
            ).all()
            
            total_pnl = sum(t.pnl for t in closed_trades)
            winning_trades = sum(1 for t in closed_trades if t.pnl > 0)
            
            return {
                'week_start': start_of_week.strftime('%Y-%m-%d'),
                'total_trades': len(closed_trades),
                'winning_trades': winning_trades,
                'win_rate': winning_trades / len(closed_trades) if closed_trades else 0,
                'total_pnl': total_pnl
            }
            
        except Exception as e:
            logger.error(f"Error getting weekly P&L: {e}")
            return {}
    
    def get_monthly_pnl(self) -> Dict:
        """Get P&L for current month"""
        try:
            # Get start of month
            today = datetime.utcnow()
            start_of_month = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            closed_trades = self.db.query(Trade).filter(
                Trade.timestamp_exit >= start_of_month,
                Trade.status == 'closed'
            ).all()
            
            total_pnl = sum(t.pnl for t in closed_trades)
            winning_trades = sum(1 for t in closed_trades if t.pnl > 0)
            
            # Group by strategy
            by_strategy = {}
            for trade in closed_trades:
                strategy = trade.strategy
                if strategy not in by_strategy:
                    by_strategy[strategy] = {'trades': 0, 'pnl': 0}
                by_strategy[strategy]['trades'] += 1
                by_strategy[strategy]['pnl'] += trade.pnl
            
            return {
                'month': start_of_month.strftime('%Y-%m'),
                'total_trades': len(closed_trades),
                'winning_trades': winning_trades,
                'win_rate': winning_trades / len(closed_trades) if closed_trades else 0,
                'total_pnl': total_pnl,
                'by_strategy': by_strategy
            }
            
        except Exception as e:
            logger.error(f"Error getting monthly P&L: {e}")
            return {}
    
    def get_all_time_stats(self) -> Dict:
        """Get all-time trading statistics"""
        try:
            all_trades = self.db.query(Trade).filter(
                Trade.status == 'closed'
            ).all()
            
            if not all_trades:
                return {
                    'total_trades': 0,
                    'message': 'No closed trades yet'
                }
            
            total_trades = len(all_trades)
            winning_trades = sum(1 for t in all_trades if t.pnl > 0)
            losing_trades = sum(1 for t in all_trades if t.pnl < 0)
            
            total_pnl = sum(t.pnl for t in all_trades)
            wins = [t.pnl for t in all_trades if t.pnl > 0]
            losses = [t.pnl for t in all_trades if t.pnl < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Best and worst trades
            best_trade = max(all_trades, key=lambda t: t.pnl)
            worst_trade = min(all_trades, key=lambda t: t.pnl)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / total_trades,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'best_trade': {
                    'symbol': best_trade.symbol,
                    'strategy': best_trade.strategy,
                    'pnl': best_trade.pnl,
                    'date': best_trade.timestamp_exit
                },
                'worst_trade': {
                    'symbol': worst_trade.symbol,
                    'strategy': worst_trade.strategy,
                    'pnl': worst_trade.pnl,
                    'date': worst_trade.timestamp_exit
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting all-time stats: {e}")
            return {}
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("\n" + "=" * 70)
        report.append("  📊 TRADING PERFORMANCE REPORT")
        report.append("=" * 70)
        report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        # Daily P&L
        daily = self.get_daily_pnl()
        if daily:
            report.append("\n📅 TODAY'S PERFORMANCE:")
            report.append(f"  Trades: {daily['total_trades']}")
            report.append(f"  Win Rate: {daily['win_rate']:.1%}")
            report.append(f"  P&L: ${daily['total_pnl']:+,.2f}")
        
        # Weekly P&L
        weekly = self.get_weekly_pnl()
        if weekly:
            report.append("\n📅 THIS WEEK:")
            report.append(f"  Trades: {weekly['total_trades']}")
            report.append(f"  Win Rate: {weekly['win_rate']:.1%}")
            report.append(f"  P&L: ${weekly['total_pnl']:+,.2f}")
        
        # Monthly P&L
        monthly = self.get_monthly_pnl()
        if monthly:
            report.append("\n📅 THIS MONTH:")
            report.append(f"  Trades: {monthly['total_trades']}")
            report.append(f"  Win Rate: {monthly['win_rate']:.1%}")
            report.append(f"  P&L: ${monthly['total_pnl']:+,.2f}")
            
            if monthly.get('by_strategy'):
                report.append("\n  By Strategy:")
                for strategy, data in monthly['by_strategy'].items():
                    report.append(f"    {strategy}: {data['trades']} trades, ${data['pnl']:+,.2f}")
        
        # All-time stats
        all_time = self.get_all_time_stats()
        if all_time and all_time.get('total_trades', 0) > 0:
            report.append("\n📊 ALL-TIME STATISTICS:")
            report.append(f"  Total Trades: {all_time['total_trades']}")
            report.append(f"  Win Rate: {all_time['win_rate']:.1%}")
            report.append(f"  Total P&L: ${all_time['total_pnl']:+,.2f}")
            report.append(f"  Profit Factor: {all_time['profit_factor']:.2f}")
            report.append(f"  Avg Win: ${all_time['avg_win']:,.2f}")
            report.append(f"  Avg Loss: ${all_time['avg_loss']:,.2f}")
            
            report.append("\n  Best Trade:")
            best = all_time['best_trade']
            report.append(f"    {best['symbol']} {best['strategy']}: ${best['pnl']:+,.2f}")
            
            report.append("\n  Worst Trade:")
            worst = all_time['worst_trade']
            report.append(f"    {worst['symbol']} {worst['strategy']}: ${worst['pnl']:+,.2f}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)

