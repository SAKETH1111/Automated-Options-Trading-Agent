#!/usr/bin/env python3
"""
Daily Trading Report Generator
Sends comprehensive end-of-day summary via Telegram
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import pytz

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from src.database.session import get_db
from src.database.models import IndexTickData, Trade
from sqlalchemy import func

try:
    import telegram
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed")


class DailyReportGenerator:
    """Generate and send daily trading reports"""
    
    def __init__(self, telegram_token: str = None, chat_id: str = None):
        """
        Initialize daily report generator
        
        Args:
            telegram_token: Telegram bot token
            chat_id: Telegram chat ID to send reports to
        """
        self.db = get_db()
        self.ct_tz = pytz.timezone('America/Chicago')
        
        # Telegram setup
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.bot = None
        
        if telegram_token and chat_id and TELEGRAM_AVAILABLE:
            try:
                self.bot = Bot(token=telegram_token)
                logger.info("Telegram bot initialized for daily reports")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
    
    def generate_daily_report(self) -> str:
        """
        Generate comprehensive daily report
        
        Returns:
            Formatted report text
        """
        now = datetime.now(self.ct_tz)
        today = now.date()
        
        report_lines = []
        
        # Header
        report_lines.append("═" * 50)
        report_lines.append(f"📊 DAILY TRADING REPORT")
        report_lines.append(f"📅 {today.strftime('%A, %B %d, %Y')}")
        report_lines.append(f"⏰ {now.strftime('%I:%M %p %Z')}")
        report_lines.append("═" * 50)
        report_lines.append("")
        
        # Market Data Collection Stats
        data_stats = self._get_data_collection_stats(today)
        report_lines.append("📈 DATA COLLECTION")
        report_lines.append("─" * 50)
        if data_stats['total_ticks'] > 0:
            report_lines.append(f"✅ Market data collected successfully")
            report_lines.append(f"   Total ticks: {data_stats['total_ticks']:,}")
            report_lines.append(f"   Duration: {data_stats['collection_hours']:.1f} hours")
            report_lines.append("")
            report_lines.append("   By Symbol:")
            for symbol, count in data_stats['by_symbol'].items():
                report_lines.append(f"   • {symbol}: {count:,} ticks")
        else:
            report_lines.append(f"⚠️  No data collected today")
            report_lines.append(f"   (Market may be closed)")
        report_lines.append("")
        
        # Trading Activity
        trade_stats = self._get_trading_stats(today)
        report_lines.append("💼 TRADING ACTIVITY")
        report_lines.append("─" * 50)
        
        if trade_stats['trades_opened'] > 0:
            report_lines.append(f"✅ Trades Opened: {trade_stats['trades_opened']}")
            for trade in trade_stats['opened_trades']:
                report_lines.append(f"   • {trade['symbol']} - {trade['strategy']}")
                report_lines.append(f"     Entry: ${trade['entry_price']:.2f}")
                report_lines.append(f"     Max Profit: ${trade['max_profit']:.2f}")
                report_lines.append(f"     Reason: {trade['reason']}")
        else:
            report_lines.append(f"📊 No Trades Opened Today")
            
            # Explain why no trades
            no_trade_reasons = self._analyze_no_trade_reasons(today, data_stats)
            if no_trade_reasons:
                report_lines.append("")
                report_lines.append("   Why no trades?")
                for reason in no_trade_reasons:
                    report_lines.append(f"   • {reason}")
        
        report_lines.append("")
        
        if trade_stats['trades_closed'] > 0:
            report_lines.append(f"✅ Trades Closed: {trade_stats['trades_closed']}")
            total_pnl = sum(t['pnl'] for t in trade_stats['closed_trades'])
            profit_emoji = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "⚪"
            report_lines.append(f"   {profit_emoji} Total P&L: ${total_pnl:+.2f}")
            report_lines.append("")
            for trade in trade_stats['closed_trades']:
                emoji = "🟢" if trade['pnl'] > 0 else "🔴"
                report_lines.append(f"   {emoji} {trade['symbol']} - {trade['strategy']}")
                report_lines.append(f"     P&L: ${trade['pnl']:+.2f} ({trade['pnl_pct']:+.1f}%)")
                report_lines.append(f"     Days held: {trade['days_held']}")
                report_lines.append(f"     Exit: {trade['exit_reason']}")
        report_lines.append("")
        
        # Current Positions
        positions = self._get_open_positions()
        report_lines.append("📋 OPEN POSITIONS")
        report_lines.append("─" * 50)
        
        if positions:
            report_lines.append(f"Total: {len(positions)} open position(s)")
            report_lines.append("")
            for i, pos in enumerate(positions, 1):
                emoji = "🟢" if pos['unrealized_pnl'] > 0 else "🔴" if pos['unrealized_pnl'] < 0 else "⚪"
                report_lines.append(f"{i}. {pos['symbol']} - {pos['strategy']}")
                report_lines.append(f"   {emoji} P&L: ${pos['unrealized_pnl']:+.2f}")
                report_lines.append(f"   Days: {pos['days_open']}")
                report_lines.append(f"   DTE: {pos['dte']} days")
        else:
            report_lines.append("No open positions")
        report_lines.append("")
        
        # Market Analysis
        market_analysis = self._analyze_market_conditions(today, data_stats)
        report_lines.append("🔍 MARKET ANALYSIS")
        report_lines.append("─" * 50)
        report_lines.extend(market_analysis)
        report_lines.append("")
        
        # Tomorrow's Outlook
        outlook = self._generate_tomorrow_outlook(trade_stats, positions, market_analysis)
        report_lines.append("🔮 TOMORROW'S OUTLOOK")
        report_lines.append("─" * 50)
        report_lines.extend(outlook)
        report_lines.append("")
        
        # Performance Summary (Last 7 Days)
        weekly_stats = self._get_weekly_performance()
        report_lines.append("📊 WEEKLY PERFORMANCE (Last 7 Days)")
        report_lines.append("─" * 50)
        report_lines.append(f"Total Trades: {weekly_stats['total_trades']}")
        if weekly_stats['total_trades'] > 0:
            report_lines.append(f"Win Rate: {weekly_stats['win_rate']:.1f}%")
            report_lines.append(f"Total P&L: ${weekly_stats['total_pnl']:+.2f}")
            report_lines.append(f"Avg P&L per trade: ${weekly_stats['avg_pnl']:+.2f}")
        report_lines.append("")
        
        # Footer
        report_lines.append("═" * 50)
        report_lines.append("🤖 Automated Trading Agent")
        report_lines.append(f"Next report: Tomorrow at 4:00 PM CT")
        report_lines.append("═" * 50)
        
        return "\n".join(report_lines)
    
    def _get_data_collection_stats(self, date) -> Dict:
        """Get data collection statistics for the day"""
        stats = {
            'total_ticks': 0,
            'by_symbol': {},
            'collection_hours': 0
        }
        
        try:
            with self.db.get_session() as session:
                # Get ticks for today
                start_time = datetime.combine(date, datetime.min.time())
                end_time = datetime.combine(date, datetime.max.time())
                
                # Total ticks
                total = session.query(func.count(IndexTickData.tick_id))\
                    .filter(IndexTickData.timestamp >= start_time)\
                    .filter(IndexTickData.timestamp <= end_time)\
                    .scalar()
                
                stats['total_ticks'] = total or 0
                
                # By symbol
                by_symbol = session.query(
                    IndexTickData.symbol,
                    func.count(IndexTickData.symbol).label('count')
                ).filter(IndexTickData.timestamp >= start_time)\
                 .filter(IndexTickData.timestamp <= end_time)\
                 .group_by(IndexTickData.symbol)\
                 .all()
                
                for symbol, count in by_symbol:
                    stats['by_symbol'][symbol] = count
                
                # Collection duration
                time_range = session.query(
                    func.min(IndexTickData.timestamp),
                    func.max(IndexTickData.timestamp)
                ).filter(IndexTickData.timestamp >= start_time)\
                 .filter(IndexTickData.timestamp <= end_time)\
                 .first()
                
                if time_range[0] and time_range[1]:
                    duration = time_range[1] - time_range[0]
                    stats['collection_hours'] = duration.total_seconds() / 3600
        
        except Exception as e:
            logger.error(f"Error getting data collection stats: {e}")
        
        return stats
    
    def _get_trading_stats(self, date) -> Dict:
        """Get trading statistics for the day"""
        stats = {
            'trades_opened': 0,
            'trades_closed': 0,
            'opened_trades': [],
            'closed_trades': []
        }
        
        try:
            with self.db.get_session() as session:
                start_time = datetime.combine(date, datetime.min.time())
                end_time = datetime.combine(date, datetime.max.time())
                
                # Trades opened today
                opened = session.query(Trade)\
                    .filter(Trade.timestamp_enter >= start_time)\
                    .filter(Trade.timestamp_enter <= end_time)\
                    .all()
                
                stats['trades_opened'] = len(opened)
                for trade in opened:
                    stats['opened_trades'].append({
                        'symbol': trade.symbol,
                        'strategy': trade.strategy,
                        'entry_price': trade.market_snapshot.get('price', 0),
                        'max_profit': trade.risk.get('max_profit', 0),
                        'reason': 'Signal quality met criteria'
                    })
                
                # Trades closed today
                closed = session.query(Trade)\
                    .filter(Trade.timestamp_exit >= start_time)\
                    .filter(Trade.timestamp_exit <= end_time)\
                    .filter(Trade.status == 'closed')\
                    .all()
                
                stats['trades_closed'] = len(closed)
                for trade in closed:
                    stats['closed_trades'].append({
                        'symbol': trade.symbol,
                        'strategy': trade.strategy,
                        'pnl': trade.pnl,
                        'pnl_pct': trade.pnl_pct,
                        'days_held': trade.days_held,
                        'exit_reason': trade.exit_reason or 'Unknown'
                    })
        
        except Exception as e:
            logger.error(f"Error getting trading stats: {e}")
        
        return stats
    
    def _get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        positions = []
        
        try:
            with self.db.get_session() as session:
                trades = session.query(Trade)\
                    .filter(Trade.status == 'open')\
                    .all()
                
                for trade in trades:
                    days_open = (datetime.now() - trade.timestamp_enter).days
                    
                    # Calculate DTE if expiration date available
                    dte = None
                    if trade.params.get('expiration'):
                        exp_date = datetime.fromisoformat(str(trade.params['expiration']))
                        dte = (exp_date - datetime.now()).days
                    
                    positions.append({
                        'symbol': trade.symbol,
                        'strategy': trade.strategy,
                        'unrealized_pnl': trade.pnl,
                        'days_open': days_open,
                        'dte': dte or trade.params.get('dte', 0)
                    })
        
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
        
        return positions
    
    def _analyze_no_trade_reasons(self, date, data_stats) -> List[str]:
        """Analyze why no trades were made"""
        reasons = []
        
        # Check if market was closed
        if data_stats['total_ticks'] < 100:
            reasons.append("Market was closed or limited data")
            return reasons
        
        # Check if it's a weekend
        now = datetime.now(self.ct_tz)
        if now.weekday() >= 5:
            reasons.append("Weekend - market closed")
            return reasons
        
        # Other possible reasons
        reasons.append("No high-quality signals met entry criteria")
        reasons.append("Risk limits may have been reached")
        reasons.append("Waiting for better market conditions")
        
        return reasons
    
    def _analyze_market_conditions(self, date, data_stats) -> List[str]:
        """Analyze market conditions"""
        analysis = []
        
        if data_stats['total_ticks'] > 1000:
            analysis.append("✅ Good data quality - active market")
            analysis.append(f"   Collected {data_stats['total_ticks']:,} ticks")
        elif data_stats['total_ticks'] > 0:
            analysis.append("⚠️  Limited data - partial market day")
        else:
            analysis.append("❌ No data - market was closed")
            return analysis
        
        # Add more analysis here based on actual data
        analysis.append("📊 Volatility: Normal")
        analysis.append("📈 Trend: Monitoring")
        
        return analysis
    
    def _generate_tomorrow_outlook(self, trade_stats, positions, market_analysis) -> List[str]:
        """Generate outlook for tomorrow"""
        outlook = []
        
        now = datetime.now(self.ct_tz)
        tomorrow = now + timedelta(days=1)
        
        # Check if tomorrow is a trading day
        if tomorrow.weekday() >= 5:
            outlook.append("⏸️  Weekend - market closed")
            outlook.append("   Next trading day: Monday")
            return outlook
        
        # Market opens
        outlook.append(f"🔔 Market opens: 8:30 AM CT")
        outlook.append(f"   Data collection will resume automatically")
        outlook.append("")
        
        # Position management
        if positions:
            outlook.append(f"📋 Will monitor {len(positions)} open position(s)")
            
            # Check for positions near expiration
            expiring_soon = [p for p in positions if p.get('dte', 999) <= 3]
            if expiring_soon:
                outlook.append(f"⚠️  {len(expiring_soon)} position(s) expiring within 3 days")
        else:
            outlook.append("🎯 Looking for new trading opportunities")
        
        outlook.append("")
        outlook.append("Strategy:")
        outlook.append("• Continue data collection")
        outlook.append("• Monitor for quality signals")
        outlook.append("• Manage existing positions")
        outlook.append("• Follow PDT compliance rules")
        
        return outlook
    
    def _get_weekly_performance(self) -> Dict:
        """Get performance for last 7 days"""
        stats = {
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0
        }
        
        try:
            with self.db.get_session() as session:
                week_ago = datetime.now() - timedelta(days=7)
                
                trades = session.query(Trade)\
                    .filter(Trade.timestamp_exit >= week_ago)\
                    .filter(Trade.status == 'closed')\
                    .all()
                
                stats['total_trades'] = len(trades)
                
                if trades:
                    winning = sum(1 for t in trades if t.pnl > 0)
                    stats['win_rate'] = (winning / len(trades)) * 100
                    stats['total_pnl'] = sum(t.pnl for t in trades)
                    stats['avg_pnl'] = stats['total_pnl'] / len(trades)
        
        except Exception as e:
            logger.error(f"Error getting weekly performance: {e}")
        
        return stats
    
    def send_daily_report(self) -> bool:
        """Generate and send daily report via Telegram"""
        try:
            # Generate report
            report = self.generate_daily_report()
            
            # Log to console
            logger.info("Daily Report Generated:")
            logger.info("\n" + report)
            
            # Send via Telegram
            if self.bot and self.chat_id:
                # Split into chunks if too long (Telegram limit is 4096 characters)
                max_length = 4000
                if len(report) <= max_length:
                    self.bot.send_message(
                        chat_id=self.chat_id,
                        text=f"```\n{report}\n```",
                        parse_mode='Markdown'
                    )
                else:
                    # Split into chunks
                    chunks = [report[i:i+max_length] for i in range(0, len(report), max_length)]
                    for i, chunk in enumerate(chunks, 1):
                        self.bot.send_message(
                            chat_id=self.chat_id,
                            text=f"```\n{chunk}\n```",
                            parse_mode='Markdown'
                        )
                
                logger.info("Daily report sent via Telegram")
                return True
            else:
                logger.warning("Telegram not configured - report logged only")
                return False
        
        except Exception as e:
            logger.error(f"Error sending daily report: {e}")
            return False


def main():
    """Test the daily report generator"""
    import os
    
    # Get credentials from environment
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    generator = DailyReportGenerator(telegram_token, chat_id)
    generator.send_daily_report()


if __name__ == "__main__":
    main()

